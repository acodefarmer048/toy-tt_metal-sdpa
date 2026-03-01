#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // 1. Args
    uint32_t q_addr = get_arg_val<uint32_t>(0);
    uint32_t k_addr = get_arg_val<uint32_t>(1);
    uint32_t v_addr = get_arg_val<uint32_t>(2);
    // uint32_t unused = get_arg_val<uint32_t>(3);
    uint32_t num_cores = get_arg_val<uint32_t>(4);
    uint32_t my_x = get_arg_val<uint32_t>(5);
    uint32_t my_y = get_arg_val<uint32_t>(6);
    uint32_t prev_core_x = get_arg_val<uint32_t>(7);
    uint32_t prev_core_y = get_arg_val<uint32_t>(8);

    // Compile time args
    constexpr uint32_t sender_sem_addr = get_compile_time_arg_val(0);
    constexpr uint32_t receiver_sem_addr = get_compile_time_arg_val(1);
    constexpr uint32_t packed_scaler_val = get_compile_time_arg_val(2);

    // Generate Scaler
    constexpr uint32_t cb_scaler = tt::CBIndex::c_3;
    generate_reduce_scaler(cb_scaler, packed_scaler_val);

    // CB Allocations
    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    
    // Block Size (assuming uniform)
    // 8 tiles * 2048 bytes
    constexpr uint32_t block_bytes = 16384; 
    constexpr uint32_t num_tiles = 8;
    
    
    // NOC Addresses for Local Slots (Loopback)
    uint64_t my_q_slot_noc = get_noc_addr(my_x, my_y, q_addr);
    uint64_t my_k_slot_noc = get_noc_addr(my_x, my_y, k_addr);
    uint64_t my_v_slot_noc = get_noc_addr(my_x, my_y, v_addr);
    uint64_t my_sender_sem_noc = get_noc_addr(my_x, my_y, sender_sem_addr);

    // 2. Load Initial Local Data (Step 0)
    // We assume data is already present at `q_addr`, `k_addr`, `v_addr` (Storage L1)
    // Load Slot -> CB
    
    // Load Q
    cb_reserve_back(cb_q, num_tiles);
    uint32_t wr_ptr_q = get_write_ptr(cb_q);
    noc_async_read(my_q_slot_noc, wr_ptr_q, block_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_q, num_tiles);

    // Load K
    cb_reserve_back(cb_k, num_tiles);
    uint32_t wr_ptr_k = get_write_ptr(cb_k);
    noc_async_read(my_k_slot_noc, wr_ptr_k, block_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_k, num_tiles);

    // Load V
    cb_reserve_back(cb_v, num_tiles);
    uint32_t wr_ptr_v = get_write_ptr(cb_v);
    noc_async_read(my_v_slot_noc, wr_ptr_v, block_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_v, num_tiles);
    
    // Step 0 Data is ready in Slot (Assuming k_addr already has it). Signal Next Core (Sender Ready)
    // Note: If k_addr was loaded from DRAM previously, it is valid.
    noc_semaphore_inc(my_sender_sem_noc, 1);
    
    // 3. Ring Loop (Steps 1 to N-1)
    for (uint32_t step = 1; step < num_cores; ++step) {
        // A. Wait for Prev Core to have produced "Step S" data (which they signaled as Sem count = step)
        // Step 1: Wait for Sender >= 1.
        uint64_t prev_sender_sem_noc = get_noc_addr(prev_core_x, prev_core_y, sender_sem_addr);
        noc_semaphore_wait(prev_sender_sem_noc, step);

        // B. Wait for Next Core to have consumed PREVIOUS data from My Slot
        // We utilize receiver_sem to track consumption count
        // Step 1: Wait for Receiver >= 1 (Next Core read Step 0 data).
        uint64_t my_receiver_sem_noc = get_noc_addr(my_x, my_y, receiver_sem_addr);
        noc_semaphore_wait(my_receiver_sem_noc, step); 

        // C. READ: Prev Slot -> My CB
        // Wait, standard Ring: We read from Prev's Slot.
        uint64_t prev_k_slot = get_noc_addr(prev_core_x, prev_core_y, k_addr);
        uint64_t prev_v_slot = get_noc_addr(prev_core_x, prev_core_y, v_addr);
        
        cb_reserve_back(cb_k, num_tiles);
        cb_reserve_back(cb_v, num_tiles);
        wr_ptr_k = get_write_ptr(cb_k);
        wr_ptr_v = get_write_ptr(cb_v);
        
        noc_async_read(prev_k_slot, wr_ptr_k, block_bytes);
        noc_async_read(prev_v_slot, wr_ptr_v, block_bytes);
        
        // Ensure Read completes before Step D/E
        noc_async_read_barrier();
        
        // D. Signal Prev "I read your data" (Increment Prev Receiver Sem)
        uint64_t prev_receiver_sem_noc = get_noc_addr(prev_core_x, prev_core_y, receiver_sem_addr);
        noc_semaphore_inc(prev_receiver_sem_noc, 1);
        
        // E. WRITE: My CB -> My Slot (Pass to Next)
        // We overwrite My Slot with data JUST read from Prev.
        // This propagates the "bucket chain".
        noc_async_write(wr_ptr_k, my_k_slot_noc, block_bytes);
        noc_async_write(wr_ptr_v, my_v_slot_noc, block_bytes);
        
        // Ensure Write completes before signaling Next validity
        noc_async_write_barrier(); 
        
        // F. Signal Next "New Data Ready" (Increment My Sender Sem)
        // Next core waiting for Sender >= step + 1?
        // Step 1 output becomes Step 2 input for Next?
        // Wait. Step 1: Me reads K(-1). Writes to Slot.
        // Next Core (at Step 1) needs K(Next-1) which is K(Me).
        // My Slot now holds K(-1). Is that correct for Next?
        // Next Core is index (Me+1).
        // At Step 1, Next Core wants K((Me+1)-1) = K(Me).
        // But K(Me) was available at Step 0.
        // At Step 1, Next Core wants...
        // Ring Shift:
        // Step 0: [0, 1, 2, 3] -> Core 0 has 0.
        // Step 1: [3, 0, 1, 2] -> Core 0 has 3 (from Prev 3).
        // Core 1 has 0 (from Prev 0).
        // So at Step 1, Me (Core 0) has 3. Written to Slot.
        // Next (Core 1) needs... 0.
        // My Slot previously held 0 (at Step 0).
        // Next read 0 at Step 1 (of loop).
        // Ah.
        // Loop Step 1:
        //   Me reads 3. Writes 3 to Slot.
        //   Next reads ??
        //   If Next runs parallel Step 1:
        //   Next reads My Slot.
        //   If I overwrote 0 with 3 BEFORE Next read 0...
        //   Then Next reads 3. WRONG. Next needs 0.
        //   That's why we wait for `receiver_sem`.
        //   We wait for Next to read 0 before we write 3.
        //   Correct.
        //   After we write 3, we signal SenderReady (count goes to 2).
        //   Next Core (at Step 2) will wait for Sender >= 2. And read 3.
        //   Correct.
        
        noc_semaphore_inc(my_sender_sem_noc, 1);
        
        cb_push_back(cb_k, num_tiles);
        cb_push_back(cb_v, num_tiles);
    }
}

