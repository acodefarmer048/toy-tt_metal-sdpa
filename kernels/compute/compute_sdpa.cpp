#include <algorithm>
#include <cstdint>
#include "compute_kernel_api.h"
#include "debug/dprint.h"
#include "compute_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1); // = S_t = seq_chunk_tiles
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2); // = S_t = seq_chunk_tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_fp32= get_compile_time_arg_val(4);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_slots[2] = {tt::CBIndex::c_1, tt::CBIndex::c_4};
    constexpr uint32_t cb_v_slots[2] = {tt::CBIndex::c_2, tt::CBIndex::c_5};
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_lse_in = tt::CBIndex::c_18;
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_19;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;
    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_mm2_prev = tt::CBIndex::c_25;
    constexpr uint32_t cb_mm2_cur = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_prev = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_cur = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_prev = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_cur = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t kv_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    uint32_t alias_prev_sum = cb_sum_prev;
    uint32_t alias_cur_sum = cb_sum_cur;
    uint32_t alias_prev_max = cb_max_prev;
    uint32_t alias_cur_max = cb_max_cur;
    uint32_t alias_mm2_prev_out = cb_mm2_prev;
    uint32_t alias_mm2_cur_out = cb_mm2_cur;

    bool first_block = true;

    cb_wait_front(cb_q, q_chunk_tiles);


    for (uint32_t step = 0; step < ring_size; ++step) {
        uint32_t parity = step & 0x1;
        uint32_t cb_k_cur = cb_k_slots[parity];
        uint32_t cb_v_cur = cb_v_slots[parity];
		// DPRINT << "entered compute main loop step=" << step << ENDL();

		mm_init(cb_q, cb_k_cur, cb_qk_im);
		// DPRINT << "wait for [" << step << "] K block in cb"  << ENDL();
        cb_wait_front(cb_k_cur, kv_chunk_tiles);

		matmul_blocks(
			cb_q,
			cb_k_cur,  // cb_k_cur would be empty, so no need to pop again
			cb_qk_im,
			Sq_chunk_t,
			Sk_chunk_t,
			DHt,
			qk_chunk_tiles, // unused parameter
			Sq_chunk_t,     // subblock num, here we just treat one tile as one subblock
			Sk_chunk_t,     // as above
			DHt,
			1,              // num of tiles per block in height dim
			1,              // num of tiles per block in width dim
			true /*transpose*/);
		/*
                 // reduce_c can perform both reduce_max and eltwise max with previous result.
                 // if do_eltwise_max:
                 //  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                 // else:
                 //  cur_max = max(qk, dim=-1)
                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
		 */
	// template <.., uint32_t in0_cb, uint32_t scale_cb, ..>
	// void reduce_c(uint32_t out_cb, uint32_t prev_cb, bool do_eltwise_max = false) {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols produced
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced
        reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t>(
            alias_cur_max, alias_prev_max, !first_block);

		/**
		 * sub_exp fuses a few operations.
		 * In-place it performs `QK = exp((QK - cur_max) * scale)`
		 *
		 * It also partially performs reduce_sum on the output using L1 accumulation.
		 * `cur_sum = sum_tiles(exp((QK - cur_max) * scale), dim=-1)`
		 *
		 * Partial reduce_sum is used to push the final row_reduction within a tile
		 * outside of the loop over K chunks.
		 */

		// template <uint32_t in0_cb, uint32_t rows, uint32_t cols, uint32_t scale_fp32>
		// void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb) {
		// Precondition: in0_cb has rows*cols produced
		// Precondition: in1_cb has rows produced
		// Postcondition: in0_cb has rows*cols produced
		// Postcondition: in1_cb has rows produced
        sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32>(
            alias_cur_max, alias_cur_sum);

		DPRINT << "wait for [" << step << "] V block in cb"  << ENDL();
        cb_wait_front(cb_v_cur, kv_chunk_tiles);
		cb_wait_front(cb_qk_im, qk_chunk_tiles);
		// OUT_IM = QK @ V_CHUNK 
		// void matmul_blocks( const uint32_t& in0_cb, const uint32_t& in1_cb, const uint32_t& out_cb,...
		// precondition: in0_cb has M*K produced
		// preconditino: in1_cb has K*N produced
		// postcondition: in0_cb is full, in1_cb is empty
		// postcondition: out_cb has M*N produced
		matmul_blocks(
			cb_qk_im,
			cb_v_cur,  // cb_v_cur would be empty, so no need to pop again
			alias_mm2_cur_out,
			Sq_chunk_t,
			DHt,
			Sk_chunk_t,
			out_chunk_tiles,
			Sq_chunk_t,
			DHt,
			Sk_chunk_t,
			1,
			1,
			false );

		cb_pop_front(cb_qk_im, qk_chunk_tiles);
		DPRINT << "[" << step << "] K/V block is computed and consumed"  << ENDL();
        if (!first_block) {
			/* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
			// void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
			// Precondition: in0_cb and in1_cb have num_tiles produced
			// Postcondition: out_cb has num_tiles produced
			// Postcondition: in0_cb and in1_cb has num_tiles produced
            sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);

			cb_pop_front(alias_prev_max, Sq_chunk_t);
			/* cb_prev_sum *= cb_exp_max_diff */
			// void mul_tiles_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
			// Precondition: in0_cb and in1_cb have num_tiles produced
			// Postcondition: in0_cb has num_tiles produced
			// Postcondition: in1_cb has num_tiles produced
            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

			DPRINT << "compute got prev_sum and prev_max in [" << step << "]" << ENDL(); // yes we got in step=1
			/* cb_cur_sum += cb_prev_sum */
			// void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
			// Precondition: in0_cb and in1_cb have num_tiles produced
			// Postcondition: in0_cb has num_tiles produced
			// Postcondition: in1_cb has num_tiles consumed
            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

			/* cb_out_accumulate_im *= cb_exp_max_diff */
			// void mul_block_bcast_cols(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, bool pack_accumulate = false) {
			// Precondition: in0_cb has rows*cols produced
			// Precondition: in1_cb has rows produced
			// Precondition: out_cb has rows*cols produced
			// Postcondition: in0_cb empty
			// Postcondition: in1_cb empty
			// Postcondition: out_cb has rows*cols produced
			DPRINT << "compute waiting mm2_prev_out in [" << step << "]" << ENDL(); 
            mul_block_bcast_cols<Sq_chunk_t, DHt>(alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, true);
			DPRINT << "compute got mm2_prev_out in [" << step << "]" << ENDL(); 
        }

        std::swap(alias_prev_sum, alias_cur_sum);
        std::swap(alias_prev_max, alias_cur_max);
        std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
        first_block = false;

		// DPRINT << "LSE computation thing in [" << step << "]" << ENDL();
        // Finalize current iteration's contribution
		// void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb) {
		// precondition: in0_cb has M*K produced
		// preconditino: in1_cb has K*N produced
		// postcondition: in0_cb is full, in1_cb is empty
		// postcondition: out_cb has M*N produced
		// DPRINT << "compute waiting for cb_col_identity in [" << step << "]"  << ENDL();
        matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum); // cb_col_identity is empty
		DPRINT << "compute got cb_col_identity in [" << step << "]" << ENDL();
        log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);
		// DPRINT << "compute finisehd log_block in [" << step << "]" << ENDL();
		// template <uint32_t in1_scalar_cb, uint32_t num_tiles>
		// void mul_block_bcast_scalar_inplace(uint32_t in0_cb) {
		// Precondition: in0_cb has num_tiles produced
		// Precondition: in1_scalar_cb has 1 produced
		// Postcondition: in0_cb has num_tiles produced
		// Postcondition: in1_scalar_cb has 1 produced
        mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);
		// DPRINT << "compute finished mul_block_bcast_scalar_inplace in [" << step << "]" << ENDL();
		// void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
		// Precondition: in0_cb and in1_cb have num_tiles produced
		// Postcondition: in0_cb has num_tiles produced
		// Postcondition: in1_cb has num_tiles consumed
        add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);
		// DPRINT << "compute finished add_block_inplace in [" << step << "]" << ENDL();
		// void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
		// Precondition: in_cb has num_tiles produced
		// Postcondition: in_cb has num_tiles produced
        recip_block_inplace(alias_prev_sum, Sq_chunk_t);
		// DPRINT << "compute finished recip_block_inplace in [" << step << "]" << ENDL();
		// void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb) {
		// Precondition: in0_cb has rows*cols produced
		// Precondition: in1_cb has rows produced
		// Postcondition: in0_cb has rows*cols produced
		// Postcondition: in1_cb has rows produced, modified in my own version of compute_common.hpp
        mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_mm2_prev_out, alias_prev_sum);
		// DPRINT << "compute finished mul_block_bcast_cols_inplace  in [" << step << "]" << ENDL();

        if (step > 0) {

            uint32_t alias_cur_lse = alias_prev_max; // full
            uint32_t alias_sig = alias_cur_max;  // empty
            uint32_t alias_cur_out_block = alias_mm2_prev_out; //full
            uint32_t alias_sub = alias_mm2_cur_out; // empty

			// DPRINT << "compute got stuck waiting for prev_max in [" << step << "]" << ENDL();
            sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);
			// DPRINT << "compute got stuck waiting for prev_out in [" << step << "]" << ENDL();
            sub_block(cb_prev_out, alias_cur_out_block, alias_sub, out_chunk_tiles);
            mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
            sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
            cb_pop_front(cb_prev_out, out_chunk_tiles);
            // cb_pop_front(alias_cur_out_block, out_chunk_tiles); // we need mm2_prev_out later
            cb_pop_front(alias_sub, out_chunk_tiles);
            cb_pop_front(alias_sig, out_chunk_tiles);

            logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
            sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
            cb_pop_front(alias_sig, Sq_chunk_t);
            // cb_pop_front(alias_cur_lse, Sq_chunk_t); // we need prev_max later
            cb_pop_front(cb_lse_in, Sq_chunk_t);
        } else {
            pack_reconfig_data_format(cb_out);
            pack_reconfig_data_format(cb_lse_out);
			// void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
			// Precondition: in_cb has num_tiles produced
			// Precondition: out_cb has num_tiles free
			// Postcondition: in_cb has num_tiles produced // modified in my own version of compute_common.hpp
			// Postcondition: out_cb has num_tiles produced
            copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);
			// here dataflow_writer should got cb_out and write it to DRAM, then read it to cb_prev_out
			// DPRINT <<"compute stuck waiting for prev_max when step=0" << ENDL();
            copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
			// DPRINT <<"compute got prev_max when step = 0" << ENDL();  // this is printed, we do got prev_max
        }
		DPRINT << "end of step=" << step << ENDL();
    }

    cb_pop_front(cb_q, q_chunk_tiles);
	DPRINT << "end of computation" << ENDL();
}
}
