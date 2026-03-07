#include <algorithm>
#include <cstdint>
#include "compute_kernel_api.h"

#include "compute_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1); // = S_t = seq_chunk_tiles
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2); // = S_t = seq_chunk_tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_fp32_bits = get_compile_time_arg_val(4);

	constexpr uint32_t cb_q = tt::CBIndex::c_0;
	constexpr uint32_t cb_k_slots[2] = {tt::CBIndex::c_1, tt::CBIndex::c_4};
	constexpr uint32_t cb_v_slots[2] = {tt::CBIndex::c_2, tt::CBIndex::c_5};
	constexpr uint32_t cb_reduce_scale_in = tt::CBIndex::c_6;
	constexpr uint32_t cb_scale_in = tt::CBIndex::c_7;
	constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
	constexpr uint32_t cb_out = tt::CBIndex::c_16;
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
    uint32_t alias_prev_out = cb_mm2_prev;
    uint32_t alias_cur_out = cb_mm2_cur;

    bool first_block = true;

    cb_wait_front(cb_q, q_chunk_tiles);


    for (uint32_t step = 0; step < ring_size; ++step) {
        uint32_t parity = step & 0x1;
        uint32_t cb_k_cur = cb_k_slots[parity];
        uint32_t cb_v_cur = cb_v_slots[parity];

        cb_wait_front(cb_k_cur, kv_chunk_tiles);

		mm_init(cb_q, cb_k_cur, cb_qk_im);

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
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t>(
                    alias_cur_max, alias_prev_max, k_chunk > iter_k_chunk_start);
		 */
		reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_reduce_scale_in, Sq_chunk_t, Sk_chunk_t>(
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
		/*
			template <uint32_t in0_cb, uint32_t rows, uint32_t cols, uint32_t scale_fp32>
			void sub_exp_block_bcast_cols_inplace(uint32_t in1_cb, uint32_t reduce_cb) {
		 */
        sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32_bits>(
            alias_cur_max, alias_cur_sum);

        cb_wait_front(cb_v_cur, kv_chunk_tiles);
		cb_wait_front(cb_qk_im, qk_chunk_tiles);
		// OUT_IM = QK @ V_CHUNK 
		matmul_blocks(
			cb_qk_im,
			cb_v_cur,  // cb_v_cur would be empty, so no need to pop again
			alias_cur_out,
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
        if (!first_block) {
			/* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
            sub_exp_block<scale_fp32_bits>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
			cb_pop_front(alias_prev_max, Sq_chunk_t);
			/* cb_prev_sum *= cb_exp_max_diff */
            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
			/* cb_cur_sum += cb_prev_sum */
            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

			/* cb_out_accumulate_im *= cb_exp_max_diff */
            mul_block_bcast_cols<Sq_chunk_t, DHt>(alias_prev_out, cb_exp_max_diff, alias_cur_out, true);
        }

        std::swap(alias_prev_sum, alias_cur_sum);
        std::swap(alias_prev_max, alias_cur_max);
        std::swap(alias_prev_out, alias_cur_out);
        first_block = false;

    }
	// TODO
	// Calculate current LSE
	// Use alias_cur_max as intermediate buffer.
	// template <uint32_t M>
	// void matmul_reduce(uint32_t in1_cb, const uint32_t& out_cb), out_cb is used as in0_cb
	// N is set to 1
    // precondition: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: in0_cb is full, in1_cb is empty
    // postcondition: out_cb has M*N produced
	matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);

	// void log_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
	log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);

	// Scale prev_max by scale_fp32
	mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);
	add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);

	/* cb_cur_sum = 1.0 / cb_cur_sum */
	recip_block_inplace(alias_prev_sum, Sq_chunk_t);
	/* cb_out_accumulate_im *= cb_cur_sum */
	mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_prev_out, alias_prev_sum);
	if (step > 0) {
		// Update output according to previous and current LSE
		/**
		 * sig = torch.sigmoid(cur_lse - prev_lse)
		 * out = prev_out - sig * (prev_out - cur_out)
		 * lse = prev_lse - torch.logsigmoid(prev_lse - cur_lse)
		 */
		cb_wait_front(cb_lse_in, Sq_chunk_t);
		cb_wait_front(cb_prev_out, out_chunk_tiles);

		uint32_t alias_cur_lse = alias_prev_max;      // full
		uint32_t alias_sig = alias_cur_max;           // empty
		uint32_t alias_cur_out = alias_prev_out;  // full
		uint32_t alias_sub = alias_cur_out;       // empty

		// alias_sig = sigmoid(alias_cur_lse - cb_lse_in)
		sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

		// alias_sub = cb_prev_out - alias_cur_out
		sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
		// alias_sub *= alias_sig
		mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
		// cb_out = cb_prev_out - alias_sub
		sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
		cb_pop_front(cb_prev_out, out_chunk_tiles);
		cb_pop_front(alias_cur_out, out_chunk_tiles);
		cb_pop_front(alias_sub, out_chunk_tiles);

		// alias_sig = sigmoid(cb_lse_in - alias_cur_lse)
		// alias_cur_lse = log(alias_sig)
		// cb_lse_out = cb_lse_in - alias_cur_lse
		logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
		sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
		cb_pop_front(alias_sig, Sq_chunk_t);
		cb_pop_front(alias_cur_lse, Sq_chunk_t);
		cb_pop_front(cb_lse_in, Sq_chunk_t);

	} else {
		pack_reconfig_data_format(cb_out);
		copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);

		copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
	}

	cb_pop_front(cb_q_in, q_chunk_tiles);
}
}
