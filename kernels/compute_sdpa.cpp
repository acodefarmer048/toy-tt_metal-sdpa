#include <algorithm>
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary_decl.h"
#include "compute_kernel_api/eltwise_binary/eltwise_binary_decl.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t ring_size = get_compile_time_arg_val(0);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_fp32_bits = get_compile_time_arg_val(4);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_identity = tt::CBIndex::c_3;
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

    cb_wait_front(cb_q, q_chunk_tiles);

    uint32_t alias_prev_sum = cb_sum_prev;
    uint32_t alias_cur_sum = cb_sum_cur;
    uint32_t alias_prev_max = cb_max_prev;
    uint32_t alias_cur_max = cb_max_cur;
    uint32_t alias_prev_out = cb_mm2_prev;
    uint32_t alias_cur_out = cb_mm2_cur;

    bool first_block = true;

    for (uint32_t step = 0; step < ring_size; ++step) {
        cb_wait_front(cb_k, kv_chunk_tiles);
        cb_wait_front(cb_v, kv_chunk_tiles);

        mm_init(cb_q, cb_k, cb_qk_im);
        matmul_tiles(cb_q, cb_k, 0, kv_chunk_tiles, 0, true);

        reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity, Sq_chunk_t, Sk_chunk_t>(
            alias_cur_max, alias_prev_max, !first_block);

        sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32_bits>(
            alias_cur_max, alias_cur_sum);

        mm_init(cb_qk_im, cb_v, alias_cur_out);
        matmul_tiles(cb_qk_im, cb_v, 0, qk_chunk_tiles, 0, false);

        if (!first_block) {
            sub_exp_block<scale_fp32_bits>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);

            mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
            add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

            mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_prev_out, cb_exp_max_diff);
            add_block_inplace(alias_cur_out, alias_prev_out, out_chunk_tiles);
        }

        std::swap(alias_prev_sum, alias_cur_sum);
        std::swap(alias_prev_max, alias_cur_max);
        std::swap(alias_prev_out, alias_cur_out);
        first_block = false;

        cb_pop_front(cb_k, kv_chunk_tiles);
        cb_pop_front(cb_v, kv_chunk_tiles);
    }

    recip_block_inplace(alias_prev_sum, Sq_chunk_t);
    mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_prev_out, alias_prev_sum);
    copy_block(alias_prev_out, cb_out, out_chunk_tiles);

    cb_pop_front(cb_q, q_chunk_tiles);
}
}
