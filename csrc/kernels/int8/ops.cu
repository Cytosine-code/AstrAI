// Stateless W8A8 inference primitives. INT8 weights use [N, K] layout and
// one scale per output channel; activations use one dynamic scale per call.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "../common/device.cuh"
#include "../common/cp_async.cuh"
#include "../common/mma.cuh"
#include "../common/reduce.cuh"

namespace {
constexpr int kThreads = 256;
constexpr float kQMax = 127.0f;

__device__ __forceinline__ int8_t quantize_s8(float x, float scale) {
    int q = __float2int_rn(x / scale);
    q = max(-127, min(127, q));
    return static_cast<int8_t>(q);
}

__global__ void global_amax_kernel(const __nv_bfloat16* x, int64_t n, float* amax) {
    float local = 0.0f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)blockDim.x * gridDim.x)
        local = fmaxf(local, fabsf(__bfloat162float(x[i])));
    local = astrai::warp_reduce_max(local);
    __shared__ float warps[32];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float value = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; ++i) value = fmaxf(value, warps[i]);
        astrai::atomic_max_float(amax, value);
    }
}

__global__ void scale_from_amax_kernel(const float* amax, float* scale) {
    *scale = fmaxf(*amax / kQMax, 1e-12f);
}

__global__ void quantize_tensor_kernel(const __nv_bfloat16* x, int8_t* out,
                                        int64_t n, const float* amax, float* scale) {
    const float s = fmaxf(*amax / kQMax, 1e-12f);
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale = s;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)blockDim.x * gridDim.x)
        out[i] = quantize_s8(__bfloat162float(x[i]), s);
}

__global__ void weight_scales_kernel(const __nv_bfloat16* w, float* scales,
                                     int n, int k) {
    const int row = blockIdx.x;
    if (row >= n) return;
    float local = 0.0f;
    for (int col = threadIdx.x; col < k; col += blockDim.x)
        local = fmaxf(local, fabsf(__bfloat162float(w[row * k + col])));
    local = astrai::warp_reduce_max(local);
    __shared__ float warps[32];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float value = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; ++i) value = fmaxf(value, warps[i]);
        scales[row] = fmaxf(value / kQMax, 1e-12f);
    }
}

__global__ void quantize_weight_kernel(const __nv_bfloat16* w, int8_t* out,
                                       const float* scales, int64_t total, int k) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)blockDim.x * gridDim.x)
        out[i] = quantize_s8(__bfloat162float(w[i]), scales[i / k]);
}

__device__ __forceinline__ unsigned pack_s8x4(const int8_t* x, int row, int col,
                                               int rows, int k) {
    unsigned packed = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned byte = row < rows && col + i < k
            ? static_cast<unsigned>(static_cast<unsigned char>(x[(int64_t)row * k + col + i])) : 0u;
        packed |= byte << (8 * i);
    }
    return packed;
}

// Correctness fallback: one warp owns one m16n8 tile. Keeping this separate
// makes the fragment contract easy to diagnose if the CTA-tiled path regresses.
__global__ void int8_gemm_raw_kernel(const int8_t* a, const int8_t* b,
                                     const float* scale_a, const float* scale_b,
                                     const __nv_bfloat16* bias, __nv_bfloat16* out,
                                     int m, int n, int k) {
    const int lane = threadIdx.x;
    const int group = lane >> 2;
    const int t4 = lane & 3;
    const int row0 = blockIdx.y * 16;
    const int col0 = blockIdx.x * 8;
    int acc[4] = {0, 0, 0, 0};
    for (int kk = 0; kk < k; kk += 32) {
        const int k0 = kk + t4 * 4;
        unsigned af[4];
        af[0] = pack_s8x4(a, row0 + group, k0, m, k);
        af[1] = pack_s8x4(a, row0 + group + 8, k0, m, k);
        af[2] = pack_s8x4(a, row0 + group, k0 + 16, m, k);
        af[3] = pack_s8x4(a, row0 + group + 8, k0 + 16, m, k);
        unsigned bf[2];
        bf[0] = pack_s8x4(b, col0 + group, k0, n, k);
        bf[1] = pack_s8x4(b, col0 + group, k0 + 16, n, k);
        astrai::mma_sync_s8(acc, af, bf, acc);
    }
    const int col = col0 + t4 * 2;
    const float sx = *scale_a;
    auto store = [&](int row, int column, int value) {
        if (row < m && column < n) {
            float y = static_cast<float>(value) * sx * scale_b[column];
            if (bias) y += __bfloat162float(bias[column]);
            out[(int64_t)row * n + column] = __float2bfloat16(y);
        }
    };
    store(row0 + group, col, acc[0]);
    store(row0 + group, col + 1, acc[1]);
    store(row0 + group + 8, col, acc[2]);
    store(row0 + group + 8, col + 1, acc[3]);
}

// CTA tiled W8A8 GEMM. A 64x64 activation tile is reused by four 64x32 warp
// tiles across N; each warp's 64x32 weight tile is reused across 64 M rows.
// The fragment mapping is identical to the raw kernel, but ldmatrix loads it
// from two staged 64-wide K slices instead of repeatedly reading global memory.
constexpr int kGemmBlockM = 64;
constexpr int kGemmBlockN = 128;
constexpr int kGemmBlockK = 64;
constexpr int kGemmStages = 2;
constexpr int kGemmThreads = 128;

constexpr int kRegTileM = 64;
constexpr int kRegTileN = 64;
constexpr int kRegTileK = 128;

constexpr int kWideTileM = 64;
constexpr int kWideTileN = 128;
constexpr int kWideThreads = 256;

// Wide-N register tile. Eight warps cover a 64x128 CTA as 2 M-warps x 4
// N-warps; each warp remains a 32x32 register tile, preserving the 96-register
// accumulator footprint while reducing A tile reloads per output element.
__global__ __launch_bounds__(kWideThreads, 2) void int8_gemm_wide_regtile_kernel(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b,
    const float* __restrict__ scale_a, const float* __restrict__ scale_b,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    __shared__ __align__(16) int8_t a_smem[kGemmStages][kWideTileM * kGemmBlockK];
    __shared__ __align__(16) int8_t b_smem[kGemmStages][kWideTileN * kGemmBlockK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int group = lane >> 2, t4 = lane & 3;
    const int row_block = blockIdx.y * kWideTileM;
    const int col_block = blockIdx.x * kWideTileN;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const int warp_row = warp_m * 32, warp_col = warp_n * 32;
    const int r7 = lane & 7, rh8 = (lane >> 3) & 1, rh16 = lane >> 4;
    int acc[4][2][4] = {};
    const int tiles = (k + kGemmBlockK - 1) / kGemmBlockK;

    auto stage_tile = [&](int stage, int k_base) {
        const int a_chunk = tid;
        const int a_row = a_chunk / 4, a_col = (a_chunk & 3) * 16;
        int8_t* a_dst = a_smem[stage] + a_row * kGemmBlockK + a_col;
        const bool a_full = row_block + a_row < m && k_base + a_col + 15 < k &&
                            (k & 15) == 0;
        if (a_full) {
            astrai::cp_async_16(a_dst,
                a + (int64_t)(row_block + a_row) * k + k_base + a_col, true);
        } else {
#pragma unroll
            for (int j = 0; j < 16; ++j)
                a_dst[j] = row_block + a_row < m && k_base + a_col + j < k
                    ? a[(int64_t)(row_block + a_row) * k + k_base + a_col + j] : 0;
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid * 2 + i;
            const int row = chunk / 4, col = (chunk & 3) * 16;
            int8_t* dst = b_smem[stage] + row * kGemmBlockK + col;
            const bool full = col_block + row < n && k_base + col + 15 < k &&
                              (k & 15) == 0;
            if (full) {
                astrai::cp_async_16(dst,
                    b + (int64_t)(col_block + row) * k + k_base + col, true);
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    dst[j] = col_block + row < n && k_base + col + j < k
                        ? b[(int64_t)(col_block + row) * k + k_base + col + j] : 0;
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < kGemmStages; ++stage) {
        if (stage < tiles) {
            stage_tile(stage, stage * kGemmBlockK);
            astrai::cp_async_commit_group();
        }
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) astrai::cp_async_wait_group<1>();
        else astrai::cp_async_wait_group<0>();
        __syncthreads();
        unsigned b_frag[2][4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int b_row = warp_col + nt * 8 + r7;
            astrai::ldmatrix_x2_lane(b_frag[0][nt],
                __cvta_generic_to_shared(b_smem[stage] + b_row * kGemmBlockK + rh8 * 16));
        }
#pragma unroll
        for (int k_seg = 0; k_seg < 2; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg == 0) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int b_row = warp_col + nt * 8 + r7;
                    astrai::ldmatrix_x2_lane(b_frag[bnext][nt],
                        __cvta_generic_to_shared(b_smem[stage] + b_row * kGemmBlockK +
                                                 (2 + rh8) * 16));
                }
            }
            unsigned a_frag[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int a_row = warp_row + mt * 16 + rh8 * 8 + r7;
                astrai::ldmatrix_x4_lane(a_frag[mt],
                    __cvta_generic_to_shared(a_smem[stage] + a_row * kGemmBlockK +
                                             (k_seg * 2 + rh16) * 16));
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    astrai::mma_sync_s8(acc[nt][mt], a_frag[mt], b_frag[bcur][nt],
                                        acc[nt][mt]);
            }
        }
        __syncthreads();
        if (tile + kGemmStages < tiles) {
            stage_tile(stage, (tile + kGemmStages) * kGemmBlockK);
            astrai::cp_async_commit_group();
        }
    }

    const float sx = *scale_a;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int col = col_block + warp_col + t4 * 2 + nt * 8;
        if (col >= n) continue;
        const float sw0 = scale_b[col];
        const float sw1 = col + 1 < n ? scale_b[col + 1] : 0.0f;
        const float bias0 = bias ? __bfloat162float(bias[col]) : 0.0f;
        const float bias1 = bias && col + 1 < n ? __bfloat162float(bias[col + 1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int row = row_block + warp_row + mt * 16 + group;
            if (row < m) {
                out[(int64_t)row * n + col] = __float2bfloat16(acc[nt][mt][0] * sx * sw0 + bias0);
                if (col + 1 < n)
                    out[(int64_t)row * n + col + 1] =
                        __float2bfloat16(acc[nt][mt][1] * sx * sw1 + bias1);
            }
            if (row + 8 < m) {
                out[(int64_t)(row + 8) * n + col] =
                    __float2bfloat16(acc[nt][mt][2] * sx * sw0 + bias0);
                if (col + 1 < n)
                    out[(int64_t)(row + 8) * n + col + 1] =
                        __float2bfloat16(acc[nt][mt][3] * sx * sw1 + bias1);
            }
        }
    }
}

// Register-tiled candidate for medium/large M. Each warp computes a 32x32
// output tile (2 M fragments x 4 N fragments), so each thread keeps 32 INT32
// accumulators instead of the 64 in the 64x128 kernel. This lowers register
// pressure and permits substantially higher resident-warp count on SM86.
__global__ __launch_bounds__(kGemmThreads, 2) void int8_gemm_regtile_kernel(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b,
    const float* __restrict__ scale_a, const float* __restrict__ scale_b,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    __shared__ __align__(16) int8_t a_smem[kGemmStages][kRegTileM * kRegTileK];
    __shared__ __align__(16) int8_t b_smem[kGemmStages][kRegTileN * kRegTileK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int group = lane >> 2, t4 = lane & 3;
    const int row_block = blockIdx.y * kRegTileM;
    const int col_block = blockIdx.x * kRegTileN;
    const int warp_m = warp >> 1, warp_n = warp & 1;
    const int warp_row = warp_m * 32;
    const int warp_col = warp_n * 32;
    const int r7 = lane & 7, rh8 = (lane >> 3) & 1, rh16 = lane >> 4;
    int acc[4][2][4] = {};  // [N fragment][M fragment][INT32 result]
    const int tiles = (k + kRegTileK - 1) / kRegTileK;

    auto stage_tile = [&](int stage, int k_base) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int chunk = tid * 4 + i;
            const int row = chunk / 8, col = (chunk & 7) * 16;
            int8_t* dst = a_smem[stage] + row * kRegTileK + col;
            const bool full = row_block + row < m && k_base + col + 15 < k &&
                              (k & 15) == 0;
            if (full) {
                astrai::cp_async_16(dst,
                    a + (int64_t)(row_block + row) * k + k_base + col, true);
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    dst[j] = row_block + row < m && k_base + col + j < k
                        ? a[(int64_t)(row_block + row) * k + k_base + col + j] : 0;
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int chunk = tid * 4 + i;
            const int row = chunk / 8, col = (chunk & 7) * 16;
            int8_t* dst = b_smem[stage] + row * kRegTileK + col;
            const bool full = col_block + row < n && k_base + col + 15 < k &&
                              (k & 15) == 0;
            if (full) {
                astrai::cp_async_16(dst,
                    b + (int64_t)(col_block + row) * k + k_base + col, true);
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    dst[j] = col_block + row < n && k_base + col + j < k
                        ? b[(int64_t)(col_block + row) * k + k_base + col + j] : 0;
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < kGemmStages; ++stage) {
        if (stage < tiles) {
            stage_tile(stage, stage * kRegTileK);
            astrai::cp_async_commit_group();
        }
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) astrai::cp_async_wait_group<1>();
        else astrai::cp_async_wait_group<0>();
        __syncthreads();
        unsigned b_frag[2][4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int b_row = warp_col + nt * 8 + r7;
            astrai::ldmatrix_x2_lane(b_frag[0][nt],
                __cvta_generic_to_shared(b_smem[stage] + b_row * kRegTileK + rh8 * 16));
        }
#pragma unroll
        for (int k_seg = 0; k_seg < 4; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg + 1 < 4) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int b_row = warp_col + nt * 8 + r7;
                    astrai::ldmatrix_x2_lane(b_frag[bnext][nt],
                        __cvta_generic_to_shared(b_smem[stage] + b_row * kRegTileK +
                                                 ((k_seg + 1) * 2 + rh8) * 16));
                }
            }
            unsigned a_frag[2][4];
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int a_row = warp_row + mt * 16 + rh8 * 8 + r7;
                astrai::ldmatrix_x4_lane(a_frag[mt],
                    __cvta_generic_to_shared(a_smem[stage] + a_row * kRegTileK +
                                             (k_seg * 2 + rh16) * 16));
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    astrai::mma_sync_s8(acc[nt][mt], a_frag[mt], b_frag[bcur][nt],
                                        acc[nt][mt]);
            }
        }
        __syncthreads();
        if (tile + kGemmStages < tiles) {
            stage_tile(stage, (tile + kGemmStages) * kRegTileK);
            astrai::cp_async_commit_group();
        }
    }

    const float sx = *scale_a;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int col = col_block + warp_col + t4 * 2 + nt * 8;
        if (col >= n) continue;
        const float sw0 = scale_b[col];
        const float sw1 = col + 1 < n ? scale_b[col + 1] : 0.0f;
        const float bias0 = bias ? __bfloat162float(bias[col]) : 0.0f;
        const float bias1 = bias && col + 1 < n ? __bfloat162float(bias[col + 1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int row = row_block + warp_row + mt * 16 + group;
            if (row < m) {
                out[(int64_t)row * n + col] = __float2bfloat16(acc[nt][mt][0] * sx * sw0 + bias0);
                if (col + 1 < n)
                    out[(int64_t)row * n + col + 1] =
                        __float2bfloat16(acc[nt][mt][1] * sx * sw1 + bias1);
            }
            if (row + 8 < m) {
                out[(int64_t)(row + 8) * n + col] =
                    __float2bfloat16(acc[nt][mt][2] * sx * sw0 + bias0);
                if (col + 1 < n)
                    out[(int64_t)(row + 8) * n + col + 1] =
                        __float2bfloat16(acc[nt][mt][3] * sx * sw1 + bias1);
            }
        }
    }
}

__global__ __launch_bounds__(kGemmThreads, 2) void int8_gemm_tiled_kernel(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b,
    const float* __restrict__ scale_a, const float* __restrict__ scale_b,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    __shared__ __align__(16) int8_t a_smem[kGemmStages][kGemmBlockM * kGemmBlockK];
    __shared__ __align__(16) int8_t b_smem[kGemmStages][kGemmBlockN * kGemmBlockK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 2;
    const int t4 = lane & 3;
    const int row_block = blockIdx.y * kGemmBlockM;
    const int col_block = blockIdx.x * kGemmBlockN;
    const int warp_col = warp * 32;
    const int r7 = lane & 7;
    const int rh8 = (lane >> 3) & 1;
    const int rh16 = lane >> 4;

    int acc[4][4][4] = {};  // [N warp MMA][M MMA][fragment result]
    const int tiles = (k + kGemmBlockK - 1) / kGemmBlockK;

    auto stage_tile = [&](int stage, int k_base) {
        // Every thread moves two 16-byte A chunks and four B chunks. K is
        // contiguous for both source operands, so cp.async is directly usable.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int chunk = tid * 2 + i;
            const int row = chunk / 4;
            const int col = (chunk % 4) * 16;
            int8_t* dst = a_smem[stage] + row * kGemmBlockK + col;
            const bool full = row_block + row < m && k_base + col + 15 < k &&
                              (k & 15) == 0;
            if (full) {
                astrai::cp_async_16(dst,
                    a + (int64_t)(row_block + row) * k + k_base + col, true);
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    dst[j] = row_block + row < m && k_base + col + j < k
                        ? a[(int64_t)(row_block + row) * k + k_base + col + j] : 0;
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int chunk = tid * 4 + i;
            const int row = chunk / 4;
            const int col = (chunk % 4) * 16;
            int8_t* dst = b_smem[stage] + row * kGemmBlockK + col;
            const bool full = col_block + row < n && k_base + col + 15 < k &&
                              (k & 15) == 0;
            if (full) {
                astrai::cp_async_16(dst,
                    b + (int64_t)(col_block + row) * k + k_base + col, true);
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    dst[j] = col_block + row < n && k_base + col + j < k
                        ? b[(int64_t)(col_block + row) * k + k_base + col + j] : 0;
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < kGemmStages; ++stage) {
        if (stage < tiles) {
            stage_tile(stage, stage * kGemmBlockK);
            astrai::cp_async_commit_group();
        }
    }

    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile % kGemmStages;
        if (tile + 1 < tiles)
            astrai::cp_async_wait_group<1>();
        else
            astrai::cp_async_wait_group<0>();
        __syncthreads();

        unsigned b_frag[2][4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int b_row = warp_col + nt * 8 + r7;
            astrai::ldmatrix_x2_lane(b_frag[0][nt],
                __cvta_generic_to_shared(b_smem[stage] + b_row * kGemmBlockK + rh8 * 16));
        }
#pragma unroll
        for (int k_seg = 0; k_seg < 2; ++k_seg) {
            const int bcur = k_seg & 1;
            const int bnext = bcur ^ 1;
            if (k_seg + 1 < 2) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int b_row = warp_col + nt * 8 + r7;
                    astrai::ldmatrix_x2_lane(b_frag[bnext][nt],
                        __cvta_generic_to_shared(b_smem[stage] + b_row * kGemmBlockK +
                                                 (2 + rh8) * 16));
                }
            }
            unsigned a_frag[4][4];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const int a_row = mt * 16 + rh8 * 8 + r7;
                astrai::ldmatrix_x4_lane(a_frag[mt],
                    __cvta_generic_to_shared(a_smem[stage] + a_row * kGemmBlockK +
                                             (k_seg * 2 + rh16) * 16));
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    astrai::mma_sync_s8(acc[nt][mt], a_frag[mt], b_frag[bcur][nt],
                                        acc[nt][mt]);
            }
        }
        __syncthreads();
        if (tile + kGemmStages < tiles) {
            stage_tile(stage, (tile + kGemmStages) * kGemmBlockK);
            astrai::cp_async_commit_group();
        }
    }

    const float sx = *scale_a;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int col = col_block + warp_col + t4 * 2 + nt * 8;
        if (col >= n) continue;
        const float sw0 = scale_b[col];
        const float sw1 = col + 1 < n ? scale_b[col + 1] : 0.0f;
        const float bias0 = bias ? __bfloat162float(bias[col]) : 0.0f;
        const float bias1 = bias && col + 1 < n ? __bfloat162float(bias[col + 1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const int row = row_block + mt * 16 + group;
            if (row < m) {
                out[(int64_t)row * n + col] = __float2bfloat16(acc[nt][mt][0] * sx * sw0 + bias0);
                if (col + 1 < n)
                    out[(int64_t)row * n + col + 1] =
                        __float2bfloat16(acc[nt][mt][1] * sx * sw1 + bias1);
            }
            if (row + 8 < m) {
                out[(int64_t)(row + 8) * n + col] =
                    __float2bfloat16(acc[nt][mt][2] * sx * sw0 + bias0);
                if (col + 1 < n)
                    out[(int64_t)(row + 8) * n + col + 1] =
                        __float2bfloat16(acc[nt][mt][3] * sx * sw1 + bias1);
            }
        }
    }
}

// Small-M specialization. Decode and short prefill do not have enough rows
// to justify a 64-row CTA: this variant keeps the same four N-warps but lets
// each warp own one m16n32 tile, avoiding 3/4 of the MMA work at M <= 16.
__global__ __launch_bounds__(kGemmThreads, 2) void int8_gemm_small_m_kernel(
    const int8_t* __restrict__ a, const int8_t* __restrict__ b,
    const float* __restrict__ scale_a, const float* __restrict__ scale_b,
    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    constexpr int kSmallTileK = 128;
    __shared__ __align__(16) int8_t a_smem[kGemmStages][16 * kSmallTileK];
    __shared__ __align__(16) int8_t b_smem[kGemmStages][kGemmBlockN * kSmallTileK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int group = lane >> 2, t4 = lane & 3;
    const int row_block = blockIdx.y * 16;
    const int col_block = blockIdx.x * kGemmBlockN;
    const int r7 = lane & 7, rh8 = (lane >> 3) & 1, rh16 = lane >> 4;
    int acc[4][4] = {};  // [N MMA][fragment result]
    const int tiles = (k + kSmallTileK - 1) / kSmallTileK;

    auto stage_tile = [&](int stage, int k_base) {
        // 64 A chunks and 512 B chunks. The B loop remains four chunks per
        // thread; only the first 64 threads participate in the compact A tile.
        for (int chunk = tid; chunk < 128; chunk += kGemmThreads) {
            const int row = chunk / 8, col = (chunk % 8) * 16;
            int8_t* dst = a_smem[stage] + row * kSmallTileK + col;
            const bool full = row_block + row < m && k_base + col + 15 < k &&
                              (k & 15) == 0;
            if (full) {
                astrai::cp_async_16(dst,
                    a + (int64_t)(row_block + row) * k + k_base + col, true);
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    dst[j] = row_block + row < m && k_base + col + j < k
                        ? a[(int64_t)(row_block + row) * k + k_base + col + j] : 0;
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int chunk = tid * 8 + i;
            const int row = chunk / 8, col = (chunk % 8) * 16;
            int8_t* dst = b_smem[stage] + row * kSmallTileK + col;
            const bool full = col_block + row < n && k_base + col + 15 < k &&
                              (k & 15) == 0;
            if (full) {
                astrai::cp_async_16(dst,
                    b + (int64_t)(col_block + row) * k + k_base + col, true);
            } else {
#pragma unroll
                for (int j = 0; j < 16; ++j)
                    dst[j] = col_block + row < n && k_base + col + j < k
                        ? b[(int64_t)(col_block + row) * k + k_base + col + j] : 0;
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < kGemmStages; ++stage) {
        if (stage < tiles) {
            stage_tile(stage, stage * kSmallTileK);
            astrai::cp_async_commit_group();
        }
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile % kGemmStages;
        if (tile + 1 < tiles) astrai::cp_async_wait_group<1>();
        else astrai::cp_async_wait_group<0>();
        __syncthreads();
        unsigned b_frag[2][4][2];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int b_row = warp * 32 + nt * 8 + r7;
            astrai::ldmatrix_x2_lane(b_frag[0][nt],
                __cvta_generic_to_shared(b_smem[stage] + b_row * kSmallTileK + rh8 * 16));
        }
#pragma unroll
        for (int k_seg = 0; k_seg < 4; ++k_seg) {
            const int bcur = k_seg & 1, bnext = bcur ^ 1;
            if (k_seg + 1 < 4) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int b_row = warp * 32 + nt * 8 + r7;
                    astrai::ldmatrix_x2_lane(b_frag[bnext][nt],
                        __cvta_generic_to_shared(b_smem[stage] + b_row * kSmallTileK +
                                                 ((k_seg + 1) * 2 + rh8) * 16));
                }
            }
            unsigned a_frag[4];
            astrai::ldmatrix_x4_lane(a_frag,
                __cvta_generic_to_shared(a_smem[stage] + (rh8 * 8 + r7) * kSmallTileK +
                                         (k_seg * 2 + rh16) * 16));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                astrai::mma_sync_s8(acc[nt], a_frag, b_frag[bcur][nt], acc[nt]);
        }
        __syncthreads();
        if (tile + kGemmStages < tiles) {
            stage_tile(stage, (tile + kGemmStages) * kSmallTileK);
            astrai::cp_async_commit_group();
        }
    }
    const float sx = *scale_a;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int col = col_block + warp * 32 + t4 * 2 + nt * 8;
        const int row = row_block + group;
        if (row >= m || col >= n) continue;
        const float sw0 = scale_b[col];
        const float b0 = bias ? __bfloat162float(bias[col]) : 0.0f;
        out[(int64_t)row * n + col] = __float2bfloat16(acc[nt][0] * sx * sw0 + b0);
        if (col + 1 < n) {
            const float sw1 = scale_b[col + 1];
            const float b1 = bias ? __bfloat162float(bias[col + 1]) : 0.0f;
            out[(int64_t)row * n + col + 1] =
                __float2bfloat16(acc[nt][1] * sx * sw1 + b1);
        }
        if (row + 8 < m) {
            out[(int64_t)(row + 8) * n + col] =
                __float2bfloat16(acc[nt][2] * sx * sw0 + b0);
            if (col + 1 < n) {
                const float sw1 = scale_b[col + 1];
                const float b1 = bias ? __bfloat162float(bias[col + 1]) : 0.0f;
                out[(int64_t)(row + 8) * n + col + 1] =
                    __float2bfloat16(acc[nt][3] * sx * sw1 + b1);
            }
        }
    }
}

// Decode has M=1, where m16 tensor-core tiles execute fifteen padded rows.
// This GEMV path trades that waste for signed dp4a: one warp owns one output
// channel and reduces its partial K dot product. A is staged once per 4-warp
// CTA, so its vector is not fetched independently by every output warp.
__global__ void int8_gemv_m1_kernel(const int8_t* __restrict__ a,
                                    const int8_t* __restrict__ b,
                                    const float* __restrict__ scale_a,
                                    const float* __restrict__ scale_b,
                                    const __nv_bfloat16* __restrict__ bias,
                                    __nv_bfloat16* __restrict__ out, int n, int k) {
    extern __shared__ int8_t a_smem[];
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += blockDim.x) a_smem[i] = a[i];
    __syncthreads();

    const int col = blockIdx.x * (blockDim.x / 32) + (tid >> 5);
    if (col >= n) return;
    const int lane = tid & 31;
    int sum = 0;
    int kk = lane * 4;
    // Model K dimensions are multiples of four. The scalar tail preserves the
    // public API's arbitrary-K correctness contract.
    if ((k & 3) == 0) {
        for (; kk + 3 < k; kk += 32 * 4) {
            const int av = *reinterpret_cast<const int*>(a_smem + kk);
            const int bv = *reinterpret_cast<const int*>(b + (int64_t)col * k + kk);
            sum = __dp4a(av, bv, sum);
        }
    } else {
        kk = lane;
    }
    for (; kk < k; kk += 32) sum += static_cast<int>(a_smem[kk]) *
                                  static_cast<int>(b[(int64_t)col * k + kk]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) {
        float value = static_cast<float>(sum) * *scale_a * scale_b[col];
        if (bias) value += __bfloat162float(bias[col]);
        out[col] = __float2bfloat16(value);
    }
}

void check_int8_device(const torch::Tensor& tensor) {
    const auto* p = at::cuda::getDeviceProperties(tensor.device().index());
    TORCH_CHECK(astrai::sm_at_least(p->major, p->minor,
                                    astrai::kMinSmForInt8Major,
                                    astrai::kMinSmForInt8Minor),
                "INT8 MMA requires compute capability 7.5 or newer");
}

void check_bf16_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kBFloat16,
                name, " must be a CUDA bfloat16 tensor");
}

std::tuple<torch::Tensor, torch::Tensor> quantize_dynamic_bf16(torch::Tensor x) {
    check_bf16_cuda(x, "x");
    check_int8_device(x);
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto xc = x.contiguous();
    auto q = torch::empty_like(xc, xc.options().dtype(torch::kChar));
    // Reuse the scale allocation as the amax scratch. The first kernel writes
    // the reduction result; the quantization kernel converts it in-place to
    // the public scale value, eliminating one temporary tensor allocation.
    auto scale = torch::zeros({1}, xc.options().dtype(torch::kFloat));
    const int blocks = std::max(1, static_cast<int>(std::min<int64_t>(
        (static_cast<int64_t>(xc.numel()) + kThreads - 1) / kThreads, 4096)));
    global_amax_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()), xc.numel(), scale.data_ptr<float>());
    quantize_tensor_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()), q.data_ptr<int8_t>(),
        xc.numel(), scale.data_ptr<float>(), scale.data_ptr<float>());
    C10_CUDA_CHECK(cudaGetLastError());
    return {q, scale};
}

std::tuple<torch::Tensor, torch::Tensor> quantize_weight_bf16(torch::Tensor w) {
    check_bf16_cuda(w, "w");
    TORCH_CHECK(w.dim() == 2, "w must have shape [N, K]");
    check_int8_device(w);
    const at::cuda::OptionalCUDAGuard guard(w.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto wc = w.contiguous();
    const int n = wc.size(0), k = wc.size(1);
    auto q = torch::empty_like(wc, wc.options().dtype(torch::kChar));
    auto scales = torch::empty({n}, wc.options().dtype(torch::kFloat));
    weight_scales_kernel<<<n, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(wc.data_ptr()), scales.data_ptr<float>(), n, k);
    const int blocks = std::max(1, static_cast<int>(std::min<int64_t>(
        (static_cast<int64_t>(wc.numel()) + kThreads - 1) / kThreads, 4096)));
    quantize_weight_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(wc.data_ptr()), q.data_ptr<int8_t>(),
        scales.data_ptr<float>(), wc.numel(), k);
    C10_CUDA_CHECK(cudaGetLastError());
    return {q, scales};
}

torch::Tensor mm_int8(torch::Tensor a, torch::Tensor b, torch::Tensor scale_a,
                      torch::Tensor scale_b, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.scalar_type() == torch::kChar &&
                    b.scalar_type() == torch::kChar, "a and b must be CUDA int8 tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1),
                "a must be [M,K] and b must be [N,K]");
    TORCH_CHECK(a.device() == b.device() && scale_a.device() == a.device() &&
                    scale_b.device() == a.device(), "all tensors must share a device");
    TORCH_CHECK(scale_a.scalar_type() == torch::kFloat && scale_a.numel() == 1 &&
                    scale_b.scalar_type() == torch::kFloat && scale_b.numel() == b.size(0),
                "scale_a must be float32 [1] and scale_b float32 [N]");
    check_int8_device(a);
    const at::cuda::OptionalCUDAGuard guard(a.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ac = a.contiguous(), bc = b.contiguous(), sb = scale_b.contiguous();
    torch::Tensor bias_c;
    const __nv_bfloat16* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined() && bias->numel()) {
        TORCH_CHECK(bias->is_cuda() && bias->scalar_type() == torch::kBFloat16 &&
                        bias->device() == a.device() && bias->numel() == b.size(0),
                    "bias must be CUDA bfloat16 [N]");
        bias_c = bias->contiguous();
        bias_ptr = reinterpret_cast<const __nv_bfloat16*>(bias_c.data_ptr());
    }
    auto out = torch::empty({a.size(0), b.size(0)}, a.options().dtype(torch::kBFloat16));
    if (a.size(0) == 1) {
        constexpr int kGemvWarps = 4;
        const int threads = kGemvWarps * 32;
        const int blocks = (b.size(0) + kGemvWarps - 1) / kGemvWarps;
        int8_gemv_m1_kernel<<<blocks, threads, a.size(1) * sizeof(int8_t), stream>>>(
            ac.data_ptr<int8_t>(), bc.data_ptr<int8_t>(), scale_a.data_ptr<float>(),
            sb.data_ptr<float>(), bias_ptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            b.size(0), a.size(1));
    } else if (a.size(0) <= 16) {
        dim3 grid((b.size(0) + kGemmBlockN - 1) / kGemmBlockN,
                  (a.size(0) + 15) / 16);
        int8_gemm_small_m_kernel<<<grid, kGemmThreads, 0, stream>>>(
            ac.data_ptr<int8_t>(), bc.data_ptr<int8_t>(), scale_a.data_ptr<float>(),
            sb.data_ptr<float>(), bias_ptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            a.size(0), b.size(0), a.size(1));
    } else {
        dim3 grid((b.size(0) + kRegTileN - 1) / kRegTileN,
                  (a.size(0) + kRegTileM - 1) / kRegTileM);
        int8_gemm_regtile_kernel<<<grid, kGemmThreads, 0, stream>>>(
            ac.data_ptr<int8_t>(), bc.data_ptr<int8_t>(), scale_a.data_ptr<float>(),
            sb.data_ptr<float>(), bias_ptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            a.size(0), b.size(0), a.size(1));
    }
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor linear_forward_int8(torch::Tensor x, torch::Tensor w_q,
                                  torch::Tensor scale_w,
                                  c10::optional<torch::Tensor> bias) {
    check_bf16_cuda(x, "x");
    TORCH_CHECK(w_q.dim() == 2 && w_q.scalar_type() == torch::kChar,
                "w_q must be CUDA int8 [N,K]");
    TORCH_CHECK(x.size(-1) == w_q.size(1), "inner dimension mismatch");
    auto xc = x.reshape({-1, w_q.size(1)}).contiguous();
    auto [xq, sx] = quantize_dynamic_bf16(xc);
    auto out = mm_int8(xq, w_q, sx, scale_w, bias);
    std::vector<int64_t> shape(x.sizes().begin(), x.sizes().end() - 1);
    shape.push_back(w_q.size(0));
    return out.reshape(shape);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_dynamic_bf16", &quantize_dynamic_bf16, py::arg("x"));
    m.def("quantize_weight_bf16", &quantize_weight_bf16, py::arg("w"));
    m.def("mm_int8", &mm_int8, py::arg("a"), py::arg("b"), py::arg("scale_a"),
          py::arg("scale_b"), py::arg("bias") = py::none());
    m.def("linear_forward_int8", &linear_forward_int8, py::arg("x"), py::arg("w_q"),
          py::arg("scale_w"), py::arg("bias") = py::none());
}
