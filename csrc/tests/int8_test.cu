// Standalone raw INT8 MMA smoke test (no PyTorch dependency).
// Build: nvcc -I csrc -arch=sm_86 -std=c++17 -O3 csrc/tests/int8_test.cu -o /tmp/int8_test && /tmp/int8_test

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../kernels/common/mma.cuh"

namespace {
constexpr int M = 16, N = 8, K = 32;

__device__ __forceinline__ unsigned pack4(const int8_t* x, int offset) {
    unsigned out = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        out |= static_cast<unsigned>(static_cast<unsigned char>(x[offset + i])) << (8 * i);
    return out;
}

__global__ void mma_s8_kernel(const int8_t* a, const int8_t* b, int* out) {
    const int lane = threadIdx.x;
    const int group = lane >> 2, t4 = lane & 3;
    const int k0 = t4 * 4;
    unsigned af[4] = {
        pack4(a, group * K + k0), pack4(a, (group + 8) * K + k0),
        pack4(a, group * K + k0 + 16), pack4(a, (group + 8) * K + k0 + 16),
    };
    unsigned bf[2] = {pack4(b, group * K + k0), pack4(b, group * K + k0 + 16)};
    int acc[4] = {0, 0, 0, 0};
    astrai::mma_sync_s8(acc, af, bf, acc);
    const int col = t4 * 2;
    out[group * N + col] = acc[0];
    out[group * N + col + 1] = acc[1];
    out[(group + 8) * N + col] = acc[2];
    out[(group + 8) * N + col + 1] = acc[3];
}
}  // namespace

int main() {
    std::vector<int8_t> a(M * K), b(N * K);
    std::vector<int> ref(M * N, 0), out(M * N);
    for (auto& x : a) x = static_cast<int8_t>(std::rand() % 15 - 7);
    for (auto& x : b) x = static_cast<int8_t>(std::rand() % 15 - 7);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                ref[m * N + n] += static_cast<int>(a[m * K + k]) * b[n * K + k];

    int8_t *da, *db;
    int* dout;
    cudaMalloc(&da, a.size()); cudaMalloc(&db, b.size()); cudaMalloc(&dout, out.size() * sizeof(int));
    cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
    mma_s8_kernel<<<1, 32>>>(da, db, dout);
    cudaMemcpy(out.data(), dout, out.size() * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dout);
    for (int i = 0; i < M * N; ++i) {
        if (out[i] != ref[i]) {
            std::fprintf(stderr, "mismatch at %d: got %d, expected %d\n", i, out[i], ref[i]);
            return 1;
        }
    }
    std::puts("INT8 MMA m16n8k32: PASS");
    return 0;
}
