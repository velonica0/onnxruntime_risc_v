// SQNBitGemm M1 kernel benchmark: scalar vs RVV, scaling 1->2->4->8 threads.
//
// Threading model: partition the output column (N) dimension across threads.
// Each thread calls the M1 kernel with its column slice — independent, no
// synchronization required.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>

#include "mlas.h"
#include "mlas_qnbit.h"
#include "sqnbitgemm.h"

extern const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchRiscv;

// Stub for the onnxruntime threadpool symbol referenced by threading.cpp.o
// inside libonnxruntime_mlas.a. We always pass nullptr for the threadpool,
// so this body is dead code at runtime — it just satisfies the linker.
#include <functional>
namespace onnxruntime { namespace concurrency {
class ThreadPool {
public:
    void SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn);
    static int DegreeOfParallelism(const ThreadPool* tp);
};
void ThreadPool::SimpleParallelFor(std::ptrdiff_t total,
                                   const std::function<void(std::ptrdiff_t)>& fn)
{
    for (std::ptrdiff_t i = 0; i < total; ++i) fn(i);
}
int ThreadPool::DegreeOfParallelism(const ThreadPool*) { return 1; }
} }  // namespace onnxruntime::concurrency

namespace bench
{

// Same algorithm as the RVV kernel, written in plain C++ with no SIMD
// intrinsics. The optimize attribute prevents GCC's tree-vectorizer from
// re-vectorizing this back into RVV — that's the whole point of the
// "scalar baseline".
__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize")))
void
ScalarM1Kernel(
    size_t            BlkLen,
    const float*      A,
    const std::byte*  QuantBData,
    const float*      QuantBScale,
    const std::byte*  QuantBZeroPoint,
    float*            C,
    size_t            CountN,
    size_t            CountK,
    size_t            BlockCountK,
    const float*      Bias)
{
    constexpr size_t BlkBitWidth = 4;
    constexpr size_t SubBlkLen   = 16;
    constexpr size_t SubBlkBytes = SubBlkLen / 2;

    const size_t StrideQuantBData  = BlockCountK * (BlkLen / 2);
    const size_t StrideQuantBScale = BlockCountK;

    const std::byte* QBDColPtr = QuantBData;
    const float*     QBSColPtr = QuantBScale;

    for (size_t n = 0; n < CountN; ++n) {
        float acc = 0.0f;
        const uint8_t* qbU8    = reinterpret_cast<const uint8_t*>(QBDColPtr);
        const float*   qbScale = QBSColPtr;

        for (size_t k = 0; k < CountK; k += BlkLen) {
            const float scale = *qbScale++;

            float offset_v;
            if (QuantBZeroPoint != nullptr) {
                // (path not exercised by the symmetric benchmark below)
                offset_v = 8.0f;
            } else {
                offset_v = 8.0f;
            }

            for (size_t kk = 0; kk < BlkLen; kk += SubBlkLen) {
                // After packing, byte[bp] of a SubBlk holds:
                //   low nibble  -> v[bp]      (K position bp     in [0..7])
                //   high nibble -> v[bp + 8]  (K position bp + 8 in [8..15])
                for (size_t bp = 0; bp < SubBlkBytes; ++bp) {
                    const uint8_t b = qbU8[bp];
                    const float   vlo = static_cast<float>(b & 0x0F) - offset_v;
                    const float   vhi = static_cast<float>((b >> 4) & 0x0F) - offset_v;
                    acc += A[k + kk + bp] * vlo * scale;
                    acc += A[k + kk + bp + 8] * vhi * scale;
                }
                qbU8 += SubBlkBytes;
            }
        }

        if (Bias != nullptr) {
            acc += Bias[n];
        }
        C[n] = acc;

        QBDColPtr += StrideQuantBData;
        QBSColPtr += StrideQuantBScale;
    }
}

template <typename Fn>
double
TimeMs(Fn&& fn, int iters)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

}  // namespace bench

struct Shape
{
    size_t      N;
    size_t      K;
    size_t      BlkLen;
    const char* name;
};

// Run kernel `slice_fn(n_start, n_count)` partitioned across `num_threads`
// over the N dimension. Slices are roughly equal-size; OpenMP manages a
// persistent thread pool so successive calls are cheap.
template <typename Fn>
static inline void
ParallelCallByN(size_t CountN, int num_threads, Fn&& slice_fn)
{
    if (num_threads <= 1) {
        slice_fn(size_t{0}, CountN);
        return;
    }
    #pragma omp parallel num_threads(num_threads)
    {
        const int t  = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const size_t cols_per = (CountN + nt - 1) / nt;
        const size_t n_start  = static_cast<size_t>(t) * cols_per;
        if (n_start < CountN) {
            const size_t n_count = std::min(cols_per, CountN - n_start);
            slice_fn(n_start, n_count);
        }
    }
}

int
main()
{
    const Shape shapes[] = {
        {4096,  4096,  32, "Llama proj 4096x4096 BlkLen=32"},
        {11008, 4096,  32, "Llama FFN-up 11008x4096 BlkLen=32"},
        {4096,  11008, 32, "Llama FFN-down 4096x11008 BlkLen=32"},
        {32000, 4096,  32, "Llama lm_head 32000x4096 BlkLen=32"},
        {2048, 2048, 128, "Mid 2048x2048 BlkLen=128"},
    };

    const int thread_counts[] = {1, 2, 4, 8};

    std::printf("SQNBit M1Kernel CompFp32 — scalar (no autovec) vs RVV, multi-thread scaling\n\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dfloat(-1.0f, 1.0f);
    std::uniform_int_distribution<int>    dbyte(0, 255);

    for (const auto& sh : shapes) {
        const size_t N = sh.N, K = sh.K, BlkLen = sh.BlkLen;
        const size_t BlockCountK = (K + BlkLen - 1) / BlkLen;
        const size_t QBDataBytes = N * BlockCountK * (BlkLen / 2);
        const size_t StrideQBData  = BlockCountK * (BlkLen / 2);
        const size_t StrideQBScale = BlockCountK;

        std::vector<float>    A(K);
        std::vector<std::byte> QBRaw(QBDataBytes);
        std::vector<std::byte> QBPacked(QBDataBytes);
        std::vector<float>    Scale(N * BlockCountK);
        std::vector<float>    CScalar(N), CRvv(N);

        for (auto& a : A) a = dfloat(rng);
        for (auto& s : Scale) s = dfloat(rng) * 0.1f + 0.01f;
        for (auto& b : QBRaw) b = static_cast<std::byte>(dbyte(rng));

        MlasSQNBitGemmDispatchRiscv.SQ4BitGemmPackQuantBData(
            N, K, BlkLen, CompFp32,
            QBRaw.data(), QBPacked.data(), nullptr);

        // Slice callables
        auto scalar_slice = [&](size_t n_start, size_t n_count) {
            bench::ScalarM1Kernel(
                BlkLen, A.data(),
                QBPacked.data() + n_start * StrideQBData,
                Scale.data()    + n_start * StrideQBScale,
                nullptr,
                CScalar.data() + n_start,
                n_count, K, BlockCountK, nullptr);
        };
        auto rvv_slice = [&](size_t n_start, size_t n_count) {
            MlasSQNBitGemmDispatchRiscv.SQ4BitGemmM1Kernel_CompFp32(
                BlkLen, A.data(),
                QBPacked.data() + n_start * StrideQBData,
                Scale.data()    + n_start * StrideQBScale,
                nullptr,
                CRvv.data() + n_start,
                n_count, K, BlockCountK, nullptr);
        };

        const double approx_flops = 2.0 * static_cast<double>(N) * static_cast<double>(K);

        std::printf("Shape: %s   (FLOPs/call = %.0f M)\n", sh.name, approx_flops / 1e6);
        std::printf("%6s %16s %16s %12s\n",
                    "thr", "scalar GFLOPS", "RVV GFLOPS", "RVV/scalar");

        // Run each thread count
        for (int nt : thread_counts) {
            // Warmup
            ParallelCallByN(N, nt, scalar_slice);
            ParallelCallByN(N, nt, rvv_slice);

            // Iter counts based on serial estimate
            const int iters_scalar = std::max(2, static_cast<int>(1.5e9 / approx_flops * nt));
            const int iters_rvv    = std::max(20, iters_scalar * 6);

            const double scalar_ms = bench::TimeMs(
                [&] { ParallelCallByN(N, nt, scalar_slice); }, iters_scalar);
            const double rvv_ms = bench::TimeMs(
                [&] { ParallelCallByN(N, nt, rvv_slice); }, iters_rvv);

            const double scalar_gflops = approx_flops / scalar_ms / 1e6;
            const double rvv_gflops    = approx_flops / rvv_ms    / 1e6;

            std::printf("%6d %16.3f %16.3f %11.2fx\n",
                        nt, scalar_gflops, rvv_gflops,
                        rvv_gflops / scalar_gflops);
        }

        // Sanity diff (single-thread results)
        ParallelCallByN(N, 1, scalar_slice);
        ParallelCallByN(N, 1, rvv_slice);
        float max_diff = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            const float d = std::fabs(CScalar[i] - CRvv[i]);
            if (d > max_diff) max_diff = d;
        }
        std::printf("       max_diff(scalar,rvv) = %.2e\n\n", max_diff);
    }

    return 0;
}
