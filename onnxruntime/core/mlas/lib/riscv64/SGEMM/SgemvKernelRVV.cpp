// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SgemvKernelRVV.cpp
//
// SGEMV (M=1) micro-kernel for RISC-V Vector (RVV 1.0), specialized for
// VLEN=256 (vlenb=32). Drop-in replacement for scalar/SgemvKernelScalar.cpp
// on RISC-V builds. Used by the M=1 fast path in
// sgemm.cpp::MlasSgemmOperation, which bypasses the normal B-packing and
// passes the original (unpacked) B matrix directly with row stride ldb.
//
// Computes:
//     ZeroMode == true:    C[0..N) := A[0..K) * B[0..K, 0..N)
//     ZeroMode == false:   C[0..N) += A[0..K) * B[0..K, 0..N)
//
// Tile shape on VLEN=256:
//   Bulk path     : 4 x e32m4 accumulators per outer iter (128 N elements)
//                   - reads 4 contiguous 32-element vectors from each B row
//                     (a total of 128 floats = 512 bytes = 8 cache lines)
//                     per K-step, giving the prefetcher / cache subsystem a
//                     much wider working set per memory transaction than the
//                     previous single-m8 (64-element / 256-byte) version
//                   - holds 4 m4 accumulators across the K loop (16 phys
//                     regs), so C is loaded once at the top and stored once
//                     at the bottom of the outer N iteration
//                   - prefetches B several K-rows ahead inside the inner
//                     loop to hide DRAM latency on the strided K access
//   Tail path     : single e32m8 vector for any N < 128 remainder
//
// SGEMV is bandwidth-bound on this chip - the strided K access pattern
// (each next K-row of B is `ldb` floats away) is the main cost. The wider
// outer-N strip and the prefetch hints both target that bottleneck.

#include "mlasi.h"

#include <riscv_vector.h>

void
MLASCALL
MlasGemvFloatKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode)
{
    // Number of K-rows to look ahead with software prefetch hints. Tuned
    // by feel for the X100 - aggressive enough to hide DRAM latency but
    // not so far that prefetched lines get evicted before use.
    constexpr ptrdiff_t kPrefetchAhead = 8;

    constexpr size_t kStripWide = 128;  // bulk N-strip width
    size_t n = 0;

    // ---------- Bulk path: process 128-element N strips ----------
    while (n + kStripWide <= CountN) {
        const size_t vl = __riscv_vsetvl_e32m4(32);  // == 32 on VLEN=256

        // Four m4 accumulators cover 4 * 32 = 128 contiguous N elements.
        vfloat32m4_t acc0, acc1, acc2, acc3;
        if (ZeroMode) {
            acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            acc3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        } else {
            acc0 = __riscv_vle32_v_f32m4(C + n + 0  * 32, vl);
            acc1 = __riscv_vle32_v_f32m4(C + n + 1  * 32, vl);
            acc2 = __riscv_vle32_v_f32m4(C + n + 2  * 32, vl);
            acc3 = __riscv_vle32_v_f32m4(C + n + 3  * 32, vl);
        }

        const float* b = B + n;
        for (size_t k = 0; k < CountK; ++k) {
            // Prefetch the same N strip from a future K-row.
            __builtin_prefetch(b + kPrefetchAhead * static_cast<ptrdiff_t>(ldb));

            const float ak = A[k];
            vfloat32m4_t b0 = __riscv_vle32_v_f32m4(b + 0  * 32, vl);
            vfloat32m4_t b1 = __riscv_vle32_v_f32m4(b + 1  * 32, vl);
            vfloat32m4_t b2 = __riscv_vle32_v_f32m4(b + 2  * 32, vl);
            vfloat32m4_t b3 = __riscv_vle32_v_f32m4(b + 3  * 32, vl);
            acc0 = __riscv_vfmacc_vf_f32m4(acc0, ak, b0, vl);
            acc1 = __riscv_vfmacc_vf_f32m4(acc1, ak, b1, vl);
            acc2 = __riscv_vfmacc_vf_f32m4(acc2, ak, b2, vl);
            acc3 = __riscv_vfmacc_vf_f32m4(acc3, ak, b3, vl);
            b += ldb;
        }

        __riscv_vse32_v_f32m4(C + n + 0  * 32, acc0, vl);
        __riscv_vse32_v_f32m4(C + n + 1  * 32, acc1, vl);
        __riscv_vse32_v_f32m4(C + n + 2  * 32, acc2, vl);
        __riscv_vse32_v_f32m4(C + n + 3  * 32, acc3, vl);

        n += kStripWide;
    }

    // ---------- Tail path: any remaining N (< 128) handled with one m8 ----------
    while (n < CountN) {
        const size_t vl = __riscv_vsetvl_e32m8(CountN - n);

        vfloat32m8_t acc;
        if (ZeroMode) {
            acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        } else {
            acc = __riscv_vle32_v_f32m8(C + n, vl);
        }

        const float* b = B + n;
        for (size_t k = 0; k < CountK; ++k) {
            __builtin_prefetch(b + kPrefetchAhead * static_cast<ptrdiff_t>(ldb));
            vfloat32m8_t bk = __riscv_vle32_v_f32m8(b, vl);
            acc             = __riscv_vfmacc_vf_f32m8(acc, A[k], bk, vl);
            b += ldb;
        }

        __riscv_vse32_v_f32m8(C + n, acc, vl);
        n += vl;
    }
}
