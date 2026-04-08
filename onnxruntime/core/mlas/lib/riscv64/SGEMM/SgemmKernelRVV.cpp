// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// SgemmKernelRVV.cpp
//
// Inner micro-kernel for SGEMM on RISC-V Vector (RVV 1.0), specialized for
// VLEN=256 (vlenb=32). Drop-in replacement for scalar/SgemmKernelScalar.cpp
// on RISC-V builds. Consumes B in the 16-wide N-panel layout produced by
// MlasSgemmCopyPackB (sgemm.cpp around line 213).
//
// Tile shape on VLEN=256:
//   LMUL    = e32m2 (VLMAX = 2*256/32 = 16 elements - exactly the panel
//                    K-stride; every vfmacc.vf fully utilizes all 16 lanes)
//   N-tile  = 16    (one packed subpanel per kernel iteration)
//   M-tile  = 1 / 2 / 4 / 8 / 12 rows of A, picked at the dispatch site
//             based on CountM. M=12 uses 12 m2 accumulators = 24 phys regs,
//             leaving 6 phys regs for B and scratch under a 32-reg budget.
//   K-loop  = 2x manually unrolled in the M=8 and M=12 hot paths so the
//             second B load can issue while the first batch of fmaccs is
//             still draining its latency. The compiler does not unroll
//             this loop on its own at -O2 with these intrinsics.
//
// The kernel returns the number of M rows it actually handled. The driver
// in sgemm.cpp::MlasSgemmKernelLoop calls it repeatedly until CountM is
// exhausted, picking the largest tile each time.

#include "mlasi.h"

#include <riscv_vector.h>

namespace {

// =============================================================================
// Generic templated path for M_TILE in {1, 2, 4} - simple, single B load
// per K-step, no manual unroll. Used by the smaller tail tiles.
// =============================================================================

template <bool ZeroMode, int M_TILE>
static MLAS_FORCEINLINE size_t
sgemm_inner_small(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha)
{
    static_assert(M_TILE >= 1 && M_TILE <= 4, "small inner: M_TILE must be in [1,4]");

    do {
        const size_t step = CountN < 16 ? CountN : 16;
        const size_t vl   = __riscv_vsetvl_e32m2(step);

        vfloat32m2_t acc0, acc1, acc2, acc3;
        if constexpr (M_TILE >= 1) acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        if constexpr (M_TILE >= 2) acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        if constexpr (M_TILE >= 3) acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        if constexpr (M_TILE >= 4) acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

        const float* a = A;
        for (size_t k = 0; k < CountK; ++k) {
            vfloat32m2_t b = __riscv_vle32_v_f32m2(B, vl);
            if constexpr (M_TILE >= 1) acc0 = __riscv_vfmacc_vf_f32m2(acc0, a[0 * lda], b, vl);
            if constexpr (M_TILE >= 2) acc1 = __riscv_vfmacc_vf_f32m2(acc1, a[1 * lda], b, vl);
            if constexpr (M_TILE >= 3) acc2 = __riscv_vfmacc_vf_f32m2(acc2, a[2 * lda], b, vl);
            if constexpr (M_TILE >= 4) acc3 = __riscv_vfmacc_vf_f32m2(acc3, a[3 * lda], b, vl);
            B += 16;
            a += 1;
        }

        auto finish = [&](int m, vfloat32m2_t accv) {
            accv = __riscv_vfmul_vf_f32m2(accv, alpha, vl);
            if constexpr (!ZeroMode) {
                vfloat32m2_t c = __riscv_vle32_v_f32m2(C + m * ldc, vl);
                accv           = __riscv_vfadd_vv_f32m2(accv, c, vl);
            }
            __riscv_vse32_v_f32m2(C + m * ldc, accv, vl);
        };
        if constexpr (M_TILE >= 1) finish(0, acc0);
        if constexpr (M_TILE >= 2) finish(1, acc1);
        if constexpr (M_TILE >= 3) finish(2, acc2);
        if constexpr (M_TILE >= 4) finish(3, acc3);

        if (CountN <= 16) break;
        C      += 16;
        CountN -= 16;
    } while (true);

    return M_TILE;
}

// =============================================================================
// Hot path for M_TILE in {8, 12} - K loop is manually unrolled by 2 and the
// two B loads are scheduled before any fmacc, so the loads can be in-flight
// while the first batch of fmaccs drains. M=8 uses 8 m2 accs (16 phys regs);
// M=12 uses 12 m2 accs (24 phys regs).
// =============================================================================

template <bool ZeroMode, int M_TILE>
static MLAS_FORCEINLINE size_t
sgemm_inner_big(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha)
{
    static_assert(M_TILE == 8 || M_TILE == 12, "big inner: M_TILE must be 8 or 12");

    do {
        const size_t step = CountN < 16 ? CountN : 16;
        const size_t vl   = __riscv_vsetvl_e32m2(step);

        vfloat32m2_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
        vfloat32m2_t acc8, acc9, acc10, acc11;
        acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc4 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc5 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc6 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        acc7 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        if constexpr (M_TILE == 12) {
            acc8  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            acc9  = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            acc10 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            acc11 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        }

        const float* a = A;
        size_t k = 0;

        // 2x unrolled K loop
        for (; k + 1 < CountK; k += 2) {
            // Issue both B loads before any fmacc so the second load can
            // overlap with the first round of fmaccs.
            vfloat32m2_t b0 = __riscv_vle32_v_f32m2(B,      vl);
            vfloat32m2_t b1 = __riscv_vle32_v_f32m2(B + 16, vl);

            const float a00 = a[0 * lda]; const float a10 = a[1 * lda];
            const float a20 = a[2 * lda]; const float a30 = a[3 * lda];
            const float a40 = a[4 * lda]; const float a50 = a[5 * lda];
            const float a60 = a[6 * lda]; const float a70 = a[7 * lda];

            acc0 = __riscv_vfmacc_vf_f32m2(acc0, a00, b0, vl);
            acc1 = __riscv_vfmacc_vf_f32m2(acc1, a10, b0, vl);
            acc2 = __riscv_vfmacc_vf_f32m2(acc2, a20, b0, vl);
            acc3 = __riscv_vfmacc_vf_f32m2(acc3, a30, b0, vl);
            acc4 = __riscv_vfmacc_vf_f32m2(acc4, a40, b0, vl);
            acc5 = __riscv_vfmacc_vf_f32m2(acc5, a50, b0, vl);
            acc6 = __riscv_vfmacc_vf_f32m2(acc6, a60, b0, vl);
            acc7 = __riscv_vfmacc_vf_f32m2(acc7, a70, b0, vl);
            if constexpr (M_TILE == 12) {
                const float a80  = a[ 8 * lda];
                const float a90  = a[ 9 * lda];
                const float a100 = a[10 * lda];
                const float a110 = a[11 * lda];
                acc8  = __riscv_vfmacc_vf_f32m2(acc8,  a80,  b0, vl);
                acc9  = __riscv_vfmacc_vf_f32m2(acc9,  a90,  b0, vl);
                acc10 = __riscv_vfmacc_vf_f32m2(acc10, a100, b0, vl);
                acc11 = __riscv_vfmacc_vf_f32m2(acc11, a110, b0, vl);
            }

            const float a01 = a[0 * lda + 1]; const float a11 = a[1 * lda + 1];
            const float a21 = a[2 * lda + 1]; const float a31 = a[3 * lda + 1];
            const float a41 = a[4 * lda + 1]; const float a51 = a[5 * lda + 1];
            const float a61 = a[6 * lda + 1]; const float a71 = a[7 * lda + 1];

            acc0 = __riscv_vfmacc_vf_f32m2(acc0, a01, b1, vl);
            acc1 = __riscv_vfmacc_vf_f32m2(acc1, a11, b1, vl);
            acc2 = __riscv_vfmacc_vf_f32m2(acc2, a21, b1, vl);
            acc3 = __riscv_vfmacc_vf_f32m2(acc3, a31, b1, vl);
            acc4 = __riscv_vfmacc_vf_f32m2(acc4, a41, b1, vl);
            acc5 = __riscv_vfmacc_vf_f32m2(acc5, a51, b1, vl);
            acc6 = __riscv_vfmacc_vf_f32m2(acc6, a61, b1, vl);
            acc7 = __riscv_vfmacc_vf_f32m2(acc7, a71, b1, vl);
            if constexpr (M_TILE == 12) {
                const float a81  = a[ 8 * lda + 1];
                const float a91  = a[ 9 * lda + 1];
                const float a101 = a[10 * lda + 1];
                const float a111 = a[11 * lda + 1];
                acc8  = __riscv_vfmacc_vf_f32m2(acc8,  a81,  b1, vl);
                acc9  = __riscv_vfmacc_vf_f32m2(acc9,  a91,  b1, vl);
                acc10 = __riscv_vfmacc_vf_f32m2(acc10, a101, b1, vl);
                acc11 = __riscv_vfmacc_vf_f32m2(acc11, a111, b1, vl);
            }

            B += 32;
            a += 2;
        }

        // K-loop tail (if CountK is odd)
        if (k < CountK) {
            vfloat32m2_t b = __riscv_vle32_v_f32m2(B, vl);
            acc0 = __riscv_vfmacc_vf_f32m2(acc0, a[0 * lda], b, vl);
            acc1 = __riscv_vfmacc_vf_f32m2(acc1, a[1 * lda], b, vl);
            acc2 = __riscv_vfmacc_vf_f32m2(acc2, a[2 * lda], b, vl);
            acc3 = __riscv_vfmacc_vf_f32m2(acc3, a[3 * lda], b, vl);
            acc4 = __riscv_vfmacc_vf_f32m2(acc4, a[4 * lda], b, vl);
            acc5 = __riscv_vfmacc_vf_f32m2(acc5, a[5 * lda], b, vl);
            acc6 = __riscv_vfmacc_vf_f32m2(acc6, a[6 * lda], b, vl);
            acc7 = __riscv_vfmacc_vf_f32m2(acc7, a[7 * lda], b, vl);
            if constexpr (M_TILE == 12) {
                acc8  = __riscv_vfmacc_vf_f32m2(acc8,  a[ 8 * lda], b, vl);
                acc9  = __riscv_vfmacc_vf_f32m2(acc9,  a[ 9 * lda], b, vl);
                acc10 = __riscv_vfmacc_vf_f32m2(acc10, a[10 * lda], b, vl);
                acc11 = __riscv_vfmacc_vf_f32m2(acc11, a[11 * lda], b, vl);
            }
            B += 16;
            a += 1;
        }

        auto finish = [&](int m, vfloat32m2_t accv) {
            accv = __riscv_vfmul_vf_f32m2(accv, alpha, vl);
            if constexpr (!ZeroMode) {
                vfloat32m2_t c = __riscv_vle32_v_f32m2(C + m * ldc, vl);
                accv           = __riscv_vfadd_vv_f32m2(accv, c, vl);
            }
            __riscv_vse32_v_f32m2(C + m * ldc, accv, vl);
        };
        finish(0, acc0); finish(1, acc1); finish(2, acc2); finish(3, acc3);
        finish(4, acc4); finish(5, acc5); finish(6, acc6); finish(7, acc7);
        if constexpr (M_TILE == 12) {
            finish( 8, acc8 ); finish( 9, acc9 );
            finish(10, acc10); finish(11, acc11);
        }

        if (CountN <= 16) break;
        C      += 16;
        CountN -= 16;
    } while (true);

    return M_TILE;
}

template <bool ZeroMode>
static size_t
dispatch(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha)
{
    if (CountM >= 12) return sgemm_inner_big<ZeroMode, 12>(A, B, C, CountK, CountN, lda, ldc, alpha);
    if (CountM >=  8) return sgemm_inner_big<ZeroMode,  8>(A, B, C, CountK, CountN, lda, ldc, alpha);
    if (CountM >=  4) return sgemm_inner_small<ZeroMode, 4>(A, B, C, CountK, CountN, lda, ldc, alpha);
    if (CountM >=  2) return sgemm_inner_small<ZeroMode, 2>(A, B, C, CountK, CountN, lda, ldc, alpha);
    return                   sgemm_inner_small<ZeroMode, 1>(A, B, C, CountK, CountN, lda, ldc, alpha);
}

}  // namespace

size_t MLASCALL
MlasSgemmKernelZero(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha)
{
    return dispatch<true>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}

size_t MLASCALL
MlasSgemmKernelAdd(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha)
{
    return dispatch<false>(A, B, C, CountK, CountM, CountN, lda, ldc, alpha);
}
