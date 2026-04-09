/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SQNBitGemmKernelRVV_fp32.cpp

Abstract:

    RVV (RISC-V Vector 1.0) implementation of the SQ4BitGemm CompFp32 path:
    multiply a float matrix A with a 4-bit-per-weight quantized matrix B
    (block quantized, column major). Specialized for VLEN=256 (e32m2 -> 16
    lanes), with vsetvl-based tail handling so the code remains correct on
    other VLENs.

    Provides:
        SQ4BitGemmPackQuantBDataSize
        SQ4BitGemmPackQuantBData          (matches sqnbitgemm_kernel_neon
                                           layout exactly so the M1 / dequant
                                           kernels can mirror the Neon math)
        SQ4BitGemmPerGemmWorkspaceSize
        SQ4BitGemmPerGemmWorkspaceAlignment
        SQ4BitGemmM1Kernel_CompFp32       (M=1: A is a 1xK row, B is 4-bit;
                                           dot product per output column)
        Q4BitBlkDequantBForSgemm_CompFp32 (M>1: dequantize B into the 16-wide
                                           K-major layout consumed by the
                                           RVV SGEMM kernel)

    CompInt8 paths are intentionally left null. Only CompFp32 is supported.

    Version 1: simple, correct, NCols=1 inner loop. Future work: NCols=4
    unrolling and optional cross-block prefetching.

--*/

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include <riscv_vector.h>

#include "sqnbitgemm.h"

namespace sqnbitgemm_riscv
{

namespace
{

constexpr size_t kBlkBitWidth = 4;
constexpr size_t kSubBlkLen   = 16;          // 16 4-bit values per SubBlk -> 8 bytes packed
constexpr size_t kSubBlkBytes = kSubBlkLen / 2;

//
// Quantized B data packing.
//
// For SubBlkLen == 16, pack 16 4-bit values (8 bytes) at a time like this:
//
// src: | v0 v1 | v2 v3 | v4 v5 | v6 v7 | v8 v9 | vA vB | vC vD | vE vF |
//   =>
// dst: | v0 v8 | v1 v9 | v2 vA | v3 vB | v4 vC | v5 vD | v6 vE | v7 vF |
//
// After this layout, an 8-byte vector load + low/high nibble extraction
// produces two 8-element vectors of consecutive K positions:
//   low  -> v0 v1 v2 v3 v4 v5 v6 v7
//   high -> v8 v9 vA vB vC vD vE vF
// which a vslideup combines into one 16-lane f32 vector in K order.
//

size_t
SQ4BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    return N * BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
}

void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    assert(BlkLen >= kSubBlkLen && BlkLen % kSubBlkLen == 0);

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    const size_t Iterations  = N * BlockCountK;

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            const size_t n     = tid / BlockCountK;
            const size_t k_blk = tid % BlockCountK;
            const size_t off   = n * BlockCountK * BlkDataSize + k_blk * BlkDataSize;

            const std::byte* src = QuantBDataBegin + off;
            std::byte*       dst = PackedQuantBDataBegin + off;

            for (size_t kk = 0; kk < BlkLen; kk += kSubBlkLen) {
                // 8 byte pairs per SubBlk; each iteration writes 2 destination bytes.
                for (size_t bp = 0; bp < kSubBlkBytes / 2; ++bp) {
                    const std::byte src0 = src[bp];
                    const std::byte src1 = src[bp + kSubBlkBytes / 2];

                    dst[2 * bp + 0] = (src0 & std::byte{0x0F}) | ((src1 & std::byte{0x0F}) << 4);
                    dst[2 * bp + 1] = (src0 >> 4) | ((src1 >> 4) << 4);
                }
                src += kSubBlkBytes;
                dst += kSubBlkBytes;
            }
        });
}

size_t
SQ4BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType)
{
    MLAS_UNREFERENCED_PARAMETER(M);
    MLAS_UNREFERENCED_PARAMETER(N);
    MLAS_UNREFERENCED_PARAMETER(K);
    MLAS_UNREFERENCED_PARAMETER(BlkLen);
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    // CompFp32 needs no per-GEMM workspace; CompInt8 unsupported on this dispatch.
    return 0;
}

size_t
SQ4BitGemmPerGemmWorkspaceAlignment(
    size_t BlkLen,
    MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType)
{
    MLAS_UNREFERENCED_PARAMETER(BlkLen);
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    return 1;
}

//
// Load 8 packed bytes (16 nibbles) and produce two 8-lane f32m1 vectors
// holding the low and high nibbles, in K order:
//   loF[i] = float(byte[i] & 0x0F)         for i in 0..7  (K positions 0..7)
//   hiF[i] = float((byte[i] >> 4) & 0x0F)  for i in 0..7  (K positions 8..15)
// Caller subtracts the offset (= zp or 8) and multiplies by scale.
//
MLAS_FORCEINLINE void
LoadDequant16Nibbles_LoHi_m1(
    const uint8_t* src,
    vfloat32m1_t&  loF,
    vfloat32m1_t&  hiF)
{
    const size_t vl8 = __riscv_vsetvl_e8mf4(8);
    vuint8mf4_t  pack = __riscv_vle8_v_u8mf4(src, vl8);
    vuint8mf4_t  lo8  = __riscv_vand_vx_u8mf4(pack, 0x0F, vl8);
    vuint8mf4_t  hi8  = __riscv_vsrl_vx_u8mf4(pack, 4,    vl8);

    vuint16mf2_t lo16 = __riscv_vzext_vf2_u16mf2(lo8, vl8);
    vuint16mf2_t hi16 = __riscv_vzext_vf2_u16mf2(hi8, vl8);

    vuint32m1_t lo32 = __riscv_vzext_vf2_u32m1(lo16, vl8);
    vuint32m1_t hi32 = __riscv_vzext_vf2_u32m1(hi16, vl8);

    loF = __riscv_vfcvt_f_xu_v_f32m1(lo32, vl8);
    hiF = __riscv_vfcvt_f_xu_v_f32m1(hi32, vl8);
}

//
// CompFp32 M=1 kernel inner loop, NCols (1 or 4) output columns at a time.
// Two parallel m1 accumulators per column (lo + hi nibbles) avoid the
// slideup of the v1 m2 path. The single A load per SubBlk is shared across
// all NCols columns, which is the main amortization win.
//
// Fast-path helper for FULL SubBlks (k_sb_len == 16). v3 ops chain
// (vfsub + vfmul + vfmacc) — sumA correction trick was tried in v4 and
// turned out to regress on most shapes; reverted here.
[[gnu::always_inline]] inline void
DoOneColSubBlkFull(
    const uint8_t* qbSrc,
    float          scale,
    float          offset_v,
    vfloat32m1_t   aLo,
    vfloat32m1_t   aHi,
    size_t         vl8,
    vfloat32m1_t&  accLo,
    vfloat32m1_t&  accHi)
{
    vfloat32m1_t bLoF, bHiF;
    LoadDequant16Nibbles_LoHi_m1(qbSrc, bLoF, bHiF);

    bLoF = __riscv_vfsub_vf_f32m1(bLoF, offset_v, vl8);
    bLoF = __riscv_vfmul_vf_f32m1(bLoF, scale,    vl8);
    bHiF = __riscv_vfsub_vf_f32m1(bHiF, offset_v, vl8);
    bHiF = __riscv_vfmul_vf_f32m1(bHiF, scale,    vl8);

    accLo = __riscv_vfmacc_vv_f32m1(accLo, aLo, bLoF, vl8);
    accHi = __riscv_vfmacc_vv_f32m1(accHi, aHi, bHiF, vl8);
}

// Tail helper for partial last SubBlk (k_sb_len in [1..15]).
[[gnu::always_inline]] inline void
DoOneColSubBlkTail(
    const uint8_t* qbSrc,
    float          scale,
    float          offset_v,
    vfloat32m1_t   aLo,
    vfloat32m1_t   aHi,
    size_t         vl_lo,
    size_t         vl_hi,
    vfloat32m1_t&  accLo,
    vfloat32m1_t&  accHi)
{
    vfloat32m1_t bLoF, bHiF;
    LoadDequant16Nibbles_LoHi_m1(qbSrc, bLoF, bHiF);

    const size_t vl_full = __riscv_vsetvl_e32m1(8);
    bLoF = __riscv_vfsub_vf_f32m1(bLoF, offset_v, vl_full);
    bLoF = __riscv_vfmul_vf_f32m1(bLoF, scale,    vl_full);
    bHiF = __riscv_vfsub_vf_f32m1(bHiF, offset_v, vl_full);
    bHiF = __riscv_vfmul_vf_f32m1(bHiF, scale,    vl_full);

    accLo = __riscv_vfmacc_vv_f32m1(accLo, aLo, bLoF, vl_lo);
    if (vl_hi > 0) {
        accHi = __riscv_vfmacc_vv_f32m1(accHi, aHi, bHiF, vl_hi);
    }
}

MLAS_FORCEINLINE float
ReduceLoHiToScalar(vfloat32m1_t accLo, vfloat32m1_t accHi)
{
    const size_t vl8r = __riscv_vsetvl_e32m1(8);
    vfloat32m1_t sum = __riscv_vfadd_vv_f32m1(accLo, accHi, vl8r);
    vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t red  = __riscv_vfredusum_vs_f32m1_f32m1(sum, zero, vl8r);
    return __riscv_vfmv_f_s_f32m1_f32(red);
}

// Computes a partial dot product for NCols output columns over a K range.
// Pointers (QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr) and
// ARowPtr must already point at the K-tile's starting block. CountK is the
// number of K elements *in this tile*. The first block in the K-tile is
// required to be at an even absolute block index so the zp nibble alignment
// reads the LOW nibble of byte 0 first (caller enforces this).
//
// Result is *added* to SumPtr[0..NCols-1] (caller zero-initializes C once).
// Bias is applied by the outer driver after all K tiles, not here.
template <size_t NCols, bool HasZeroPoint>
MLAS_FORCEINLINE void
ComputeDotProducts_NCols(
    size_t        BlkLen,
    const float*  ARowPtr,
    const std::byte* QuantBDataColPtr,
    const float*  QuantBScaleColPtr,
    const std::byte* QuantBZeroPointColPtr,
    size_t        CountK,
    size_t        StrideQuantBData,
    size_t        StrideQuantBScale,
    size_t        StrideQuantBZeroPoint,
    float*        SumPtr)
{
    static_assert(NCols == 1 || NCols == 4, "NCols must be 1 or 4");

    const size_t vl8 = __riscv_vsetvl_e32m1(8);

    // Sizeless RVV vars cannot be array elements; use named locals + constexpr-if.
    vfloat32m1_t accLo0 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
    vfloat32m1_t accHi0 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
    [[maybe_unused]] vfloat32m1_t accLo1, accHi1, accLo2, accHi2, accLo3, accHi3;
    if constexpr (NCols >= 4) {
        accLo1 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
        accHi1 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
        accLo2 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
        accHi2 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
        accLo3 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
        accHi3 = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
    }

    const uint8_t* qb0 = reinterpret_cast<const uint8_t*>(QuantBDataColPtr);
    [[maybe_unused]] const uint8_t* qb1 = qb0 + StrideQuantBData;
    [[maybe_unused]] const uint8_t* qb2 = qb0 + 2 * StrideQuantBData;
    [[maybe_unused]] const uint8_t* qb3 = qb0 + 3 * StrideQuantBData;

    const float* qs0 = QuantBScaleColPtr;
    [[maybe_unused]] const float* qs1 = qs0 + StrideQuantBScale;
    [[maybe_unused]] const float* qs2 = qs0 + 2 * StrideQuantBScale;
    [[maybe_unused]] const float* qs3 = qs0 + 3 * StrideQuantBScale;

    [[maybe_unused]] size_t zpIdx = 0;

    for (size_t k = 0; k < CountK; k += BlkLen) {
        const size_t k_blk_len = std::min(CountK - k, BlkLen);

        const float scale0 = *qs0++;
        [[maybe_unused]] float scale1, scale2, scale3;
        if constexpr (NCols >= 4) {
            scale1 = *qs1++;
            scale2 = *qs2++;
            scale3 = *qs3++;
        }

        float offset_v0;
        [[maybe_unused]] float offset_v1, offset_v2, offset_v3;
        if constexpr (HasZeroPoint) {
            auto getZp = [&](size_t col_off) -> float {
                const std::byte zp_packed =
                    QuantBZeroPointColPtr[col_off + zpIdx / 2];
                const std::byte zp = ((zpIdx & 1) == 1)
                                         ? (zp_packed >> 4)
                                         : (zp_packed & std::byte{0x0F});
                return static_cast<float>(std::to_integer<uint8_t>(zp));
            };
            offset_v0 = getZp(0);
            if constexpr (NCols >= 4) {
                offset_v1 = getZp(StrideQuantBZeroPoint);
                offset_v2 = getZp(2 * StrideQuantBZeroPoint);
                offset_v3 = getZp(3 * StrideQuantBZeroPoint);
            }
            zpIdx++;
        } else {
            offset_v0 = 8.0f;
            if constexpr (NCols >= 4) {
                offset_v1 = 8.0f;
                offset_v2 = 8.0f;
                offset_v3 = 8.0f;
            }
        }


        // Full-SubBlk fast path: process floor(k_blk_len / 16) iterations with
        // no per-iteration branches and a precomputed vl=8.
        const size_t full_subblks = k_blk_len / kSubBlkLen;
        const size_t tail_kk      = full_subblks * kSubBlkLen;
        for (size_t s = 0; s < full_subblks; ++s) {
            const size_t kk = s * kSubBlkLen;

            // Software prefetch: pull A and B ahead of the working position.
            __builtin_prefetch(ARowPtr + k + kk + 64, 0, 0);
            __builtin_prefetch(qb0 + 64, 0, 0);
            if constexpr (NCols >= 4) {
                __builtin_prefetch(qb1 + 64, 0, 0);
                __builtin_prefetch(qb2 + 64, 0, 0);
                __builtin_prefetch(qb3 + 64, 0, 0);
            }

            vfloat32m1_t aLo = __riscv_vle32_v_f32m1(ARowPtr + k + kk, vl8);
            vfloat32m1_t aHi = __riscv_vle32_v_f32m1(ARowPtr + k + kk + 8, vl8);

            DoOneColSubBlkFull(qb0, scale0, offset_v0, aLo, aHi, vl8, accLo0, accHi0);
            qb0 += kSubBlkBytes;
            if constexpr (NCols >= 4) {
                DoOneColSubBlkFull(qb1, scale1, offset_v1, aLo, aHi, vl8, accLo1, accHi1);
                qb1 += kSubBlkBytes;
                DoOneColSubBlkFull(qb2, scale2, offset_v2, aLo, aHi, vl8, accLo2, accHi2);
                qb2 += kSubBlkBytes;
                DoOneColSubBlkFull(qb3, scale3, offset_v3, aLo, aHi, vl8, accLo3, accHi3);
                qb3 += kSubBlkBytes;
            }
        }

        // Tail: at most one partial SubBlk left.
        if (tail_kk < k_blk_len) {
            const size_t kk = tail_kk;
            const size_t k_sb_len = k_blk_len - kk;
            const size_t lo_len = std::min(k_sb_len, size_t{8});
            const size_t hi_len = (k_sb_len > 8) ? (k_sb_len - 8) : 0;

            const size_t vl_lo = __riscv_vsetvl_e32m1(lo_len);
            vfloat32m1_t aLo   = __riscv_vle32_v_f32m1(ARowPtr + k + kk, vl_lo);
            vfloat32m1_t aHi   = __riscv_vfmv_v_f_f32m1(0.0f, vl8);
            size_t       vl_hi = 0;
            if (hi_len > 0) {
                vl_hi = __riscv_vsetvl_e32m1(hi_len);
                aHi   = __riscv_vle32_v_f32m1(ARowPtr + k + kk + 8, vl_hi);
            }

            DoOneColSubBlkTail(qb0, scale0, offset_v0, aLo, aHi, vl_lo, vl_hi, accLo0, accHi0);
            qb0 += kSubBlkBytes;
            if constexpr (NCols >= 4) {
                DoOneColSubBlkTail(qb1, scale1, offset_v1, aLo, aHi, vl_lo, vl_hi, accLo1, accHi1);
                qb1 += kSubBlkBytes;
                DoOneColSubBlkTail(qb2, scale2, offset_v2, aLo, aHi, vl_lo, vl_hi, accLo2, accHi2);
                qb2 += kSubBlkBytes;
                DoOneColSubBlkTail(qb3, scale3, offset_v3, aLo, aHi, vl_lo, vl_hi, accLo3, accHi3);
                qb3 += kSubBlkBytes;
            }
        }
    }

    SumPtr[0] += ReduceLoHiToScalar(accLo0, accHi0);
    if constexpr (NCols >= 4) {
        SumPtr[1] += ReduceLoHiToScalar(accLo1, accHi1);
        SumPtr[2] += ReduceLoHiToScalar(accLo2, accHi2);
        SumPtr[3] += ReduceLoHiToScalar(accLo3, accHi3);
    }
}

template <bool HasZeroPoint>
void
SQ4BitGemmM1Kernel_CompFp32_Impl(
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
    const size_t StrideQuantBData =
        BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    const size_t StrideQuantBScale = BlockCountK;
    const size_t StrideQuantBZeroPoint =
        MlasQNBitZeroPointsForBlksSizeInBytes<kBlkBitWidth>(BlockCountK);

    constexpr size_t kCols = 4;

    // K-tile selection: target ~16 KB of A per tile so it stays L1-resident
    // across the inner N sweep. Each tile must contain an EVEN number of
    // K-blocks so each tile boundary lands on a low-nibble byte position
    // for the zero-point packing.
    constexpr size_t kTileTargetFloats = 4096;          // 16 KB
    const size_t blocks_per_tile_target = std::max(size_t{2}, kTileTargetFloats / BlkLen);
    const size_t blocks_per_tile        = (blocks_per_tile_target / 2) * 2;  // even
    const size_t k_tile_full            = blocks_per_tile * BlkLen;

    // Zero-init C so the inner kernel can always *add* to existing values.
    std::memset(C, 0, sizeof(float) * CountN);

    for (size_t kt = 0; kt < CountK; kt += k_tile_full) {
        const size_t k_count_tile = std::min(k_tile_full, CountK - kt);

        // Pointer offsets for this tile (caller-side per-tile advance).
        const size_t blk_skipped         = kt / BlkLen;             // even by construction
        const size_t bytes_skipped_data  = blk_skipped * (BlkLen / 2);
        const size_t zp_bytes_skipped    = blk_skipped / 2;         // 2 nibbles per byte

        const std::byte* QuantBDataColPtr      = QuantBData      + bytes_skipped_data;
        const float*     QuantBScaleColPtr     = QuantBScale     + blk_skipped;
        const std::byte* QuantBZeroPointColPtr =
            HasZeroPoint ? (QuantBZeroPoint + zp_bytes_skipped) : nullptr;
        float*           SumPtr                = C;

        int64_t nblk = static_cast<int64_t>(CountN) - kCols;
        while (nblk >= 0) {
            ComputeDotProducts_NCols<kCols, HasZeroPoint>(
                BlkLen, A + kt,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr,
                k_count_tile,
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                SumPtr);

            QuantBDataColPtr  += kCols * StrideQuantBData;
            QuantBScaleColPtr += kCols * StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += kCols * StrideQuantBZeroPoint;
            }
            SumPtr += kCols;
            nblk   -= kCols;
        }

        nblk += kCols;
        for (int64_t n = 0; n < nblk; ++n) {
            ComputeDotProducts_NCols<1, HasZeroPoint>(
                BlkLen, A + kt,
                QuantBDataColPtr, QuantBScaleColPtr, QuantBZeroPointColPtr,
                k_count_tile,
                StrideQuantBData, StrideQuantBScale, StrideQuantBZeroPoint,
                SumPtr);

            QuantBDataColPtr  += StrideQuantBData;
            QuantBScaleColPtr += StrideQuantBScale;
            if constexpr (HasZeroPoint) {
                QuantBZeroPointColPtr += StrideQuantBZeroPoint;
            }
            SumPtr += 1;
        }
    }

    // Bias once at the end (not per tile).
    if (Bias != nullptr) {
        for (size_t n = 0; n < CountN; ++n) {
            C[n] += Bias[n];
        }
    }
}

void
SQ4BitGemmM1Kernel_CompFp32(
    size_t           BlkLen,
    const float*     A,
    const std::byte* QuantBData,
    const float*     QuantBScale,
    const std::byte* QuantBZeroPoint,
    float*           C,
    size_t           CountN,
    size_t           CountK,
    size_t           BlockCountK,
    const float*     Bias)
{
    if (QuantBZeroPoint != nullptr) {
        SQ4BitGemmM1Kernel_CompFp32_Impl<true>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C,
            CountN, CountK, BlockCountK, Bias);
    } else {
        SQ4BitGemmM1Kernel_CompFp32_Impl<false>(
            BlkLen, A, QuantBData, QuantBScale, QuantBZeroPoint, C,
            CountN, CountK, BlockCountK, Bias);
    }
}

//
// Dequantize one column block (16 K rows) and write into the 16-wide K-major
// destination layout. The destination layout for a 16-column wide SgemmCopyPackB
// region is `Dst[K_row * 16 + N_col]`. We write 16 K-rows of one column at a
// time using a strided store with byte stride 64 (= 16 floats).
//

template <bool HasZeroPoint>
MLAS_FORCEINLINE void
DequantOneColInto16WidePack(
    const uint8_t* qbSrc,           // packed B, 8 bytes for 16 nibbles
    float          scale,
    float          offset_v,
    float*         DstColPtr,       // address of [K_row=0][N_col]
    size_t         k_sb_len)        // 1..16 valid K rows in this SubBlk
{
    MLAS_UNREFERENCED_PARAMETER(HasZeroPoint);

    vfloat32m1_t loF, hiF;
    LoadDequant16Nibbles_LoHi_m1(qbSrc, loF, hiF);

    const size_t vl_lo = __riscv_vsetvl_e32m1(std::min(k_sb_len, size_t{8}));
    loF = __riscv_vfsub_vf_f32m1(loF, offset_v, vl_lo);
    loF = __riscv_vfmul_vf_f32m1(loF, scale,    vl_lo);
    // Strided store: write vl_lo floats with stride 16 floats = 64 bytes.
    __riscv_vsse32_v_f32m1(DstColPtr, /*byte_stride=*/64, loF, vl_lo);

    if (k_sb_len > 8) {
        const size_t vl_hi = __riscv_vsetvl_e32m1(k_sb_len - 8);
        hiF = __riscv_vfsub_vf_f32m1(hiF, offset_v, vl_hi);
        hiF = __riscv_vfmul_vf_f32m1(hiF, scale,    vl_hi);
        __riscv_vsse32_v_f32m1(DstColPtr + 8 * 16, 64, hiF, vl_hi);
    }
}

template <bool HasZeroPoint>
void
Q4BitBlkDequantBForSgemm_CompFp32_Impl(
    size_t           BlkLen,
    float*           FpData,
    const std::byte* QuantBData,
    const float*     QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t           CountN,
    size_t           CountK,
    size_t           BlockCountK)
{
    const size_t StrideQuantBData =
        BlockCountK * MlasQNBitBlkDataSizeInBytes(kBlkBitWidth, BlkLen);
    [[maybe_unused]] const size_t StrideQuantBZeroPoint =
        MlasQNBitZeroPointsForBlksSizeInBytes<kBlkBitWidth>(BlockCountK);

    float* Dst = FpData;

    const std::byte* QuantBDataCol      = QuantBData;
    const float*     QuantBScaleCol     = QuantBScale;
    [[maybe_unused]] const std::byte* QuantBZeroPointCol = QuantBZeroPoint;

    // Output is laid out as a sequence of 16-column-wide K-major panels.
    // For each panel of up to 16 columns we write a 16-row x 16-col tile per
    // SubBlk, repeated across all SubBlks (and zero-padded for n < 16).
    //
    // Within a single panel, the destination buffer advances by
    //   16 * std::min(remaining_in_block, 16)  floats
    // every 16 K-rows worth of dequantization, matching the Neon code path.

    auto run_panel = [&](size_t panel_cols) {
        // panel_cols is in [1..16]. We zero the panel each SubBlk for safety
        // when panel_cols < 16 (so the trailing 16 - panel_cols columns are 0).
        const bool need_zero_pad = (panel_cols < 16);

        float* DstSubBlkBase = Dst;

        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            const size_t k_blk_len = std::min(CountK - k, BlkLen);

            for (size_t kk = 0; kk < k_blk_len; kk += kSubBlkLen) {
                const size_t k_sb_len = std::min(k_blk_len - kk, kSubBlkLen);

                if (need_zero_pad) {
                    // Write 16 * 16 zeros, then overwrite the valid columns.
                    const size_t vl16z = __riscv_vsetvl_e32m2(16);
                    vfloat32m2_t zero_v = __riscv_vfmv_v_f_f32m2(0.0f, vl16z);
                    for (size_t r = 0; r < 16; ++r) {
                        __riscv_vse32_v_f32m2(DstSubBlkBase + r * 16, zero_v, vl16z);
                    }
                }

                for (size_t nn = 0; nn < panel_cols; ++nn) {
                    const float scale = QuantBScaleCol[nn * BlockCountK + k_blk_idx];

                    float offset_v;
                    if constexpr (HasZeroPoint) {
                        const std::byte zp_packed =
                            QuantBZeroPointCol[nn * StrideQuantBZeroPoint + k_blk_idx / 2];
                        const std::byte zp = ((k_blk_idx & 1) == 1)
                                                 ? (zp_packed >> 4)
                                                 : (zp_packed & std::byte{0x0F});
                        offset_v = static_cast<float>(std::to_integer<uint8_t>(zp));
                    } else {
                        offset_v = 8.0f;
                    }

                    const uint8_t* qbSrc =
                        reinterpret_cast<const uint8_t*>(QuantBDataCol)
                        + nn * StrideQuantBData
                        + (k + kk) * kBlkBitWidth / 8;

                    DequantOneColInto16WidePack<HasZeroPoint>(
                        qbSrc, scale, offset_v,
                        DstSubBlkBase + nn,
                        k_sb_len);
                }

                DstSubBlkBase += 16 * std::min(k_sb_len, size_t{16});
            }
        }

        Dst = DstSubBlkBase;
    };

    while (CountN >= 16) {
        run_panel(16);
        QuantBDataCol  += 16 * StrideQuantBData;
        QuantBScaleCol += 16 * BlockCountK;
        if constexpr (HasZeroPoint) {
            QuantBZeroPointCol += 16 * StrideQuantBZeroPoint;
        }
        CountN -= 16;
    }

    if (CountN > 0) {
        run_panel(CountN);
    }
}

void
Q4BitBlkDequantBForSgemm_CompFp32(
    size_t           BlkLen,
    float*           FpData,
    const std::byte* QuantBData,
    const float*     QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t           CountN,
    size_t           CountK,
    size_t           BlockCountK)
{
    if (QuantBZeroPoint != nullptr) {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<true>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint,
            CountN, CountK, BlockCountK);
    } else {
        Q4BitBlkDequantBForSgemm_CompFp32_Impl<false>(
            BlkLen, FpData, QuantBData, QuantBScale, QuantBZeroPoint,
            CountN, CountK, BlockCountK);
    }
}

}  // namespace
}  // namespace sqnbitgemm_riscv

//
// Dispatch struct definition.
//

const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchRiscv = []() {
    MLAS_SQNBIT_GEMM_DISPATCH d;

    d.SQ4BitGemmPackQuantBDataSize = sqnbitgemm_riscv::SQ4BitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData     = sqnbitgemm_riscv::SQ4BitGemmPackQuantBData;

    d.SQ4BitGemmPerGemmWorkspaceSize      = sqnbitgemm_riscv::SQ4BitGemmPerGemmWorkspaceSize;
    d.SQ4BitGemmPerGemmWorkspaceAlignment = sqnbitgemm_riscv::SQ4BitGemmPerGemmWorkspaceAlignment;

    d.SQ4BitGemmM1Kernel_CompFp32       = sqnbitgemm_riscv::SQ4BitGemmM1Kernel_CompFp32;
    d.Q4BitBlkDequantBForSgemm_CompFp32 = sqnbitgemm_riscv::Q4BitBlkDequantBForSgemm_CompFp32;

    // CompInt8 paths intentionally left null.
    return d;
}();
