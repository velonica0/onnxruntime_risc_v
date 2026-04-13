/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    PoolingKernelRVV.cpp

Abstract:

    RVV (RISC-V Vector 1.0) implementation of the 2D float pooling vector
    kernels (Max, Average-IncludePad, Average-ExcludePad). Specialized for
    VLEN=256 (e32m2 -> 16 lanes), but uses vsetvl-based tail handling so
    the code remains correct on other VLENs.

    The CNN-typical hot path is 2D pooling with kernel 2x2 or 3x3, stride
    1 or 2. The original `MlasPool2DVectorKernel` template in pooling.cpp
    uses MLAS_FLOAT32X4 (4-lane abstraction). On RISC-V that abstraction
    falls into the GCC `vector_size(16)` path which autovectorizes to RVV
    at VL=4 — only 1/4 of the 16-lane width available at VLEN=256.

    This file provides three native f32m2 kernels:

        MlasPool2DMaxVectorKernel_RVV
        MlasPool2DAvgIncludePadVectorKernel_RVV
        MlasPool2DAvgExcludePadVectorKernel_RVV

    They are wired into pooling.cpp's MlasPoolVectorKernels[][2] dispatch
    table via a `#if defined(MLAS_TARGET_RISCV64)` switch.

    Algorithm: same two-phase structure as the original.
      Phase 1 (H-reduction): for each output row, fold the kernel-height
        rows of A into a 1D ReductionBuffer using vfmax / vfadd, processing
        16 width positions per pass.
      Phase 2 (W-reduction): slide a window of KernelWidth across the
        ReductionBuffer producing OutputWidth outputs per row. Stride 1
        uses contiguous f32m2 loads; stride 2 uses strided f32m2 loads
        (vlse32 with byte stride 8).

    AvgPool ExcludePad: divisor varies per output column (= count of
    valid input positions in that window). The kernel precomputes the
    per-column divisor vector for the current output row at the start of
    each output row.

--*/

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <riscv_vector.h>

#include "mlasi.h"

// MLAS_POOL_WORK_BLOCK is defined inside pooling.cpp's anonymous TU and not
// exposed in a header. Re-declare a layout-compatible struct here. Field
// names mirror pooling.cpp; layout MUST match.
struct MLAS_POOL_WORK_BLOCK_RVV
{
    int      PoolingKind;
    size_t   InputShape[3];
    size_t   InputSize;
    size_t   OutputShape[3];
    int64_t  KernelShape[3];
    int64_t  Padding[6];
    int64_t  StrideShape[3];
};

// Pooling-kind enum values must match MLAS_POOLING_KIND in mlas.h.
// Reproduced here so this file does not depend on a header that varies.
static constexpr int kMlasMaximumPooling          = 0;
static constexpr int kMlasAveragePoolingExcludePad = 1;
static constexpr int kMlasAveragePoolingIncludePad = 2;

// Stack size of the per-output-row reduction buffer (must match the
// constant in pooling.cpp).
static constexpr size_t kReductionBufferStack = 2048;

namespace
{

// =================================================================
// Phase 1: H-reduction (kernel height) into ReductionBuffer.
// =================================================================
//
// For one output row, fold InputRowsCount input rows into a single 1D
// vector of length InputWidth. Op is vfmax for MaxPool or vfadd for
// AvgPool. Caller has already filled the padding cells (left/right of
// the InputWidth window) with the appropriate identity element.
template <bool IsMax>
inline void
HReduceRow(
    const float* InputRowStart,
    size_t       InputRowsCount,
    size_t       InputWidth,
    size_t       InputWidthStride,    // bytes per row (== InputWidth*sizeof(float))
    float*       ReductionOutput)
{
    size_t remaining = InputWidth;
    const float* row0 = InputRowStart;

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(remaining);
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(row0, vl);

        const float* rN = row0 + (InputWidthStride / sizeof(float));
        for (size_t r = 0; r < InputRowsCount; ++r) {
            vfloat32m2_t v = __riscv_vle32_v_f32m2(rN, vl);
            if constexpr (IsMax) {
                acc = __riscv_vfmax_vv_f32m2(acc, v, vl);
            } else {
                acc = __riscv_vfadd_vv_f32m2(acc, v, vl);
            }
            rN += (InputWidthStride / sizeof(float));
        }

        __riscv_vse32_v_f32m2(ReductionOutput, acc, vl);
        ReductionOutput += vl;
        row0            += vl;
        remaining       -= vl;
    }
}

// =================================================================
// Phase 2a: W-reduction with stride 1 (Max).
// =================================================================
inline void
WReduceMaxStride1(
    const float* ReductionBuffer,
    size_t       OutputWidth,
    int64_t      KernelWidth,
    float*       Output)
{
    size_t remaining = OutputWidth;
    const float* src = ReductionBuffer;

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(remaining);
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(src, vl);
        for (int64_t k = 1; k < KernelWidth; ++k) {
            vfloat32m2_t v = __riscv_vle32_v_f32m2(src + k, vl);
            acc = __riscv_vfmax_vv_f32m2(acc, v, vl);
        }
        __riscv_vse32_v_f32m2(Output, acc, vl);
        Output    += vl;
        src       += vl;
        remaining -= vl;
    }
}

// =================================================================
// Phase 2b: W-reduction with stride 2 (Max).
// =================================================================
inline void
WReduceMaxStride2(
    const float* ReductionBuffer,
    size_t       OutputWidth,
    int64_t      KernelWidth,
    float*       Output)
{
    size_t remaining = OutputWidth;
    const float* src = ReductionBuffer;
    const ptrdiff_t bs = static_cast<ptrdiff_t>(2 * sizeof(float));

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(remaining);
        vfloat32m2_t acc = __riscv_vlse32_v_f32m2(src, bs, vl);
        for (int64_t k = 1; k < KernelWidth; ++k) {
            vfloat32m2_t v = __riscv_vlse32_v_f32m2(src + k, bs, vl);
            acc = __riscv_vfmax_vv_f32m2(acc, v, vl);
        }
        __riscv_vse32_v_f32m2(Output, acc, vl);
        Output    += vl;
        src       += 2 * vl;
        remaining -= vl;
    }
}

// =================================================================
// Phase 2a/2b for Average pool, IncludePad: same as Max but with
// vfadd and a constant divisor applied at the end.
// =================================================================
inline void
WReduceAvgStride1(
    const float* ReductionBuffer,
    size_t       OutputWidth,
    int64_t      KernelWidth,
    float        Divisor,        // NB: divisor, not inverse — must use vfdiv to bit-match scalar `m /= K`
    float*       Output)
{
    size_t remaining = OutputWidth;
    const float* src = ReductionBuffer;

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(remaining);
        vfloat32m2_t acc = __riscv_vle32_v_f32m2(src, vl);
        for (int64_t k = 1; k < KernelWidth; ++k) {
            vfloat32m2_t v = __riscv_vle32_v_f32m2(src + k, vl);
            acc = __riscv_vfadd_vv_f32m2(acc, v, vl);
        }
        acc = __riscv_vfdiv_vf_f32m2(acc, Divisor, vl);
        __riscv_vse32_v_f32m2(Output, acc, vl);
        Output    += vl;
        src       += vl;
        remaining -= vl;
    }
}

inline void
WReduceAvgStride2(
    const float* ReductionBuffer,
    size_t       OutputWidth,
    int64_t      KernelWidth,
    float        Divisor,
    float*       Output)
{
    size_t remaining = OutputWidth;
    const float* src = ReductionBuffer;
    const ptrdiff_t bs = static_cast<ptrdiff_t>(2 * sizeof(float));

    while (remaining > 0) {
        const size_t vl = __riscv_vsetvl_e32m2(remaining);
        vfloat32m2_t acc = __riscv_vlse32_v_f32m2(src, bs, vl);
        for (int64_t k = 1; k < KernelWidth; ++k) {
            vfloat32m2_t v = __riscv_vlse32_v_f32m2(src + k, bs, vl);
            acc = __riscv_vfadd_vv_f32m2(acc, v, vl);
        }
        acc = __riscv_vfdiv_vf_f32m2(acc, Divisor, vl);
        __riscv_vse32_v_f32m2(Output, acc, vl);
        Output    += vl;
        src       += 2 * vl;
        remaining -= vl;
    }
}

// =================================================================
// Phase 2 for Average pool ExcludePad: per-output divisor varies with
// the count of valid input positions in the window. Build a per-column
// divisor table for the current output row, then apply it.
// =================================================================
inline void
WReduceAvgExcludeStride(
    const float* ReductionBuffer,
    size_t       OutputWidth,
    int64_t      KernelWidth,
    int64_t      InputWidth,
    int64_t      PaddingLeftWidth,
    int64_t      StrideWidth,
    float        InputRowsCount,  // count of valid H rows in this output row
    float*       Output)
{
    // For each output position p, compute:
    //   iwStart = p*StrideWidth - PaddingLeftWidth   (clamped to [0, InputWidth])
    //   iwEnd   = p*StrideWidth - PaddingLeftWidth + KernelWidth (clamped)
    //   divisor = InputRowsCount * (iwEnd - iwStart)
    //
    // Inverse divisor is multiplied with the (sum-pooled) value.
    // We compute these with scalar arithmetic (output is small, this is
    // fast and avoids vector-mask gymnastics).
    for (size_t p = 0; p < OutputWidth; ++p) {
        int64_t iwStart = static_cast<int64_t>(p) * StrideWidth - PaddingLeftWidth;
        int64_t iwEnd   = iwStart + KernelWidth;
        if (iwStart < 0)            iwStart = 0;
        if (iwEnd   > InputWidth)   iwEnd   = InputWidth;
        const float div = InputRowsCount * static_cast<float>(iwEnd - iwStart);

        // Reduce one window via scalar walk over the buffer (rare path).
        const float* w = ReductionBuffer + (size_t)(p) * (size_t)StrideWidth;
        float acc = w[0];
        for (int64_t k = 1; k < KernelWidth; ++k) acc += w[k];
        Output[p] = acc / div;
    }
}

// =================================================================
// Top-level kernels (one per dispatch slot).
// =================================================================

template <bool IsMax, int AvgKind /* 0=N/A, 1=ExcludePad, 2=IncludePad */>
inline void
Pool2DKernelRVV_Impl(
    const MLAS_POOL_WORK_BLOCK_RVV* WorkBlock,
    size_t        ChannelCount,
    const float*  Input,
    float*        Output)
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex  = 1;
    constexpr size_t Dimensions       = 2;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth  = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize   = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth  = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth  = WorkBlock->KernelShape[WidthShapeIndex];
    const size_t  PaddingLeftHeight = static_cast<size_t>(WorkBlock->Padding[HeightShapeIndex]);
    const size_t  PaddingLeftWidth  = static_cast<size_t>(WorkBlock->Padding[WidthShapeIndex]);
    const size_t  PaddingRightWidth = static_cast<size_t>(WorkBlock->Padding[Dimensions + WidthShapeIndex]);
    const int64_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const int64_t StrideWidth  = WorkBlock->StrideShape[WidthShapeIndex];

    // Reduction buffer with padding-pad on both sides.
    constexpr size_t kBufferPad = 16;  // safety margin (one m2 vector worth)
    float ReductionBuffer[kReductionBufferStack];

    // Fill the left-pad and right-pad regions of the buffer with the
    // identity element (lowest float for max, 0 for avg).
    constexpr float kIdentity = IsMax ? -std::numeric_limits<float>::infinity() : 0.0f;
    for (size_t i = 0; i < PaddingLeftWidth; ++i) ReductionBuffer[i] = kIdentity;
    const size_t right_start = PaddingLeftWidth + InputWidth;
    const size_t right_end   = right_start + PaddingRightWidth + kBufferPad;
    for (size_t i = right_start; i < right_end && i < kReductionBufferStack; ++i) {
        ReductionBuffer[i] = kIdentity;
    }

    // Constant divisor for the IncludePad case. We pass the divisor (not its
    // inverse) to the W-reduce so we can use vfdiv and bit-match the scalar
    // reference's `m /= K`.
    [[maybe_unused]] const float divIncludePad =
        static_cast<float>(KernelHeight * KernelWidth);

    for (size_t c = 0; c < ChannelCount; ++c) {
        for (size_t ph = 0; ph < OutputHeight; ++ph) {
            // Compute valid H window
            int64_t ihStart64 = static_cast<int64_t>(ph) * StrideHeight - static_cast<int64_t>(PaddingLeftHeight);
            int64_t ihEnd64   = ihStart64 + KernelHeight;
            const size_t ihStart = ihStart64 < 0 ? 0 : static_cast<size_t>(ihStart64);
            const size_t ihEnd   = ihEnd64   > static_cast<int64_t>(InputHeight)
                                       ? InputHeight
                                       : static_cast<size_t>(ihEnd64);
            const size_t inputRowsCount = ihEnd - ihStart;
            if (inputRowsCount == 0) {
                // Output row entirely from padding
                for (size_t i = 0; i < OutputWidth; ++i) Output[i] = kIdentity;
                Output += OutputWidth;
                continue;
            }
            const size_t inputRowsMinusOne = inputRowsCount - 1;

            // Phase 1: H-reduce into ReductionBuffer + PaddingLeftWidth..
            HReduceRow<IsMax>(
                Input + ihStart * InputWidth,
                inputRowsMinusOne,
                InputWidth,
                InputWidth * sizeof(float),
                ReductionBuffer + PaddingLeftWidth);

            // Phase 2: W-reduce.
            if constexpr (IsMax) {
                if (StrideWidth == 1) {
                    WReduceMaxStride1(ReductionBuffer, OutputWidth, KernelWidth, Output);
                } else if (StrideWidth == 2) {
                    WReduceMaxStride2(ReductionBuffer, OutputWidth, KernelWidth, Output);
                } else {
                    // Rare: scalar fallback for arbitrary stride
                    for (size_t p = 0; p < OutputWidth; ++p) {
                        const float* w = ReductionBuffer + (size_t)p * (size_t)StrideWidth;
                        float acc = w[0];
                        for (int64_t k = 1; k < KernelWidth; ++k) {
                            if (w[k] > acc) acc = w[k];
                        }
                        Output[p] = acc;
                    }
                }
            } else if constexpr (AvgKind == 2) {  // IncludePad
                if (StrideWidth == 1) {
                    WReduceAvgStride1(ReductionBuffer, OutputWidth, KernelWidth, divIncludePad, Output);
                } else if (StrideWidth == 2) {
                    WReduceAvgStride2(ReductionBuffer, OutputWidth, KernelWidth, divIncludePad, Output);
                } else {
                    for (size_t p = 0; p < OutputWidth; ++p) {
                        const float* w = ReductionBuffer + (size_t)p * (size_t)StrideWidth;
                        float acc = w[0];
                        for (int64_t k = 1; k < KernelWidth; ++k) acc += w[k];
                        Output[p] = acc / divIncludePad;
                    }
                }
            } else {  // ExcludePad: per-column divisor
                WReduceAvgExcludeStride(
                    ReductionBuffer, OutputWidth, KernelWidth,
                    static_cast<int64_t>(InputWidth),
                    static_cast<int64_t>(PaddingLeftWidth),
                    StrideWidth,
                    static_cast<float>(inputRowsCount),
                    Output);
            }

            Output += OutputWidth;
        }

        Input += InputSize;
    }
}

}  // namespace

// =================================================================
// Public entry points (called from pooling.cpp's dispatch table).
// =================================================================

extern "C" void
MlasPool2DMaxVectorKernel_RVV(
    const void*   WorkBlockPtr,
    size_t        ChannelCount,
    const float*  Input,
    float*        Output)
{
    Pool2DKernelRVV_Impl<true, 0>(
        reinterpret_cast<const MLAS_POOL_WORK_BLOCK_RVV*>(WorkBlockPtr),
        ChannelCount, Input, Output);
}

extern "C" void
MlasPool2DAvgIncludePadVectorKernel_RVV(
    const void*   WorkBlockPtr,
    size_t        ChannelCount,
    const float*  Input,
    float*        Output)
{
    Pool2DKernelRVV_Impl<false, 2>(
        reinterpret_cast<const MLAS_POOL_WORK_BLOCK_RVV*>(WorkBlockPtr),
        ChannelCount, Input, Output);
}

extern "C" void
MlasPool2DAvgExcludePadVectorKernel_RVV(
    const void*   WorkBlockPtr,
    size_t        ChannelCount,
    const float*  Input,
    float*        Output)
{
    Pool2DKernelRVV_Impl<false, 1>(
        reinterpret_cast<const MLAS_POOL_WORK_BLOCK_RVV*>(WorkBlockPtr),
        ChannelCount, Input, Output);
}
