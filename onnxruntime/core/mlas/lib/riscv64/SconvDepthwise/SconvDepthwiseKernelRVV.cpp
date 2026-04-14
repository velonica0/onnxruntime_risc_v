/*++

Copyright (c) Microsoft Corporation. All rights reserved.
Copyright (c) SpacemiT. All rights reserved.

Licensed under the MIT License.

Module Name:

    SconvDepthwiseKernelRVV.cpp

Abstract:

    This module implements the RVV-optimized kernels for the single precision
    depthwise 3x3 convolution.

--*/

#include "mlasi.h"
#include <riscv_vector.h>

static
void
MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1_RVV(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
{
    const size_t W = Parameters->InputShape[1];
    const float beta = Parameters->Beta;

    if (W > 1) {

        const float w00 = Filter[0];
        const float w01 = Filter[1];
        const float w02 = Filter[2];
        const float w10 = Filter[3];
        const float w11 = Filter[4];
        const float w12 = Filter[5];
        const float w20 = Filter[6];
        const float w21 = Filter[7];
        const float w22 = Filter[8];

        const size_t H = Parameters->InputShape[0];
        const size_t pad_top = Parameters->Padding[0];
        const size_t pad_left = Parameters->Padding[1];
        const size_t stride_h = Parameters->StrideShape[0];
        const size_t stride_w = Parameters->StrideShape[1];

        const size_t pad_right = (((Parameters->OutputShape[1] - 1) * stride_w + 3) > (pad_left + W)) ? 1 : 0;

        const float* row0 = (pad_top > 0) ? Zeros : (Input - pad_left);
        const float* row1 = (H + pad_top <= 1) ? Zeros : (Input + (1 - pad_top) * W) - pad_left;
        const float* row2 = (H + pad_top <= 2) ? Zeros : (row1 + W);

        for (size_t h = 0, out_row = Parameters->OutputShape[0]; out_row > 0; --out_row) {
            auto out_col = Parameters->OutputShape[1];

            // Handle left padding column (scalar, single element)
            if (pad_left == 1) {
                float dotsum = w01 * row0[1] + w02 * row0[2] + w11 * row1[1] + w12 * row1[2] +
                               w21 * row2[1] + w22 * row2[2] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
                out_col--;
                row0 += stride_w;
                row1 += stride_w;
                row2 += stride_w;
            }

            // RVV-vectorized main body: process multiple output columns in parallel
            {
                size_t body_cols = (out_col > pad_right) ? (out_col - pad_right) : 0;

                if (stride_w == 1) {
                    // Optimized path for stride_w == 1: contiguous loads
                    size_t n = body_cols;
                    while (n > 0) {
                        size_t vl = __riscv_vsetvl_e32m4(n);

                        vfloat32m4_t vr0_0 = __riscv_vle32_v_f32m4(row0 + 0, vl);
                        vfloat32m4_t vr0_1 = __riscv_vle32_v_f32m4(row0 + 1, vl);
                        vfloat32m4_t vr0_2 = __riscv_vle32_v_f32m4(row0 + 2, vl);
                        vfloat32m4_t vr1_0 = __riscv_vle32_v_f32m4(row1 + 0, vl);
                        vfloat32m4_t vr1_1 = __riscv_vle32_v_f32m4(row1 + 1, vl);
                        vfloat32m4_t vr1_2 = __riscv_vle32_v_f32m4(row1 + 2, vl);
                        vfloat32m4_t vr2_0 = __riscv_vle32_v_f32m4(row2 + 0, vl);
                        vfloat32m4_t vr2_1 = __riscv_vle32_v_f32m4(row2 + 1, vl);
                        vfloat32m4_t vr2_2 = __riscv_vle32_v_f32m4(row2 + 2, vl);

                        vfloat32m4_t vacc;
                        if (beta != 0.f) {
                            vfloat32m4_t vout = __riscv_vle32_v_f32m4(Output, vl);
                            vacc = __riscv_vfmul_vf_f32m4(vout, beta, vl);
                        } else {
                            vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                        }

                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w00, vr0_0, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w01, vr0_1, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w02, vr0_2, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w10, vr1_0, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w11, vr1_1, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w12, vr1_2, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w20, vr2_0, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w21, vr2_1, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w22, vr2_2, vl);

                        __riscv_vse32_v_f32m4(Output, vacc, vl);

                        Output += vl;
                        row0 += vl;
                        row1 += vl;
                        row2 += vl;
                        n -= vl;
                    }
                } else {
                    // General path for stride_w > 1: strided loads
                    ptrdiff_t byte_stride = (ptrdiff_t)stride_w * (ptrdiff_t)sizeof(float);
                    size_t n = body_cols;
                    while (n > 0) {
                        size_t vl = __riscv_vsetvl_e32m4(n);

                        vfloat32m4_t vr0_0 = __riscv_vlse32_v_f32m4(row0 + 0, byte_stride, vl);
                        vfloat32m4_t vr0_1 = __riscv_vlse32_v_f32m4(row0 + 1, byte_stride, vl);
                        vfloat32m4_t vr0_2 = __riscv_vlse32_v_f32m4(row0 + 2, byte_stride, vl);
                        vfloat32m4_t vr1_0 = __riscv_vlse32_v_f32m4(row1 + 0, byte_stride, vl);
                        vfloat32m4_t vr1_1 = __riscv_vlse32_v_f32m4(row1 + 1, byte_stride, vl);
                        vfloat32m4_t vr1_2 = __riscv_vlse32_v_f32m4(row1 + 2, byte_stride, vl);
                        vfloat32m4_t vr2_0 = __riscv_vlse32_v_f32m4(row2 + 0, byte_stride, vl);
                        vfloat32m4_t vr2_1 = __riscv_vlse32_v_f32m4(row2 + 1, byte_stride, vl);
                        vfloat32m4_t vr2_2 = __riscv_vlse32_v_f32m4(row2 + 2, byte_stride, vl);

                        vfloat32m4_t vacc;
                        if (beta != 0.f) {
                            vfloat32m4_t vout = __riscv_vle32_v_f32m4(Output, vl);
                            vacc = __riscv_vfmul_vf_f32m4(vout, beta, vl);
                        } else {
                            vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                        }

                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w00, vr0_0, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w01, vr0_1, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w02, vr0_2, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w10, vr1_0, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w11, vr1_1, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w12, vr1_2, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w20, vr2_0, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w21, vr2_1, vl);
                        vacc = __riscv_vfmacc_vf_f32m4(vacc, w22, vr2_2, vl);

                        __riscv_vse32_v_f32m4(Output, vacc, vl);

                        Output += vl;
                        row0 += vl * stride_w;
                        row1 += vl * stride_w;
                        row2 += vl * stride_w;
                        n -= vl;
                    }
                }

                out_col = pad_right;
            }

            // Handle right padding column (scalar, single element)
            if (out_col == 1) {
                float dotsum = w00 * row0[0] + w01 * row0[1] + w10 * row1[0] + w11 * row1[1] +
                               w20 * row2[0] + w21 * row2[1] + (beta == 0.f ? 0.f : *Output * beta);
                *Output++ = dotsum;
            }

            h += stride_h;
            row0 = (Input + (h - pad_top) * W) - pad_left;
            row1 = row0 + W;
            row2 = (h + 2 >= H + pad_top) ? Zeros : (row1 + W);
        }

    } else { // W == 1

        const size_t H = Parameters->InputShape[0];
        const size_t pad_left = Parameters->Padding[1];
        const size_t pad_top = Parameters->Padding[0];
        const size_t stride_h = Parameters->StrideShape[0];
        size_t out_row = Parameters->OutputShape[0];

        size_t pad_bottom = ((out_row - 1) * stride_h + 3) > (pad_top + H) ?
                                ((out_row - 1) * stride_h + 3) - (pad_top + H) : 0;

        const float w0 = Filter[pad_left ? 1 : 0];
        const float w1 = Filter[pad_left ? 4 : 3];
        const float w2 = Filter[pad_left ? 7 : 6];
        auto init_v = (beta == 0.f ? 0.f : *Output * beta);

        if (pad_top == 1) {
            *Output++ = w1 * Input[0] + w2 * ((H + pad_top <= 2) ? 0.0f : Input[1]) + init_v;
            out_row--;
        }

        // RVV-vectorized W==1 body loop
        {
            const float* row = Input + pad_top * stride_h - pad_top;
            size_t body_rows = (out_row > pad_bottom) ? (out_row - pad_bottom) : 0;

            if (stride_h == 1) {
                size_t n = body_rows;
                while (n > 0) {
                    size_t vl = __riscv_vsetvl_e32m4(n);

                    vfloat32m4_t vr0 = __riscv_vle32_v_f32m4(row + 0, vl);
                    vfloat32m4_t vr1 = __riscv_vle32_v_f32m4(row + 1, vl);
                    vfloat32m4_t vr2 = __riscv_vle32_v_f32m4(row + 2, vl);

                    vfloat32m4_t vacc;
                    if (beta != 0.f) {
                        vfloat32m4_t vout = __riscv_vle32_v_f32m4(Output, vl);
                        vacc = __riscv_vfmul_vf_f32m4(vout, beta, vl);
                    } else {
                        vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                    }

                    vacc = __riscv_vfmacc_vf_f32m4(vacc, w0, vr0, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, w1, vr1, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, w2, vr2, vl);

                    __riscv_vse32_v_f32m4(Output, vacc, vl);

                    Output += vl;
                    row += vl;
                    n -= vl;
                }
            } else {
                ptrdiff_t byte_stride = (ptrdiff_t)stride_h * (ptrdiff_t)sizeof(float);
                size_t n = body_rows;
                while (n > 0) {
                    size_t vl = __riscv_vsetvl_e32m4(n);

                    vfloat32m4_t vr0 = __riscv_vlse32_v_f32m4(row + 0, byte_stride, vl);
                    vfloat32m4_t vr1 = __riscv_vlse32_v_f32m4(row + 1, byte_stride, vl);
                    vfloat32m4_t vr2 = __riscv_vlse32_v_f32m4(row + 2, byte_stride, vl);

                    vfloat32m4_t vacc;
                    if (beta != 0.f) {
                        vfloat32m4_t vout = __riscv_vle32_v_f32m4(Output, vl);
                        vacc = __riscv_vfmul_vf_f32m4(vout, beta, vl);
                    } else {
                        vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                    }

                    vacc = __riscv_vfmacc_vf_f32m4(vacc, w0, vr0, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, w1, vr1, vl);
                    vacc = __riscv_vfmacc_vf_f32m4(vacc, w2, vr2, vl);

                    __riscv_vse32_v_f32m4(Output, vacc, vl);

                    Output += vl;
                    row += vl * stride_h;
                    n -= vl;
                }
            }

            out_row -= body_rows;
        }

        if (out_row > 0) {
            if (pad_bottom == 1) {
                const float* row = Input + H - 2;
                *Output++ = w0 * row[0] + w1 * row[1] + init_v;
            } else {
                *Output++ = w0 * Input[0] + init_v;
            }
        }
    }
}


void
MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
{
    MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1_RVV(Parameters, Input, Filter, Output, Zeros);
}
