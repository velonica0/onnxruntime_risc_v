/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rope_ln_gather_bench.cpp

Abstract:

    Benchmarks for RVV-optimized RotaryEmbedding and LayerNorm
    vs scalar implementations. Reports speedup and precision.

--*/

#include "mlas.h"
#include "mlas_float16.h"
#include "rotary_embedding.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace {

float MakeValue(size_t index) {
    uint32_t x = static_cast<uint32_t>(index * 747796405u + 2891336453u);
    x ^= x >> 16;
    x *= 2246822519u;
    x ^= x >> 13;
    return (static_cast<float>(x % 2048u) / 1024.0f) - 1.0f;
}

template <typename Fn>
double TimeLoop(size_t iterations, Fn&& fn) {
    const auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        fn();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void CompareResults(const char* label, const float* ref, const float* got, size_t n) {
    double max_abs = 0.0, max_rel = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < n; i++) {
        double abs_err = std::abs(ref[i] - got[i]);
        double rel_err = (std::abs(ref[i]) > 1e-7) ? abs_err / std::abs(ref[i]) : abs_err;
        if (abs_err > max_abs) max_abs = abs_err;
        if (rel_err > max_rel) max_rel = rel_err;
        if (abs_err > 1e-5) mismatches++;
    }
    std::cout << "  " << label << ": max_abs=" << max_abs
              << " max_rel=" << max_rel
              << " mismatches=" << mismatches << "/" << n
              << (mismatches == 0 ? " PASS" : " FAIL") << "\n";
}

// ===================== Scalar RoPE fallback =====================
void ScalarRoPE(const float* input, const float* sin_data, const float* cos_data,
                size_t dim, bool interleaved, float* output) {
    const size_t half = dim / 2;
    for (size_t i = 0; i < dim; i++) {
        size_t cache_idx;
        bool sign;
        size_t j;
        if (interleaved) {
            cache_idx = (i / 2) % half;
            sign = i & 1;
            j = sign ? i - 1 : i + 1;
        } else {
            cache_idx = i % half;
            sign = (i >= half);
            j = (i + half) % dim;
        }
        float out_i = input[i] * cos_data[cache_idx];
        if (sign)
            out_i += input[j] * sin_data[cache_idx];
        else
            out_i -= input[j] * sin_data[cache_idx];
        output[i] = out_i;
    }
}

// ===================== Scalar LayerNorm =====================
void ScalarLayerNorm(const float* input, const float* scale, size_t n,
                     float epsilon, bool simplified, float* output) {
    float sum = 0.0f, sumsq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += input[i];
        sumsq += input[i] * input[i];
    }
    float mean = sum / static_cast<float>(n);
    float denom;
    if (simplified)
        denom = sqrtf(sumsq / static_cast<float>(n) + epsilon);
    else
        denom = sqrtf(sumsq / static_cast<float>(n) - mean * mean + epsilon);
    float inv_denom = 1.0f / denom;
    for (size_t i = 0; i < n; i++) {
        if (simplified)
            output[i] = input[i] * inv_denom * scale[i];
        else
            output[i] = (input[i] - mean) * inv_denom * scale[i];
    }
}

}  // namespace

int main(int, char**) {
    const size_t iters = 500;
    const size_t warmup = 50;

    std::cout << "=== RVV Operator Benchmarks: RoPE, LayerNorm ===\n\n";

    // ======================== RoPE ========================
    {
        const size_t dim = 128;  // head_dim for Qwen3
        const size_t half = dim / 2;

        std::vector<float> input(dim), sin_data(half), cos_data(half);
        std::vector<float> out_scalar(dim), out_rvv(dim);

        for (size_t i = 0; i < dim; i++) input[i] = MakeValue(i);
        for (size_t i = 0; i < half; i++) {
            sin_data[i] = sinf(static_cast<float>(i) * 0.01f);
            cos_data[i] = cosf(static_cast<float>(i) * 0.01f);
        }

        ScalarRoPE(input.data(), sin_data.data(), cos_data.data(), dim, false, out_scalar.data());
        MlasRotaryEmbedOneRow<float>(input.data(), sin_data.data(), cos_data.data(), dim, false, out_rvv.data());

        std::cout << "--- RotaryEmbedding (dim=" << dim << ", non-interleaved) ---\n";
        CompareResults("RVV vs Scalar", out_scalar.data(), out_rvv.data(), dim);

        for (size_t i = 0; i < warmup; i++) {
            ScalarRoPE(input.data(), sin_data.data(), cos_data.data(), dim, false, out_scalar.data());
            MlasRotaryEmbedOneRow<float>(input.data(), sin_data.data(), cos_data.data(), dim, false, out_rvv.data());
        }

        double scalar_ms = TimeLoop(iters, [&]() {
            ScalarRoPE(input.data(), sin_data.data(), cos_data.data(), dim, false, out_scalar.data());
        }) / iters;
        double rvv_ms = TimeLoop(iters, [&]() {
            MlasRotaryEmbedOneRow<float>(input.data(), sin_data.data(), cos_data.data(), dim, false, out_rvv.data());
        }) / iters;

        std::cout << std::fixed << std::setprecision(4)
                  << "  Scalar: " << scalar_ms * 1000 << " us\n"
                  << "  RVV:    " << rvv_ms * 1000 << " us\n"
                  << "  Speedup: " << scalar_ms / rvv_ms << "x\n\n";
    }

    // ======================== LayerNorm ========================
    {
        const size_t hidden = 1024;  // hidden_size for Qwen3
        const float epsilon = 1e-6f;

        std::vector<float> input(hidden), scale(hidden);
        std::vector<float> out_scalar(hidden), out_rvv(hidden);

        for (size_t i = 0; i < hidden; i++) {
            input[i] = MakeValue(i) * 0.1f;
            scale[i] = 1.0f + MakeValue(i + hidden) * 0.01f;
        }

        ScalarLayerNorm(input.data(), scale.data(), hidden, epsilon, true, out_scalar.data());

#if defined(__riscv) && defined(__riscv_vector)
        // RVV SimplifiedLayerNorm (RMSNorm)
        {
            const float* p_input = input.data();
            float* p_output = out_rvv.data();

            vfloat32m1_t vsumsq = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
            size_t idx = 0;
            while (idx < hidden) {
                size_t vl = __riscv_vsetvl_e32m4(hidden - idx);
                vfloat32m4_t vx = __riscv_vle32_v_f32m4(p_input + idx, vl);
                vfloat32m4_t vx2 = __riscv_vfmul_vv_f32m4(vx, vx, vl);
                vsumsq = __riscv_vfredusum_vs_f32m4_f32m1(vx2, vsumsq, vl);
                idx += vl;
            }
            float ms = __riscv_vfmv_f_s_f32m1_f32(vsumsq);
            float inv_denom = 1.0f / sqrtf(ms / static_cast<float>(hidden) + epsilon);

            idx = 0;
            while (idx < hidden) {
                size_t vl = __riscv_vsetvl_e32m4(hidden - idx);
                vfloat32m4_t vx = __riscv_vle32_v_f32m4(p_input + idx, vl);
                vfloat32m4_t vs = __riscv_vle32_v_f32m4(scale.data() + idx, vl);
                vfloat32m4_t vy = __riscv_vfmul_vf_f32m4(vx, inv_denom, vl);
                vy = __riscv_vfmul_vv_f32m4(vy, vs, vl);
                __riscv_vse32_v_f32m4(p_output + idx, vy, vl);
                idx += vl;
            }
        }
#else
        ScalarLayerNorm(input.data(), scale.data(), hidden, epsilon, true, out_rvv.data());
#endif

        std::cout << "--- SimplifiedLayerNorm / RMSNorm (hidden=" << hidden << ") ---\n";
        CompareResults("RVV vs Scalar", out_scalar.data(), out_rvv.data(), hidden);

        auto run_scalar_ln = [&]() {
            ScalarLayerNorm(input.data(), scale.data(), hidden, epsilon, true, out_scalar.data());
        };

#if defined(__riscv) && defined(__riscv_vector)
        auto run_rvv_ln = [&]() {
            const float* p_input = input.data();
            float* p_output = out_rvv.data();

            vfloat32m1_t vsumsq = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
            size_t idx = 0;
            while (idx < hidden) {
                size_t vl = __riscv_vsetvl_e32m4(hidden - idx);
                vfloat32m4_t vx = __riscv_vle32_v_f32m4(p_input + idx, vl);
                vfloat32m4_t vx2 = __riscv_vfmul_vv_f32m4(vx, vx, vl);
                vsumsq = __riscv_vfredusum_vs_f32m4_f32m1(vx2, vsumsq, vl);
                idx += vl;
            }
            float ms = __riscv_vfmv_f_s_f32m1_f32(vsumsq);
            float inv_denom = 1.0f / sqrtf(ms / static_cast<float>(hidden) + epsilon);

            idx = 0;
            while (idx < hidden) {
                size_t vl = __riscv_vsetvl_e32m4(hidden - idx);
                vfloat32m4_t vx = __riscv_vle32_v_f32m4(p_input + idx, vl);
                vfloat32m4_t vs = __riscv_vle32_v_f32m4(scale.data() + idx, vl);
                vfloat32m4_t vy = __riscv_vfmul_vf_f32m4(vx, inv_denom, vl);
                vy = __riscv_vfmul_vv_f32m4(vy, vs, vl);
                __riscv_vse32_v_f32m4(p_output + idx, vy, vl);
                idx += vl;
            }
        };
#else
        auto run_rvv_ln = run_scalar_ln;
#endif

        for (size_t i = 0; i < warmup; i++) { run_scalar_ln(); run_rvv_ln(); }

        double scalar_ms = TimeLoop(iters, run_scalar_ln) / iters;
        double rvv_ms = TimeLoop(iters, run_rvv_ln) / iters;

        std::cout << std::fixed << std::setprecision(4)
                  << "  Scalar: " << scalar_ms * 1000 << " us\n"
                  << "  RVV:    " << rvv_ms * 1000 << " us\n"
                  << "  Speedup: " << scalar_ms / rvv_ms << "x\n\n";
    }

    return 0;
}
