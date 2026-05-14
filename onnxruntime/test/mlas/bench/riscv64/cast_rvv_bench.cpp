/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast_rvv_bench.cpp

Abstract:

    Correctness and performance comparison of RVV-accelerated FP16<->FP32
    cast kernels against ORT's scalar fallback.

--*/

#include "mlas.h"
#include "mlas_float16.h"
#include "mlasi.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

struct Options {
    size_t count = 1024 * 1024;
    size_t iters = 50;
    size_t warmup = 5;
};

Options ParseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        const auto split = arg.find('=');
        if (split == std::string_view::npos) continue;
        const auto key = arg.substr(0, split);
        const auto value = arg.substr(split + 1);
        if (key == "--count") options.count = std::strtoull(value.data(), nullptr, 10);
        else if (key == "--iters") options.iters = std::strtoull(value.data(), nullptr, 10);
        else if (key == "--warmup") options.warmup = std::strtoull(value.data(), nullptr, 10);
    }
    return options;
}

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

void ScalarF16ToF32(const unsigned short* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = MLAS_Half2Float(src[i]);
    }
}

void ScalarF32ToF16(const float* src, unsigned short* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = MLAS_Float2Half(src[i]);
    }
}

void CheckF16ToF32(const char* label, const float* ref, const float* got, size_t count) {
    double max_abs = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        double err = std::abs(ref[i] - got[i]);
        if (err > max_abs) max_abs = err;
        if (err > 0.0) mismatches++;
    }
    std::cout << "  " << label << ": max_abs=" << max_abs
              << " mismatches=" << mismatches << "/" << count
              << (mismatches == 0 ? " PASS" : " FAIL") << "\n";
}

void CheckF32ToF16(const char* label, const unsigned short* ref, const unsigned short* got, size_t count) {
    size_t mismatches = 0;
    size_t max_diff = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t diff = (ref[i] > got[i]) ? (ref[i] - got[i]) : (got[i] - ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0) mismatches++;
    }
    std::cout << "  " << label << ": max_lsb_diff=" << max_diff
              << " mismatches=" << mismatches << "/" << count
              << (mismatches == 0 ? " PASS" : " FAIL") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const Options opts = ParseArgs(argc, argv);
    const size_t N = opts.count;

    std::cout << "=== FP16<->FP32 Cast: RVV vs Scalar ===\n"
              << "  count=" << N << " iters=" << opts.iters << " warmup=" << opts.warmup << "\n\n";

    std::vector<float> fp32_src(N);
    std::vector<unsigned short> fp16_src(N);
    for (size_t i = 0; i < N; ++i) {
        fp32_src[i] = MakeValue(i);
        fp16_src[i] = MLAS_Float2Half(fp32_src[i]);
    }

    std::vector<float> f16_to_f32_scalar(N);
    std::vector<float> f16_to_f32_rvv(N);
    std::vector<unsigned short> f32_to_f16_scalar(N);
    std::vector<unsigned short> f32_to_f16_rvv(N);

    auto* rvv_f16_to_f32 = GetMlasPlatform().CastF16ToF32Kernel;
    auto* rvv_f32_to_f16 = GetMlasPlatform().CastF32ToF16Kernel;

    if (!rvv_f16_to_f32 || !rvv_f32_to_f16) {
        std::cerr << "RVV cast kernels not registered.\n";
        return 1;
    }

    // --- Correctness ---
    ScalarF16ToF32(fp16_src.data(), f16_to_f32_scalar.data(), N);
    rvv_f16_to_f32(fp16_src.data(), f16_to_f32_rvv.data(), N);

    ScalarF32ToF16(fp32_src.data(), f32_to_f16_scalar.data(), N);
    rvv_f32_to_f16(fp32_src.data(), f32_to_f16_rvv.data(), N);

    std::cout << "Correctness:\n";
    CheckF16ToF32("F16->F32", f16_to_f32_scalar.data(), f16_to_f32_rvv.data(), N);
    CheckF32ToF16("F32->F16", f32_to_f16_scalar.data(), f32_to_f16_rvv.data(), N);

    // --- Performance ---
    for (size_t i = 0; i < opts.warmup; ++i) {
        ScalarF16ToF32(fp16_src.data(), f16_to_f32_scalar.data(), N);
        rvv_f16_to_f32(fp16_src.data(), f16_to_f32_rvv.data(), N);
        ScalarF32ToF16(fp32_src.data(), f32_to_f16_scalar.data(), N);
        rvv_f32_to_f16(fp32_src.data(), f32_to_f16_rvv.data(), N);
    }

    double scalar_h2f = TimeLoop(opts.iters, [&]() {
        ScalarF16ToF32(fp16_src.data(), f16_to_f32_scalar.data(), N);
    }) / opts.iters;

    double rvv_h2f = TimeLoop(opts.iters, [&]() {
        rvv_f16_to_f32(fp16_src.data(), f16_to_f32_rvv.data(), N);
    }) / opts.iters;

    double scalar_f2h = TimeLoop(opts.iters, [&]() {
        ScalarF32ToF16(fp32_src.data(), f32_to_f16_scalar.data(), N);
    }) / opts.iters;

    double rvv_f2h = TimeLoop(opts.iters, [&]() {
        rvv_f32_to_f16(fp32_src.data(), f32_to_f16_rvv.data(), N);
    }) / opts.iters;

    double bw_h2f_scalar = (N * (2 + 4)) / (scalar_h2f * 1e6);
    double bw_h2f_rvv = (N * (2 + 4)) / (rvv_h2f * 1e6);
    double bw_f2h_scalar = (N * (4 + 2)) / (scalar_f2h * 1e6);
    double bw_f2h_rvv = (N * (4 + 2)) / (rvv_f2h * 1e6);

    std::cout << std::fixed << std::setprecision(3)
              << "\nF16->F32 (" << N << " elements):\n"
              << "  Scalar:  " << scalar_h2f << " ms  (" << bw_h2f_scalar << " GB/s)\n"
              << "  RVV:     " << rvv_h2f << " ms  (" << bw_h2f_rvv << " GB/s)\n"
              << "  Speedup: " << scalar_h2f / rvv_h2f << "x\n"
              << "\nF32->F16 (" << N << " elements):\n"
              << "  Scalar:  " << scalar_f2h << " ms  (" << bw_f2h_scalar << " GB/s)\n"
              << "  RVV:     " << rvv_f2h << " ms  (" << bw_f2h_rvv << " GB/s)\n"
              << "  Speedup: " << scalar_f2h / rvv_f2h << "x\n";

    return 0;
}
