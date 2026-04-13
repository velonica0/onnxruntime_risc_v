# Pooling — RVV 2D vector kernels for SpaceMIT X100

This folder contains the RVV-vectorized 2D float pooling kernels for
RISC-V Vector 1.0 on the SpaceMIT X100 (VLEN=256). These replace the
auto-vectorized `MLAS_FLOAT32X4` (4-lane) path with native `f32m2`
(16-lane) operations for the 2D Max/Avg pooling hot path.

## Files

| Path | What | Status |
|---|---|---|
| `PoolingKernelRVV.cpp` | 3 native f32m2 2D vector kernels: Max, AvgIncludePad, AvgExcludePad. | new |
| `../../pooling.cpp`     | Dispatch table `MlasPoolVectorKernels[][2]` — RVV entries for 2D path via `#if defined(MLAS_TARGET_RISCV64)`. 3D path unchanged. | modified |
| `../../../../../cmake/onnxruntime_mlas.cmake` | RISCV64 source list — adds `riscv64/Pooling/PoolingKernelRVV.cpp`. | modified |

## What the kernels provide

Three functions wired into `pooling.cpp`'s `MlasPoolVectorKernels` dispatch:

- `MlasPool2DMaxVectorKernel_RVV` — 2D MaxPool
- `MlasPool2DAvgIncludePadVectorKernel_RVV` — 2D AvgPool (include padding in divisor)
- `MlasPool2DAvgExcludePadVectorKernel_RVV` — 2D AvgPool (exclude padding from divisor)

3D pooling and global pooling continue to use the existing template kernels.

## Algorithm

Same two-phase structure as the original `MlasPool2DVectorKernel<>`:

1. **H-reduction**: for each output row, fold the kernel-height input rows
   into a 1D `ReductionBuffer` using `vfmax`/`vfadd`, processing 16 width
   positions per pass (vs 4 in the original).

2. **W-reduction**: slide a window of `KernelWidth` across the buffer,
   producing `OutputWidth` outputs. Stride 1 uses contiguous `vle32`
   loads; stride 2 uses strided `vlse32` (byte stride 8). Other strides
   fall back to scalar.

For AvgPool IncludePad: divisor = `KernelHeight * KernelWidth` (constant).
Uses `vfdiv_vf` (not multiply-by-inverse) to bit-match the scalar
reference's `m /= K`.

For AvgPool ExcludePad: per-output-column divisor depending on the count
of valid (non-padding) input positions. Uses scalar per-column loop
(complexity is O(OutputWidth) per row, negligible vs H-reduction cost).

## Struct compatibility

`MLAS_POOL_WORK_BLOCK` is defined inside `pooling.cpp`'s translation unit
(not in a header). The RVV file declares a layout-compatible
`MLAS_POOL_WORK_BLOCK_RVV` struct and casts via `const void*` at the
`extern "C"` boundary. The pooling.cpp dispatch declares the RVV
functions with the native `const MLAS_POOL_WORK_BLOCK*` type;
extern "C" linkage resolves by name across TUs.

## Bug found and fixed during development

AvgPool IncludePad initially used `vfmul_vf(acc, 1.0f/divisor)` — multiply
by the precomputed inverse. The ORT test reference uses `m /= divisor`
(scalar division). IEEE 754 guarantees that `a/b` is correctly rounded,
but `a * (1/b)` is not (the intermediate `1/b` introduces a rounding
step). The two differ by up to 1 ULP, which fails `memcmp`. Fix: use
`vfdiv_vf(acc, divisor)` which is a single correctly-rounded divide,
matching the reference bit-exactly.

## Validation

`onnxruntime_mlas_test --gtest_filter="Pool*"` exercises:

- Pool2d Max / AvgExcludePad / AvgIncludePad (SingleThread + Threaded)
- Pool3d Max / AvgExcludePad / AvgIncludePad (unchanged, existing kernel)

**588 / 588 tests pass.**

## What was NOT ported (and why)

- **3D vector pooling**: rarely used in CNN models; left on the existing
  template path.
- **Global pooling**: already efficient (single reduction per channel).
- **Reorder (NCHW ↔ NCHWc)**: dead code on RISC-V. `MlasNchwcGetBlockSize()`
  returns 1 (NCHWc not supported), so the reorder functions are never
  called by ORT's convolution path. No reorder tests exist on this build.
  Skipped.
- **SconvDepthwise**: dead code on RISC-V. The `MlasConvAlgorithmDepthwise`
  path in `convolve.cpp` is gated behind `MLAS_TARGET_WASM_SCALAR`.
  Depthwise 3x3 convs go through the generic `ExpandThenGemm` path
  which already uses the RVV SGEMM. Skipped.
