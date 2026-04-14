# SconvDepthwise — RVV 3x3 depthwise convolution for SpaceMIT X100

This folder contains the RVV-vectorized single-precision depthwise 3x3
convolution kernel for RISC-V Vector 1.0 on the SpaceMIT X100 (VLEN=256).
It replaces the scalar `SconvDepthwiseKernelScalar.cpp` on RISC-V builds,
using native `f32m4` (32-lane) vector operations for the inner loops.

## Files

| Path | What | Status |
|---|---|---|
| `SconvDepthwiseKernelRVV.cpp` | RVV implementation of `MlasConvDepthwiseFloat_CHW` for 2D 3x3 depthwise with padding ≤ 1 and dilation = 1. | new |
| `../../scalar/SconvDepthwiseKernelScalar.cpp` | Scalar fallback — excluded from the RISC-V build via cmake. Unchanged; still used on other platforms (WASM scalar). | unchanged |
| `../../convolve.cpp` | Depthwise dispatch gates: `MLAS_TARGET_WASM_SCALAR` → `MLAS_TARGET_WASM_SCALAR \|\| __riscv_vector` (3 sites). | modified |
| `../../mlasi.h` | Forward declaration of `MlasConvDepthwiseFloat_CHW` — same gate change. | modified |
| `../../../inc/mlas.h` | `MlasConvAlgorithmDepthwise` enum value — same gate change. | modified |
| `../../../../../../cmake/onnxruntime_mlas.cmake` | RISCV64 source list — adds `riscv64/SconvDepthwise/SconvDepthwiseKernelRVV.cpp` and excludes the scalar file. | modified |

## Why this exists

On non-WASM builds the depthwise dispatch path in `convolve.cpp` was
gated behind `#if defined(MLAS_TARGET_WASM_SCALAR)`, so 3x3 depthwise
convolutions on RISC-V fell through to `MlasConvAlgorithmExpandThenGemm`
(im2col → SGEMM). Adding `|| defined(__riscv_vector)` to those gates
enables the direct depthwise path, and this RVV kernel provides the
vectorized implementation that gets called.

## Algorithm

Same structure as the scalar `MlasConv2dSingleChannel_CHW_Kernel3x3_Pad01_Dilation1`:

1. Left padding column (if `pad_left == 1`): handled scalar — single output.
2. **Main body**: RVV-vectorized — processes `vl` output columns per iteration.
3. Right padding column (if `pad_right == 1`): handled scalar — single output.
4. Vertical stride by advancing `row0/row1/row2` pointers; `Zeros` buffer
   substitutes for out-of-bounds rows at top/bottom.

The W == 1 path vectorizes along the output height dimension using the
same two-variant approach.

### Body-loop vectorization

For each output pixel, the 3x3 kernel needs 9 inputs at offsets
`(0,0), (0,1), (0,2), (1,0), ..., (2,2)` relative to the input position.
The vectorized inner loop produces `vl` output pixels per iteration:

- **stride_w == 1**: 9 contiguous loads (`vle32.v`) at offsets 0/1/2 per row,
  then 9 `vfmacc.vf` with the scalar filter weights. Pointers advance by `vl`.
- **stride_w > 1**: 9 strided loads (`vlse32.v`, byte stride = `stride_w * 4`)
  for each row/offset pair. Pointers advance by `vl * stride_w`.

Beta accumulation: when `beta != 0`, the existing output vector is loaded,
multiplied by beta, and used as the fmacc accumulator seed.

**LMUL choice**: `vfloat32m4` (LMUL=4) — up to 32 f32 lanes per vector
on VLEN=256. Using m4 rather than m8 leaves register pressure room for
the 9 source vectors + accumulator without spills.

## Performance

Measured on SpaceMIT X100 (VLEN=256), single-threaded, comparing this
RVV kernel against the original scalar kernel on the same inputs:

### stride = 1 (contiguous `vle32` loads)

| Input | Output W | Body cols/row | Scalar (µs) | RVV (µs) | Speedup |
|---|---|---|---|---|---|
| 224x224 pad0 | 222 | 222 | 1089.4 | 225.0 | **4.84x** |
| 112x112 pad1 | 112 | 110 | 3141.6 | 713.9 | **4.40x** |
|  56x56  pad1 |  56 |  54 | 1496.7 | 359.0 | **4.17x** |
|  28x28  pad1 |  28 |  26 |  746.7 | 185.3 | **4.03x** |
|  14x14  pad1 |  14 |  12 |  374.4 | 198.0 | **1.89x** |
|   7x7   pad1 |   7 |   5 |  188.2 | 195.9 | **0.96x** |

### stride = 2 (strided `vlse32` loads)

| Input | Output W | Body cols/row | Scalar (µs) | RVV (µs) | Speedup |
|---|---|---|---|---|---|
|  56x56  pad0 |  27 |  27 |  358.8 | 172.7 | **2.08x** |
| 112x112 pad0 |  55 |  55 |  731.1 | 360.2 | **2.03x** |
|  56x56  pad1 |  28 |  26 |  382.0 | 188.1 | **2.03x** |
| 448x448 pad1 | 224 | 222 | 1132.5 | 587.1 | **1.93x** |
| 112x112 pad1 |  56 |  54 |  768.4 | 411.7 | **1.87x** |
| 224x224 pad0 | 111 | 111 | 1496.0 | 904.3 | **1.65x** |
| 224x224 pad1 | 112 | 110 | 1532.1 |1124.0 | **1.36x** |
|  28x28  pad1 |  14 |  12 |   98.0 |  88.6 | **1.11x** |
|  14x14  pad1 |   7 |   5 |   24.1 |  42.9 | **0.56x** |

### Observations

- **stride=1, large spatial (≥ 26 body cols)** — consistent **4x–4.8x**
  speedup. Contiguous `vle32` hits the memory system in cache-line bursts;
  9 fmaccs per pixel fully amortized by vector parallelism.
- **stride=1, small spatial (< 12 body cols)** — speedup drops toward
  **1x**. With only 5 body cols the vector unit runs at ~16 % utilization
  (5/32 lanes), and the fixed per-instruction cost of 20 vector ops
  approaches the cost of the equivalent scalar loop.
- **stride=2** — `vlse32` strided loads cost roughly 2x `vle32` on the
  X100 because consecutive lanes touch separate cache lines. The ceiling
  therefore drops from ~4.8x to ~**2x**.
- **stride=2 with small spatial (≤ 12 body cols)** — strided-load overhead
  plus under-utilized vectors produces a **regression** (0.56x–1.0x).
  These cases are rare in real CNNs (late MobileNet stages), and the
  alternative `ExpandThenGemm` path is not obviously faster either.

## Validation

`onnxruntime_mlas_test --gtest_filter="*Conv*"`:

- `Conv2d_SingleThread` — 88 tests, covers all (kernel, pad, stride, dilation)
  combinations including the `G16/Cpg1/Fpg1` depthwise cases that now hit
  this kernel.
- `Conv2d_Threaded` — 88 tests, same coverage across multiple threads.

**176 / 176 tests pass.**

## Edge cases

The scalar helper handles several tricky edge cases that the RVV version
must preserve:

- **H == 1 with `pad_top == 1`**: the first output row uses `row1 = Input`,
  `row2 = Zeros` (no H+1 row exists). Handled by the initial `pad_top == 1`
  scalar fixup before the vectorized body.
- **W == 1**: the output reduces to a single column; vectorization runs
  along H. Different LMUL path (still m4) with the same
  contiguous / strided load split.
- **`out_row <= pad_bottom`**: the body loop must not execute. Handled by
  `out_row -= body_rows` (where `body_rows` is clamped to zero), preserving
  the post-loop state the scalar path relied on. An early version used
  `out_row = pad_bottom` unconditionally, which crashed the `H=1,W=1`
  test case — corrected.

## What was NOT vectorized

- **Left / right padding columns** (single element each) — scalar.
  Output width < 3 is already a degenerate case; the body-loop structure
  is where the win is.
- **Dilation ≠ 1** — the scalar helper only supports dilation 1, and the
  dispatch in `convolve.cpp` only routes to `MlasConvAlgorithmDepthwise`
  when `DilationShape[i] == 1`. Larger-dilation depthwise still uses
  `ExpandThenGemm` + RVV SGEMM, which is correct (and fast enough on
  small dilations to not be worth a dedicated kernel yet).
