# RISC-V MLAS Notes

This directory holds the RVV 1.0 inner kernels for MLAS on RISC-V. Most files
here are straightforward ports of the corresponding scalar/NEON kernels and
need no special explanation. This README documents one thing that **does** need
explaining: the B-pack layout contract that f32 SGEMM relies on, and the bug
that resulted from getting it wrong on this fork.

If you ever add a new MLAS micro-kernel for RISC-V, or port one to another
architecture that lands in the same code path, **read this first**.

## TL;DR

`MlasSgemmCopyPackB` (in `sgemm.cpp`) and the inner SGEMM micro-kernel
(in `riscv64/SgemmKernelRVV.cpp` for us) communicate through a stack-allocated
`PanelB[]` buffer that has a specific in-memory layout. There are **two**
possible layouts in the upstream MLAS source, gated by a preprocessor macro:

| Build target                       | Packer layout    | Inner-kernel layout it must consume |
|------------------------------------|------------------|-------------------------------------|
| `MLAS_TARGET_WASM_SCALAR`          | 4-wide K-major   | 4-wide K-major                      |
| Everything else (NEON/AVX/RVV/...) | 16-wide K-major  | 16-wide K-major                     |

Mixing a 16-wide packer with a 4-wide consumer (or vice-versa) silently
produces wrong f32 matmul results. That is exactly the bug this fork shipped
with for f32 SGEMM on RISC-V before the riscv64 SGEMM kernel was added.

## What "16-wide" and "4-wide" actually mean

These are not abstract tile sizes. They are the **K-row stride in `PanelB`**,
i.e. how many floats apart consecutive K-rows of the packed B-panel are stored.

### 16-wide layout (the one RISC-V uses)

For each 16-column N strip, K rows are stored contiguously, 16 floats per row:

```
PanelB[ 0..15]        = b[k=0,   n=0..15]
PanelB[16..31]        = b[k=1,   n=0..15]
PanelB[32..47]        = b[k=2,   n=0..15]
...
PanelB[16*K..16*K+15] = b[k=K-1, n=0..15]
PanelB[16*K+16 .. ]   = b[k=0,   n=16..31]   # next 16-column strip
```

If `CountN` (the actual N width) is less than 16, each K-row is **zero-padded
out to 16 floats** so the K-stride is still 16. This is critical: a kernel
can unconditionally read 16 floats per K-row even when the user asked for
fewer columns.

### 4-wide layout (only WASM-scalar)

```
PanelB[0..3]   = b[k=0, n=0..3]
PanelB[4..7]   = b[k=1, n=0..3]
PanelB[8..11]  = b[k=2, n=0..3]
...
PanelB[4*K..]  = next 4-column strip starts: b[k=0, n=4..7]
```

The K-stride is 4 floats. Each "subpanel" is only 4 columns wide.

### Where this is reflected in the code

In `sgemm.cpp` (the packer side, around line 213):

```cpp
#if !defined(MLAS_TARGET_WASM_SCALAR)
void MlasSgemmCopyPackB(...) {
    while (CountX >= 16) {
        ...
        do {
            // copy 16 contiguous floats from one source K-row into D
            MlasStoreAlignedFloat32x4(&D[0],  ...);
            MlasStoreAlignedFloat32x4(&D[4],  ...);
            MlasStoreAlignedFloat32x4(&D[8],  ...);
            MlasStoreAlignedFloat32x4(&D[12], ...);
            D += 16;             //  <-- THE 16-WIDE PROPERTY: K-stride = 16
            b += ldb;
        } while (--y);
        ...
    }
    // tail handling for CountX < 16 also uses D += 16 with zero padding
}
#else  // WASM_SCALAR
void MlasSgemmCopyPackB(...) {
    // 4-wide variant uses D += 4 per K-row
}
#endif
```

In any inner kernel that consumes the 16-wide layout (NEON, AVX, our new
`riscv64/SgemmKernelRVV.cpp`):

```cpp
for (size_t k = 0; k < CountK; ++k) {
    // read up to 16 floats from one K-row of the panel
    vfloat32m4_t b = __riscv_vle32_v_f32m4(B, vl);   // vl <= 16
    acc0 = __riscv_vfmacc_vf_f32m4(acc0, a[0],   b, vl);
    if (ProcessTwoRows)
        acc1 = __riscv_vfmacc_vf_f32m4(acc1, a[lda], b, vl);
    B += 16;                     //  <-- MUST match the packer
    a += 1;
}
```

The single line `B += 16` (and `D += 16` on the packer side) is the entire
"16-wide" property. The choice of vector width inside one K-step
(`vle32_v_f32m4` reads up to 16 lanes here) is a separate decision about
how much SIMD work to do per K-step; it does not change the K-stride.

In the **scalar** kernel (`scalar/SgemmKernelScalar.cpp`), the matching
lines are `B += 8` (per 2 K-steps) and `B += 4` (per 1 K-step), giving an
effective K-stride of 4. That kernel only matches the 4-wide packer.

## The bug this fork shipped

Before the riscv64 SGEMM kernel was added, the RISC-V build had:

1. `sgemm.cpp` compiled with `MLAS_TARGET_WASM_SCALAR` undefined (because
   we are not wasm), so the **16-wide** `MlasSgemmCopyPackB` was active.
2. No riscv64-specific f32 SGEMM kernel, so cmake fell through to
   `file(GLOB_RECURSE scalar/*.cpp)` and pulled in
   `scalar/SgemmKernelScalar.cpp`, which is **4-wide**.

The `MlasSgemmKernelZero` / `MlasSgemmKernelAdd` symbols defined in the
scalar file were called by `MlasSgemmKernelLoop` with a `PanelB` pointer
that the 16-wide packer had filled. The kernel walked it as if it were
4-wide and multiplied unrelated B elements with the wrong A coefficients.

### Concrete trace of the failure

For `M=1, N=8, K=8`:

```
Source B (2 rows shown for brevity; actual K=8):
  b[k=0] = [10, 11, 12, 13, 14, 15, 16, 17]
  b[k=1] = [20, 21, 22, 23, 24, 25, 26, 27]
  ...

Packer writes (16-wide, zero-padded because CountN=8 < 16):
  PanelB[ 0.. 7] = [10, 11, 12, 13, 14, 15, 16, 17]   # b[k=0]
  PanelB[ 8..15] = [ 0,  0,  0,  0,  0,  0,  0,  0]   # zero pad
  PanelB[16..23] = [20, 21, 22, 23, 24, 25, 26, 27]   # b[k=1]
  PanelB[24..31] = [ 0,  0,  0,  0,  0,  0,  0,  0]   # zero pad
  ...

Scalar kernel reads (assumes 4-wide layout):
  B[0..3] -> intends b[k=0,n=0..3]: gets [10,11,12,13]   CORRECT (lucky)
  B[4..7] -> intends b[k=1,n=0..3]: gets [14,15,16,17]   WRONG
                                                          (this is b[k=0,n=4..7])
```

The kernel then multiplies `B[4..7]` by `a[1]` (the k=1 coefficient),
which is the wrong source data weighted by the wrong coefficient. The math
falls apart from the very first K-step. By the time the outer loop walks
to columns 4..7, the kernel is reading zero padding and bits of unrelated
K-rows in random positions.

### Why nobody noticed

The SpaceMIT contributors who added RISC-V support to this fork were
focused on the IME quantized GEMM path
(`qgemm_kernel_spacemit_ime{1,2}.cpp`). f32 SGEMM was never validated on
RISC-V because the target workload is quantized inference. The build
compiled fine; quantized models ran fine; f32 matmul silently produced
garbage.

### How it was found

A small standalone smoke test that calls `MlasGemm` with random small
matrices and compares against a naive triple-loop reference. Before the
fix, 8 of 9 test sizes failed with relative errors around 100 percent
(not even the right sign at most positions). The only passing case was
`M=N=K=1`, which is degenerate enough to bypass the layout issue.

After replacing the scalar kernel with the new RVV kernel that consumes
the 16-wide layout, all 14 cases of an extended test pass within float
round-off (max relative error ~5e-7), and a more thorough 160-case
comparison test (transpose variants, multi-K-slab, pre-packed B,
SGEMV path, stride aliasing) passes with the K<=128 cases bit-exact
against the naive reference.

## The fix

Two changes resolve the layout mismatch (the other three changes in the
same patch are about the M=1 SGEMV fast path, not this bug).

### 1. New file: `riscv64/SgemmKernelRVV.cpp`

Provides `MlasSgemmKernelZero` and `MlasSgemmKernelAdd` (the same symbol
names the scalar file uses, so it replaces them at link time). The inner
loop uses `vfmacc.vf` with LMUL=4 and walks B with stride 16 per K-step,
matching the packer.

### 2. cmake change in the RISCV64 block of `cmake/onnxruntime_mlas.cmake`

The previous block did:

```cmake
file(GLOB_RECURSE mlas_platform_srcs_generic "${MLAS_SRC_DIR}/scalar/*.cpp")
set(mlas_platform_srcs ${mlas_platform_srcs} ${mlas_platform_srcs_generic})
```

That glob silently pulled in `SgemmKernelScalar.cpp` (the 4-wide kernel)
into the RISC-V build. The fix replaces the glob with an explicit list
that excludes `SgemmKernelScalar.cpp` and `SgemvKernelScalar.cpp` and
adds the new RVV files:

```cmake
set(mlas_platform_srcs
    ${mlas_platform_srcs}
    ${MLAS_SRC_DIR}/riscv64/SgemmKernelRVV.cpp
    ${MLAS_SRC_DIR}/riscv64/SgemvKernelRVV.cpp
    )
set(mlas_platform_srcs
    ${mlas_platform_srcs}
    ${MLAS_SRC_DIR}/scalar/SconvDepthwiseKernelScalar.cpp
    )
```

Without the cmake change, both the scalar and the RVV files would define
the same `MlasSgemmKernelZero` symbol and the link would either fail with
a multiple-definition error or pick one nondeterministically.

## Guidance for future MLAS porting work

If you add a new MLAS architecture or modify the RISC-V kernels, the
following invariants must hold or you will reproduce the bug.

1. **Pick the layout from `sgemm.cpp`, not from your kernel design.**
   The packer is the source of truth. Whatever layout `MlasSgemmCopyPackB`
   produces in your build is what your inner kernel must consume.

2. **For any non-WASM-scalar build, that means 16-wide.** Your inner
   kernel must (a) load up to 16 floats per K-row of B, possibly ignoring
   upper lanes safely (the packer zero-pads), and (b) advance B by 16 per
   K-step, regardless of how many N columns are real this iteration.

3. **Do not pull `scalar/SgemmKernelScalar.cpp` into a non-WASM-scalar
   build.** It only matches the 4-wide packer and will silently corrupt
   results otherwise. The same applies to `scalar/SgemvKernelScalar.cpp`.
   If you cmake-glob `scalar/*.cpp`, list `SconvDepthwiseKernelScalar.cpp`
   explicitly instead.

4. **Run a smoke test against a naive triple-loop reference.** The
   existence of such a test is the only reason this bug was found.
   Cross-checking against `onnxruntime_mlas_test`'s `SGemm_*` test groups
   is the canonical way once that test binary builds on the target.

5. **The K-stride in `PanelB` is independent of your inner kernel's tile
   width.** You can choose a tile of M=4 by N=16, M=8 by N=16, or anything
   else; the only thing the packer cares about is that you advance B by
   16 per K-step. The vector width of one `vfmacc` is a separate, free
   parameter.
