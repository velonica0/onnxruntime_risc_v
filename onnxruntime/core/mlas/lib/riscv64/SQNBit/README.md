# SQNBitGemm — RVV implementation for SpaceMIT X100

This folder contains the RVV-vectorized implementation of MLAS's
`SQ4BitGemm` (4-bit quantized weight × float32 activation matrix multiply)
for RISC-V Vector 1.0 on the SpaceMIT X100 development board (VLEN=256).

## What this is for

`MatMulNBits` is the ONNX op that backs 4-bit-weight quantized LLMs
(Llama / Phi / Mistral / etc.). MLAS exposes it via the
`MLAS_SQNBIT_GEMM_DISPATCH` struct, which has function-pointer slots for
sizing, packing, the M=1 dot-product hot path, and the M>1 dequant +
SGEMM path. Before this work, the RISC-V build set
`SQNBitGemmDispatch = nullptr`, so any model touching `MatMulNBits`
either fell back to the dequantize-on-host path or failed outright.

This implementation provides the **CompFp32 path** (float A × 4-bit B,
fp32 accumulator), which is what current 4-bit LLM checkpoints exercise.
The CompInt8 path is intentionally left null.

## Files

| Path | What | Status |
|---|---|---|
| `SQNBitGemmKernelRVV_fp32.cpp` | The kernel: 8 dispatch functions providing the full CompFp32 path. | new |
| `sqnbit_bench.cpp`             | Standalone benchmark: scalar (no autovec) vs RVV, single- and multi-threaded (1/2/4/8 threads via OpenMP). | new |
| `../../../../../cmake/onnxruntime_mlas.cmake` | RISCV64 source list — adds `riscv64/SQNBit/SQNBitGemmKernelRVV_fp32.cpp`. | modified |
| `../../mlasi.h`                | Adds `extern const MLAS_SQNBIT_GEMM_DISPATCH MlasSQNBitGemmDispatchRiscv;`. | modified |
| `../../platform.cpp`           | RISCV64 platform-init block — wires `this->SQNBitGemmDispatch = &MlasSQNBitGemmDispatchRiscv;`. | modified |

## What the kernel provides

`MlasSQNBitGemmDispatchRiscv` populates the following slots of
`MLAS_SQNBIT_GEMM_DISPATCH`:

- `SQ4BitGemmPackQuantBDataSize`              — workspace size for packed B (trivial)
- `SQ4BitGemmPackQuantBData`                  — repacks 4-bit B into the SubBlk-16 byte interleave (matches the layout the M1 / dequant kernels read)
- `SQ4BitGemmPerGemmWorkspaceSize`            — returns 0 (CompFp32 needs no per-GEMM workspace)
- `SQ4BitGemmPerGemmWorkspaceAlignment`       — returns 1
- `SQ4BitGemmM1Kernel_CompFp32`               — **M=1 hot path** (LLM token decode): vector × 4-bit-quantized matrix
- `Q4BitBlkDequantBForSgemm_CompFp32`         — **M>1 path**: dequantize 4-bit B into the 16-wide K-major float layout consumed by the existing RVV `MlasSgemmKernelZero/Add`

The CompInt8 slots (`SQ4BitGemmKernel_CompInt8`, `QuantizeARow_CompInt8`,
`SQ4BitGemmKernel_BlkSum_CompInt8`,
`QuantizeARowComputeBlkSum_CompInt8`) are intentionally left null.

## Layout the kernel assumes (= what `SQ4BitGemmPackQuantBData` produces)

Mirrors `sqnbitgemm_kernel_neon::SQ4BitGemmPackQuantBData`:

For each SubBlk of 16 4-bit values (= 8 bytes), the packing rearranges
them so that an 8-byte vector load + low/high nibble extraction yields
two 8-element vectors holding consecutive K positions:

```
src bytes (raw): | v0v1 | v2v3 | v4v5 | v6v7 | v8v9 | vAvB | vCvD | vEvF |
                            ↓
dst bytes (pck): | v0v8 | v1v9 | v2vA | v3vB | v4vC | v5vD | v6vE | v7vF |

After loading 8 dst bytes:
   low nibbles  →  v0 v1 v2 v3 v4 v5 v6 v7   (K positions 0..7)
   high nibbles →  v8 v9 vA vB vC vD vE vF   (K positions 8..15)
```

This is identical to the Neon packing, so the same `SQ4BitGemmPackQuantBData`
implementation works on both architectures.

## RVV-specific details

- **VLEN target:** 256 bits (X100). All hot code uses `vsetvl`-aware
  intrinsics so it remains correct on other VLENs, but tile/LMUL choices
  are tuned for 256.
- **LMUL choice:** `e32m1` (8 lanes per vector at VLEN=256). Initial
  prototype used `e32m2` with a `vslideup` to combine lo+hi nibble
  halves, but the slideup was on the critical path and the `m2` LMUL
  was no faster than `m1` on this microarch. Two parallel `m1`
  accumulators (lo + hi) is faster.
- **Dequant chain (one direction, e.g. low nibbles):**
  `vle8.v` → `vand_vx 0x0F` → `vzext_vf2 u8→u16` → `vzext_vf2 u16→u32`
  → `vfcvt.f.xu u32→f32`. Same chain for high nibbles via `vsrl_vx 4`.
- **Inner unrolling:** NCols=4 (process 4 output columns per A SubBlk
  load) for the bulk; NCols=1 fallback for the (CountN % 4) remainder.
- **Full-SubBlk fast path:** the inner loop is split into a branch-free
  fast path for SubBlks of exactly 16 K-elements (= the always-true case
  given `BlkLen ∈ {16, 32, 64, 128, 256}` and `K % BlkLen == 0`), plus a
  separate tail handler for any partial last SubBlk.
- **K-tiling:** the M1 driver walks K in tiles of ~4096 floats (=16 KB
  of A) so the A row stays L1-resident across the inner N sweep. Tiles
  are constrained to an even number of K-blocks so zero-point nibble
  alignment carries cleanly across tile boundaries.
- **Software prefetch:** `__builtin_prefetch` hints in the inner loop
  for both A and B, ~64 bytes ahead. Most useful when A overflows L1
  (e.g. K=11008).

## Optimizations attempted and **rejected**

These were tried in the optimization sweep and reverted because they
regressed on this microarch:

- **`vfwcvt.f.xu` to skip the u32 stage** — folding the u16→u32→f32
  path into a single u16→f32 widening convert. On X100 it was
  measurable but tiny, so left out for clarity.
- **sumA correction trick** (drop the per-SubBlk `vfsub_vf` for offset
  by accumulating raw `a*b*scale` and applying `scale*offset*sumA` once
  per block). Works on x86 because the AVX issue width has spare scalar
  slots; on the X100's narrower issue, the extra serial latency from
  the per-block `corr` chain ate the savings. Reverted to the v3 chain
  (`vfsub` then `vfmul` then `vfmacc`).

## Validation

`onnxruntime_mlas_test --gtest_filter="SQNBitGemm*"` exercises the
public surface across:

- `BlkBitWidth = 4`
- `BlkLen ∈ {16, 32, 64, 128, 256}`
- `M ∈ {1, 11}`,  `N` and `K` mixed including non-power-of-two
  (e.g. N=527, K=2131)
- `isSymmetric ∈ {0, 1}`  (HasZeroPoint = false / true)
- `hasBias ∈ {0, 1}`
- `computeType = Fp32`

**6760 / 6760 tests pass** at every step of the optimization (v1 → v5).

Numerical agreement vs the scalar baseline: `max_diff ≤ 2e-4` across
all benchmark shapes — pure float-rounding noise from a different
order of summation.

## Performance

### Single-thread (best shape: 2048×2048 BlkLen=128)

| Version | GFLOPS | vs Scalar | Notes |
|---|---|---|---|
| Scalar baseline (no autovec) | 1.19 | 1.0× | hand-rolled, `__attribute__((optimize("no-tree-vectorize")))` |
| RVV v1 | 1.99 | 1.7× | NCols=1, m2 with slideup |
| RVV v2 | 3.47 | 2.9× | NCols=4 + drop slideup (lo/hi m1 split) |
| RVV v3 | 3.66 | 3.1× | + full-SubBlk fast path |
| RVV v4b | 3.68 | 3.1× | + software prefetch |
| **RVV v5** | **3.76** | **3.2×** | **+ K-tiling for cache-bound shapes** |

### Single-thread, full Llama-7B-style decode shapes (BlkLen=32)

| Shape | Scalar | v1 | v3 | **v5** |
|---|---|---|---|---|
| `4096 × 4096`            (q/k/v/o proj) | 1.06 | 2.00 | 3.20 | **3.50** |
| `11008 × 4096`            (FFN up)        | 1.06 | 2.00 | 3.13 | **3.47** |
| `4096 × 11008`            (FFN down)      | 1.06 | 2.00 | 2.46 | **3.40** |
| `32000 × 4096`            (lm_head)       | 1.06 | 2.00 | 3.12 | **3.46** |

The FFN-down shape was the cache-bound outlier in v3 (44 KB A row
overflows L1). v4b's prefetch and v5's K-tiling closed that gap.

### Multi-threaded scaling (8 cores)

OpenMP partitioning over the N dimension. Each thread runs the
existing single-thread kernel on its column slice — read-only A is
shared, B and C are disjoint, no synchronization.

| Shape | 1T | 2T | 4T | 8T | 8T efficiency |
|---|---|---|---|---|---|
| `4096 × 4096`             | 3.51 |  7.03 | 13.98 | **25.70** | 91% |
| `11008 × 4096`            | 3.47 |  6.93 | 13.79 | **26.41** | 95% |
| `4096 × 11008`            | 3.40 |  6.79 | 13.53 | **25.38** | 93% |
| `32000 × 4096`            | 3.50 |  7.00 | 13.94 | **26.71** | 95% |
| `2048 × 2048` (BlkLen=128) | 3.76 |  7.50 | 14.93 | **27.63** | 92% |

**Peak: 27.6 GFLOPS at 8 threads** (Mid 2048×2048 BlkLen=128).

**Llama hot shapes hit ~26 GFLOPS at 8 threads.**

End-to-end speedup over single-threaded scalar baseline: **23×**.

## How to rebuild and re-test

From the project root:

```bash
cd build/Linux/Release

# Rebuild MLAS
ninja onnxruntime_mlas

# Relink the test binary (extracts the linker command from ninja and runs it)
ninja -t commands onnxruntime_mlas_test | tail -1 | bash

# Run the SQNBitGemm test suite
./onnxruntime_mlas_test --gtest_filter="SQNBitGemm*"
```

The full SQNBit suite is 6760 tests, ~12 seconds.

## How to rebuild and re-run the benchmark

```bash
cd /home/openkylin/github/onnxruntime_risc_v

g++-15 -march=rv64gcv_zfh_zvfh_zba_zicbop_zihintpause -mabi=lp64d -O3 \
       -DNDEBUG -DBUILD_MLAS_NO_ONNXRUNTIME -std=c++17 -fopenmp \
       -I onnxruntime/core/mlas/inc \
       -I onnxruntime/core/mlas/lib \
       -I onnxruntime \
       -o /tmp/sqnbit_bench \
       onnxruntime/core/mlas/lib/riscv64/SQNBit/sqnbit_bench.cpp \
       build/Linux/Release/libonnxruntime_mlas.a \
       -lpthread

/tmp/sqnbit_bench
```

The bench includes a tiny stub for the `onnxruntime::concurrency::ThreadPool`
symbols that `libonnxruntime_mlas.a` references — these are dead code
at runtime (we never pass a real ThreadPool) and only exist to satisfy
the linker.

## Future work (not implemented)

- **CompInt8 path** — quantize A to int8, do int8×int4 with int32
  accumulation. Needs `QuantizeARow_CompInt8` and the int8 kernel.
  Could give ~2× over the CompFp32 path on quantize-A-friendly shapes,
  and would integrate naturally with the SpaceMIT IME `vmadot`
  extension on the X100.
- **`vfwcvt.f.xu` u16→f32 fusion** in the dequant chain. Saves 2 ops
  per dequant half but the measured win is small.
- **Manual instruction scheduling** for the inner loop (8-way unroll
  with explicit register file management) — diminishing returns vs
  the current ~3.2× single-thread speedup.
