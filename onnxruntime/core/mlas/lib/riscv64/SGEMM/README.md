# SGEMM and SGEMV Optimization Notes (RISC-V Vector, VLEN=256)

This document describes the optimizations applied to
`riscv64/SgemmKernelRVV.cpp` and `riscv64/SgemvKernelRVV.cpp` on the
SpaceMIT X100 (VLEN=256, vlenb=32). It covers the design choices, the
math behind each tile parameter, the measured impact, and what is still
left on the table for future work.

For the broader background on **why** these RVV kernels exist (a B-pack
layout bug in the upstream scalar fallback that silently corrupted f32
matmul on RISC-V), see `../README.md`. This file is about the *design*
and *performance* of the replacement kernels.

## Hardware target

| Property | Value |
|---|---|
| Chip | SpaceMIT X100 |
| ISA | rv64gcv + zfh + zvfh + zba + zicbop + zihintpause |
| VLEN | **256 bits** (vlenb = 32 bytes) |
| Vector ALU width | 256 bits (one physical register per cycle) |
| Physical vector registers | 32 |
| Cores used in benchmark | 1 (single-threaded) |

The same code paths apply to any RVV 1.0 implementation with VLEN=256,
but the M-tile and prefetch-distance constants are tuned for this chip.

## SGEMM micro-kernel design

### LMUL choice: e32m2

The packer in `sgemm.cpp::MlasSgemmCopyPackB` produces panels with a
**16-float K-stride** (16 floats per K-row, zero-padded for partial-N
tails). To consume one K-row in a single vector instruction, the kernel
needs `VLMAX = 16` for `e32`. Solving for LMUL:

```
VLMAX = LMUL * VLEN / SEW
16    = LMUL * 256  / 32
LMUL  = 2  →  e32m2
```

Comparison of LMUL choices on VLEN=256:

| LMUL | VLMAX (e32) | Ops per K-row | Lane util / FMA | Phys regs / acc |
|------|------------:|--------------:|----------------:|----------------:|
| `m1`  |  8 | **2** vle + 2×M_TILE vfmacc | 100% | 1 |
| **`m2`** | **16** | **1** vle + M_TILE vfmacc  | **100%** | **2** |
| `m4`  | 32 | 1 vle + M_TILE vfmacc | 50% (16 of 32 lanes) | 4 |
| `m8`  | 64 | 1 vle + M_TILE vfmacc | 25% (16 of 64 lanes) | 8 |

`m2` is the unique LMUL that gives 100% lane utilization with the
minimum instruction count. `m1` matches the lane utilization but
doubles the instruction count, paying twice the front-end cost. `m4`
and `m8` issue fewer instructions but waste 50%/75% of the ALU's lanes
per cycle (the chip executes the upper lanes anyway, then discards
them).

Note that "m2 = 512 bits" refers to the *register-group* capacity
(2 physical regs × 256 bits each), **not** the vector ALU width. The
ALU is still 256 bits; an `m2` `vfmacc.vf` with `vl=16` takes ~2 cycles
on the 256-bit datapath, processing 8 lanes per cycle.

### M-tile sizes: 1, 2, 4, 8, 12

The kernel processes `M_TILE` rows of A per call, each row producing
its own m2 accumulator that walks the panel's N dimension in 16-element
chunks. Larger M-tile means more **B-reuse**: a single `vle32` of 16
floats from B feeds `M_TILE` `vfmacc.vf` instructions instead of just
one. This is the dominant source of arithmetic intensity.

Available M-tile values:

| M-tile | Phys regs (acc) | Phys regs (B + scratch) | Total used | Path |
|-------:|----------------:|------------------------:|-----------:|------|
| 1   | 2  | 2 | 4  | `sgemm_inner_small` |
| 2   | 4  | 2 | 6  | `sgemm_inner_small` |
| 4   | 8  | 2 | 10 | `sgemm_inner_small` |
| **8** | **16** | **4** (2× B for K-unroll) | **20** | `sgemm_inner_big` |
| **12** | **24** | **4** | **28** | `sgemm_inner_big` |

M=12 is the largest tile that fits under the 32-register budget while
leaving room for the unrolled K-loop's two B vectors. Larger tiles
(M=14, M=16) start spilling and lose more than they gain.

The dispatch picks the **largest** tile that fits `CountM`:

```cpp
if (CountM >= 12) → sgemm_inner_big<ZeroMode, 12>
else if (CountM >=  8) → sgemm_inner_big<ZeroMode,  8>
else if (CountM >=  4) → sgemm_inner_small<ZeroMode, 4>
else if (CountM >=  2) → sgemm_inner_small<ZeroMode, 2>
else                   → sgemm_inner_small<ZeroMode, 1>
```

The driver in `sgemm.cpp::MlasSgemmKernelLoop` calls the kernel
repeatedly, advancing `A`/`C` by the returned RowsHandled each time, so
e.g. `CountM=10` resolves to one M=8 call followed by one M=2 call.

### K-loop manual unroll by 2 (the "big" path)

In the M=8 and M=12 hot paths the K loop is **manually unrolled by 2**:

```cpp
for (; k + 1 < CountK; k += 2) {
    vfloat32m2_t b0 = vle32(B,      vl);    // load K-row k
    vfloat32m2_t b1 = vle32(B + 16, vl);    // load K-row k+1

    // ... fmaccs against b0 with M_TILE A scalars ...
    // ... fmaccs against b1 with M_TILE A scalars ...

    B += 32; a += 2;
}
```

Both `vle32` instructions are issued before any `vfmacc.vf`, so the
second load can occupy the load-issue port while the first batch of
fmaccs is still draining its latency. GCC at `-O2` does **not** unroll
this loop on its own when the body is built from RVV intrinsics — the
unroll has to be written by hand.

Tail handling for odd CountK is in a small post-loop block.

### SGEMM result summary

Single-thread, `gcc-15 -O2`, X100, VLEN=256:

| Shape | Scalar GFLOPS | RVV v1 (m2 + M=8) | **RVV v2 (m2 + M=12 + K-unroll)** | v2 vs scalar | v2 vs v1 |
|---|---:|---:|---:|---:|---:|
| 16×16×16        | 1.81 | 14.19 | **14.34** | 7.93×  | +1.0% |
| 32×32×32        | 1.95 | 23.41 | **24.53** | 12.55× | +4.8% |
| 64×64×64        | 2.01 | 27.94 | **29.40** | 14.62× | +5.2% |
| 128×128×128     | 2.01 | 29.10 | **30.74** | 15.27× | +5.6% |
| 256×256×256     | 2.01 | 29.35 | **30.99** | 15.40× | +5.6% |
| 128×3072×768 (BERT FFN1) | 1.97 | 23.91 | **25.04** | 12.69× | +4.7% |
| 128×768×3072 (BERT FFN2) | 1.95 | 22.18 | **22.69** | 11.62× | +2.3% |
| 512×512×512     | 2.00 | 28.84 | **28.77** | 14.39× | -0.2% |

**Peak: 30.99 GFLOPS on 256×256×256 = 60.8% of theoretical 51 GFLOPS
(1.6 GHz × 16 lanes × 2 FLOPs/FMA × 1 issue/cycle).**

### Where the SGEMM speedup comes from (per shape)

| Shape class | Bottleneck | What helped most |
|---|---|---|
| Tiny (16-32) | Driver/call overhead | Nothing the kernel can do |
| Square cache-resident (64-256) | FMA throughput | M=8 tile + K-unroll fully overlapping memory and arithmetic |
| BERT FFN1 (M=128 N=3072 K=768) | Mixed compute / panel reuse | M=12 tile (more B-reuse per panel pass) |
| BERT FFN2 (M=128 N=768 K=3072) | C bandwidth (multi-K-slab accumulate) | Almost nothing - need C-residency in driver |
| Large (512) | L2-to-L1 panel traffic | Saturated; unroll cannot help |

## SGEMV micro-kernel design

### Goal: amortize the strided K access pattern

SGEMV (M=1) walks B with **stride `ldb` between consecutive K-rows**.
For ldb=4096, that is a 16 KB jump per K-step. The X100's hardware
prefetcher detects strides up to a few cache lines but cannot keep up
with 16 KB jumps; the access pattern is fundamentally TLB-thrashing
and the kernel ends up DRAM-bound.

The first-cut SGEMV had a single `e32m8` accumulator (vlmax=64
elements) per outer N iteration. That meant each K-step touched only
**256 bytes (4 cache lines) of B contiguously** before jumping to the
next K-row. With only 4 lines hit per page-cross, the fetcher had no
chance to amortize the cost.

### Solution: 4× m4 accumulators (128-element N strip) + prefetch

Bulk path (handles all N up to a final tail < 128):

```cpp
constexpr size_t kStripWide = 128;
const size_t vl = __riscv_vsetvl_e32m4(32);  // == 32 on VLEN=256

while (n + 128 <= CountN) {
    vfloat32m4_t acc0, acc1, acc2, acc3;     // 4 × 32 = 128 N elements
    // load C[n .. n+128] into accs (or zero in ZeroMode)

    for (size_t k = 0; k < CountK; ++k) {
        __builtin_prefetch(b + 8 * ldb);     // 8 K-rows ahead
        const float ak = A[k];

        vfloat32m4_t b0 = vle32(b + 0  * 32, vl);
        vfloat32m4_t b1 = vle32(b + 1  * 32, vl);
        vfloat32m4_t b2 = vle32(b + 2  * 32, vl);
        vfloat32m4_t b3 = vle32(b + 3  * 32, vl);

        acc0 = vfmacc.vf(acc0, ak, b0);
        acc1 = vfmacc.vf(acc1, ak, b1);
        acc2 = vfmacc.vf(acc2, ak, b2);
        acc3 = vfmacc.vf(acc3, ak, b3);

        b += ldb;
    }

    // store all 4 accs back to C[n .. n+128]
    n += 128;
}
```

Three changes vs the v1 SGEMV:

1. **Wider strip per outer iteration (128 instead of 64):** each
   K-step now reads 4 × 32 floats = 512 bytes = 8 cache lines from B
   contiguously, twice the v1 footprint. The next-K-row jump stays the
   same (16 KB) but it amortizes over twice as much useful arithmetic.

2. **Software prefetch hint at +8 K-rows:** issues a non-blocking load
   for the same N strip several iterations ahead so the line lands in
   L1 before the strided K loop arrives at it. Hides DRAM latency on
   each page cross.

3. **C held in registers across the entire K loop:** the 4 m4
   accumulators stay live for `CountK` iterations, so C is loaded once
   at the top and stored once at the bottom of each outer N iteration
   (instead of every K-step in a naive loop nest). For
   `K=4096, N=4096`, this saves `K * N / 128 = 131072` C reloads.

Tail path (any remainder N < 128) uses a single `e32m8` accumulator
with the same prefetch hint, structurally identical to the v1 SGEMV.

### Why m4, not m8 or m2

| LMUL | Elements per acc | Accs for N=128 strip | Phys regs (accs) | Notes |
|------|---:|---:|---:|---|
| m2 | 16 | 8 | 16 | More instructions, smaller B reads |
| **m4** | **32** | **4** | **16** | **Sweet spot: 4 independent loads per K-step, comfortable register budget** |
| m8 | 64 | 2 | 16 | Only 2 independent loads → less ILP for the load unit |

m4 with 4 accs gives **4 independent `vle32` instructions** per
K-step, which the load unit can pipeline. m8 with 2 accs only gives
2, halving the load-issue parallelism. m2 with 8 accs has more
parallelism still but produces twice the instructions, paying more
front-end cost for the same arithmetic.

### SGEMV result summary

| Shape | Scalar | RVV v1 (m8, vl=64) | **RVV v2 (4×m4, prefetch)** | v2 vs scalar | v2 vs v1 |
|---|---:|---:|---:|---:|---:|
| 1 × 4096 × 4096 | 0.487 GFLOPS | 0.721 | **1.014** | **2.08×** | **+40.6%** |

Effective B-read bandwidth: ~4 GB/s out of an estimated 10–25 GB/s
DRAM peak. Still memory-bound but much closer to the ceiling than v1.

## What is still on the table

In rough order of effort vs payoff. None of these are done; they are
documented here so a future contributor knows where the next wins are.

### Cheap

1. **Multi-threading the workload.** All numbers above are
   single-core. The X100 has 4 application cores. Compute-bound SGEMM
   shapes (square 64–256) should scale near-linearly via
   `MlasGemmBatch` with a non-null threadpool. Expected: ~120 GFLOPS
   peak aggregate with no kernel changes.

2. **Build with `-O3` for `riscv64/*.cpp`.** RVV intrinsic-heavy code
   often gets meaningfully better instruction scheduling at `-O3`.
   Set per-source `COMPILE_FLAGS` in `cmake/onnxruntime_mlas.cmake`.
   Expected: 5–15% on the compute-bound shapes.

### Medium

3. **Huge pages for matrix allocations.** Use
   `madvise(buf, size, MADV_HUGEPAGE)` (or the THP `hugepages=always`
   sysctl) at the *caller* level so SGEMV's 16 KB-stride K access
   pattern fits in fewer TLB entries. Expected: SGEMV → 3–5 GFLOPS
   without further kernel changes.

4. **K-outer SGEMV variant.** Re-loop SGEMV with the K loop outermost
   and N inner so B is read **sequentially** (not strided). Requires
   holding all of C in registers across the K loop, which means
   tiling N to fit. Expected: another 2–3× SGEMV speedup on top of
   point 3.

### Larger refactors

5. **C-residency in the SGEMM driver.** Currently each K-slab in
   `MlasSgemmKernelLoop` calls the inner kernel with `ZeroMode=false`
   after the first slab, which means every K-slab does
   `vle C; fmadd; vse C` for the full M-tile worth of C. For K=3072
   that doubles memory traffic on the FFN2 case (down from a possible
   ~30 GFLOPS to the observed ~22). Holding C in registers across all
   K-slabs requires reordering the driver loops (M outer, K inner)
   and is a non-trivial refactor that touches `sgemm.cpp` shared with
   every other architecture.

6. **Wider K-stride in the packer.** If `MlasSgemmCopyPackB` produced
   32-wide or 64-wide N panels instead of 16-wide, we could use
   `e32m4` or `e32m8` LMUL on VLEN=256 with the same lane utilization
   but fewer instructions per K-step. This is the "right" long-term
   answer but it changes the on-disk panel layout and breaks every
   other architecture's inner kernel.

## How to re-measure

The benchmark used for the numbers in this file is at
`/tmp/sgemm_bench.cpp` on the build host. To re-run after a change:

```bash
cd /home/openkylin/github/onnxruntime_risc_v
g++-15 -O2 -std=c++17 -I onnxruntime/core/mlas/inc \
    /tmp/sgemm_bench.cpp \
    build/Linux/Release/libonnxruntime_mlas.a \
    build/Linux/Release/libonnxruntime_common.a \
    build/Linux/Release/_deps/google_nsync-build/libnsync_cpp.a \
    -lpthread -ldl \
    -o /tmp/sgemm_bench
/tmp/sgemm_bench rvv
```

To re-run ORT's own MLAS SGEMM correctness tests (the canonical
validation for any kernel change):

```bash
cd build/Linux/Release
./onnxruntime_mlas_test --gtest_filter='SGemm*:Conv2d*'
```

Both `Conv2d_*` test groups exercise SGEMM transitively via MLAS's
internal im2col → SGEMM, so they are the most useful "did I break
something" check after any inner-kernel edit.
