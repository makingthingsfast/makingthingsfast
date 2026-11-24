# Week 0: how it was run

## Hardware

- Apple M3 Pro: 6 performance + 6 efficiency cores, 18 GB unified memory
- macOS 26.5.1
- Plugged in (on battery it was about 10% slower)
- Single-threaded. macOS has no way to pin a thread to a core, so the scheduler picks where it runs.

## Compiler

Apple clang 21.0.0 (clang-2100.1.1.101). For each K, `fma_k_sweep.sh` builds with:

```sh
# scalar
clang -std=c23 -O2 -DK=<K> -DQUIET peak.c -o peak

# NEON
clang -std=c23 -O2 -DK=<K> -DQUIET -DNEON peak.c -o peak
```

- `-std=c23`: needed for the `1'000'000'000L` digit separators
- `-O2`: the optimisation level used for every table
- `-DK`: number of independent accumulators
- `-DQUIET`: one line of output per run, instead of every accumulator
- `-DNEON`: use the NEON worker (2 doubles per FMA) instead of the scalar one

## What the program measures

`peak.c` runs `n = 1,000,000,000` passes of a loop that does `K` independent FMAs per pass. Only that loop is timed, with `clock_gettime(CLOCK_MONOTONIC)`.

- **GFLOP/s** = `n × 2 × lanes × K / time`: 2 FLOPs per FMA, with 1 lane for scalar and 2 for NEON.
- **cycles per pass** = `time × 4 GHz / n`. The 4 GHz is an assumed clock, not a measured one.
- The accumulators are summed and printed at the end, so the compiler can't delete the loop.

## How to run

From this folder:

```sh
# usage: ./fma_k_sweep.sh OPT QUIET K...

# scalar table
./fma_k_sweep.sh 2 1 1 2 4 8 10 12 14 16 18 20 24 28 32

# NEON table
EXTRA=-DNEON ./fma_k_sweep.sh 2 1 1 2 4 8 10 12 14 16 18 20 24 28 32

# -Ofast: OPT goes straight into -O$OPT
./fma_k_sweep.sh fast 1 1 2 4 8 10 12 14 16 18 20 24 28 32
```

Each run builds `peak` next to the source and prints one line per K. Set QUIET to 0 to also print every accumulator.

## Files

- `README.md`: the post
- `peak.c`: the benchmark
- `fma_k_sweep.sh`: builds and runs it once per K
