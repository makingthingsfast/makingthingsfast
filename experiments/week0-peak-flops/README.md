# Week 0: peak FLOP/s

Nov 24, 2026 · Tymoteusz Niewodniczanski

How fast can my computer do maths?

Well, to answer the above, we first need to pose a more directed question.

Firstly, what computer do I have? Okay, well, I have a Mac M3 Pro. It has 6 performance cores and 6 efficiency cores (though for this we'll only need one performance core). We're also going to be using C code, compiled with clang and the -O2 flag (more on that later). Everything was run plugged in, since on battery it was about 10% slower.

That's not all. What about the maths itself? Are we adding, subtracting, dividing, or something else? For this experiment we'll be doing a Fused-Multiply-Add (FMA) operation on 64 bit (8B) large floating point numbers. As the name might suggest, it just computes a\*b + c in just one instruction, making it faster - the fused part means it only rounds once, at the end, rather than after each sub-operation, making it more accurate. It also counts as 2 FLOPs (floating point operations) with one multiply and one add.

Well, okay, seems easy enough. But first, we should probably do some napkin math and try to figure out a "target".  We do 2 FLOPs per FMA, at about 4GHz, on 4 units (physical hardware which does the FMA inside of the core, and can start a new one every cycle), using 2 lanes (each register is 128-bit, so it holds two doubles). So, 2\*2\*4\*4 comes out to an estimate of 64 GFLOP/s per core.

So, let's run it. With a simple one-accumulator loop across a billion passes, written in C, we can benchmark just how many operations a second we can do. With some simple logging, we see... wait! Does that say 1.95 GFLOP/s? That can't be right!

```c
double acc = 0;
double b = 0.999999;
double c = 6.7;

double ts = now();
for (long i = 0; i < n; i++)
{
    acc = fma(b, c, acc);   // acc = b*c + acc
}
double elapsed = now() - ts;
```

```plain
time:  1.025 s
speed: 1.95 GFLOP/s
```

Well, it is. A single FMA takes roughly 4 cycles to produce an answer. With one accumulator, each FMA requires the previous answer, so it has to wait, causing a dependency chain. This results in 2FLOPs / 4 cycles × 4GHz or 2GFLOP/s. In the meantime, 3 of 4 FMA units sit idle, and so does one lane of the unit that is working. That's why we are 4(latency)\*4(units)\*2(lanes) or 32x slower than our initial estimate.

Okay, so how do we get it faster? Well, we can't really get the latency down! This is as quick as we can get it. There is one thing we can increase, though.

Throughput. Throughput effectively measures how much of something we can move in a period of time, rather than how quickly. We can start separate units with separate accumulators, allowing us to run more FMAs at the same time, so by sweeping across a range of K accumulators, we obtain the following results:

```c
double acc[K];
for (int j = 0; j < K; j++) { acc[j] = j; }   // different start values

for (long i = 0; i < n; i++)
{
    for (int j = 0; j < K; j++)
    {
        acc[j] = fma(b, c, acc[j]);   // K independent chains
    }
}
```

| K | time (s) | GFLOP/s | cycles per pass |
| --- | --- | --- | --- |
| 1 | 1.025 | 1.95 | 4.099 |
| 2 | 1.032 | 3.88 | 4.129 |
| 4 | 1.025 | 7.81 | 4.099 |
| 8 | 1.026 | 15.59 | 4.106 |
| 10 | 1.054 | 18.97 | 4.216 |
| 12 | 1.068 | 22.47 | 4.272 |
| 14 | 1.060 | 26.42 | 4.239 |
| 16 | 1.049 | 30.50 | 4.196 |
| 18 | 1.156 | 31.14 | 4.624 |
| 20 | 1.279 | 31.27 | 5.116 |
| 24 | 1.516 | 31.67 | 6.063 |
| 28 | 1.773 | 31.59 | 7.091 |
| 32 | 2.628 | 24.35 | 10.512 |

Once again, we hit a wall, caused by all 4 FMA units being full. Each unit starts a new FMA every cycle, and each FMA takes 4 cycles to finish, so there can be at most 4 units × 4 cycles = 16 FMAs in progress at once, yielding 32 GFLOP/s. So, 16 accumulators keep every unit busy, and any more just queue up. It might seem like we've reached our best.

Of course we haven't. Apple Silicon uses specific registers called NEON registers which are 128 bits (16B) large - this means we can be doing 2 operations in one! By using NEON instructions from \<arm\_neon.h>, we can now use both lanes per FMA unit, giving us twice the FMAs and thus FLOPs!

```c
float64x2_t acc[K];
float64x2_t vb = vdupq_n_f64(b);   // b in both lanes
float64x2_t vc = vdupq_n_f64(c);   // c in both lanes
for (int j = 0; j < K; j++) { acc[j] = vdupq_n_f64(j); }

for (long i = 0; i < n; i++)
{
    for (int j = 0; j < K; j++)
    {
        acc[j] = vfmaq_f64(acc[j], vb, vc);   // 2 FMAs per instruction
    }
}
```

| K | time (s) | GFLOP/s | cycles per pass |
| --- | --- | --- | --- |
| 1 | 1.033 | 3.87 | 4.132 |
| 2 | 1.034 | 7.74 | 4.136 |
| 4 | 1.052 | 15.21 | 4.207 |
| 8 | 1.078 | 29.68 | 4.313 |
| 10 | 1.061 | 37.70 | 4.244 |
| 12 | 1.076 | 44.63 | 4.302 |
| 14 | 1.068 | 52.43 | 4.272 |
| 16 | 1.057 | 60.55 | 4.228 |
| 18 | 1.198 | 60.10 | 4.792 |
| 20 | 1.276 | 62.67 | 5.106 |
| 24 | 1.559 | 61.60 | 6.234 |
| 28 | 1.776 | 63.07 | 7.103 |
| 32 | 2.709 | 47.25 | 10.836 |

Boom! We get about 63 GFLOP/s at K=28, about 98% of the 64 estimate. Not bad!

Note: you might have spotted the drop at K=32 in both tables. That's a different wall: an Arm core only has 32 floating point/NEON registers, and our loop needs K + 2 of them (one per accumulator, plus b and c). At K=32 they don't all fit, so the compiler "spills" a few accumulators to memory, and loading and storing them every pass slows the whole loop down.

One thing that people familiar with the C compiler landscape might be wondering about is the -O flag (meaning optimisation level). It's one thing I initially thought might make our program faster. Now, the compiler isn't allowed to do our accumulator trick for us: floating point adds aren't associative, so splitting one chain into several would change the answer slightly. -Ofast relaxes that rule, so in theory it could. In practice though, even the -Ofast flag applied to the non-neon code doesn't get us there!

This was a pretty fun way to start the blog, and lots more is soon to follow. We'll be looking at matmul (matrix multiplication) next!

---

Want to run it yourself? [RUNGUIDE.md](RUNGUIDE.md) has the hardware, compiler flags and exact commands.
