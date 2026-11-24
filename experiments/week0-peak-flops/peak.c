#include <time.h>
#include <stdio.h>
#include <math.h>

#ifndef K
#define K 4
#endif

#ifdef NEON
#define LANES 2         // doubles per FMA
#define MODE "neon"
#include <arm_neon.h>
#else

#define LANES 1
#define MODE "scalar"
#endif

double now(void)
{
    struct timespec tn;
    clock_gettime(CLOCK_MONOTONIC, &tn);
    return (double)tn.tv_sec + ((double)tn.tv_nsec * 1e-9);
}

// K independent chains of scalar FMAs: acc = acc + b*c
// returns the sum of the accumulators, and writes the loop's time to *elapsed
double worker_scalar(long n, double b, double c, double *elapsed)
{
    double acc[K];

    // give each accumulator a different starting value
    for (int j=0; j<K; j++)
    {
        acc[j] = j;
    }

    double ts = now();
    for (long i=0; i<n; i++)
    {
        for (int j=0; j<K; j++)
        {
            acc[j] = fma(b, c, acc[j]);     // b*c + acc
        }
    }
    *elapsed = now() - ts;

    // always add up the accumulators and return the sum,
    // otherwise the compiler can delete the whole loop
    double sum = 0;
    for (int j=0; j<K; j++)
    {
        sum += acc[j];
        #ifndef QUIET
        printf("acc %d:   %g\n", j, acc[j]);
        #endif
    }
    return sum;
}

// same, but each acc is a NEON register holding 2 doubles,
// so every FMA instruction does 2 FMAs at once
double worker_neon(long n, double b, double c, double *elapsed)
{
    float64x2_t acc[K];
    float64x2_t vb = vdupq_n_f64(b);    // b in both lanes
    float64x2_t vc = vdupq_n_f64(c);    // c in both lanes

    for (int j=0; j<K; j++)
    {
        acc[j] = vdupq_n_f64(j);
    }

    double ts = now();
    for (long i=0; i<n; i++)
    {
        for (int j=0; j<K; j++)
        {
            acc[j] = vfmaq_f64(acc[j], vb, vc);    // accumulator goes first!
        }
    }
    *elapsed = now() - ts;

    double sum = 0;
    for (int j=0; j<K; j++)
    {
        double lo = vgetq_lane_f64(acc[j], 0);
        double hi = vgetq_lane_f64(acc[j], 1);
        sum += lo + hi;
        #ifndef QUIET
        printf("acc %d:   %g %g\n", j, lo, hi);
        #endif
    }
    return sum;
}

int main(void)
{
    const long n = 1'000'000'000L;
    double b = 0.999999;
    double c = 6.7;
    double elapsed;

    #ifdef NEON
    double sum = worker_neon(n, b, c, &elapsed);
    #else
    double sum = worker_scalar(n, b, c, &elapsed);
    #endif

    double flops = n * 2.0 * LANES * K;
    double gflops = flops / elapsed * 1e-9;
    double hz = 4.0e9;
    double cycles_per_pass = elapsed*hz/n;

    #ifdef QUIET
    printf("%-6s %6.3f s  %6.2f GFLOP/s  %6.3f cycles/pass  (sum %g)\n",
           MODE, elapsed, gflops, cycles_per_pass, sum);
    #else
    printf("mode:  %s\n", MODE);
    printf("time:  %.3f s\n", elapsed);
    printf("speed: %.2f GFLOP/s\n", gflops);
    printf("cycles per loop pass: %.3f\n", cycles_per_pass);
    printf("sum:   %g\n", sum);
    #endif

    return 0;
}
