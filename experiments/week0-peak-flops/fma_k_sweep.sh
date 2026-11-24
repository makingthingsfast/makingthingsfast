#!/bin/sh
# usage: ./fma_k_sweep.sh OPT QUIET K...
#   e.g. ./fma_k_sweep.sh 2 1 12 16 20 24
# extra compiler flags go in EXTRA, e.g. for the NEON version:
#   EXTRA=-DNEON ./fma_k_sweep.sh 2 1 12 16 20 24

OPT=$1
QUIET=$2
shift 2

if [ "$QUIET" = 1 ]; then QFLAG="-DQUIET"; else QFLAG=""; fi

for k in "$@"; do
	clang -std=c23 -O$OPT -DK=$k $QFLAG $EXTRA peak.c -o peak || exit 1
	printf "K=%-3s " "$k"
	./peak
done
