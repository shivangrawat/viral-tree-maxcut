# viral_tree_hpc -- VIRAL on the 3-regular tree (MaxCut lower bound)
#
#   make            build ./viral_tree_hpc   (portable -O3; runs on any node)
#   make check      build and run the self-tests (about 1 minute)
#   make clean
#
# Any OpenMP-capable C compiler works (gcc, clang, icc: CC=icc make).
# CPU-specific flags are deliberately NOT used: the run time is dominated by
# sin/cos from libm, and -march=native was measured to change neither the
# speed nor the results.  A binary built on a login node runs on the compute
# nodes.  (make native exists for completeness.)

CC      ?= gcc
CFLAGS  ?= -O3 -fopenmp -Wall -Wextra
LDLIBS   = -lm

all: viral_tree_hpc

viral_tree_hpc: viral_tree_hpc.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

native: viral_tree_hpc.c
	$(CC) $(CFLAGS) -march=native -o viral_tree_hpc $< $(LDLIBS)

check: viral_tree_hpc
	./check.sh

clean:
	rm -f viral_tree_hpc .build.lock
	rm -rf check_tmp

.PHONY: all native check clean
