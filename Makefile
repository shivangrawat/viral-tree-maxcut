# VIRAL on the 3-regular tree: build, self-test and tables.
#
#   make                 build the three solvers with the campaign flags
#   make check           run check.sh on each solver (a few minutes)
#   make check-sources   prove the documented sources compile to the code that ran
#   make table           print Table I of the paper from the committed results
#   make clean
#
# The flags are those of the campaign builds recorded in results/*/manifest.json:
#   -Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3 -fopenmp, plus -ffp-contract=off
#   for viral_tree_fast and viral_tree_fast2.
# The campaigns used gcc 14.3.1 on RHEL 10 (x86-64-v3 by default, so gcc fuses
# multiply-adds in viral_tree_hpc).  On another x86-64 system add
#   make CFLAGS_EXTRA=-march=x86-64-v3
# to get the same fusion in viral_tree_hpc; fast and fast2 call fma() explicitly
# and do not depend on it.  See README.md, "Reproducing the published numbers".
#
# macOS: Apple clang needs an OpenMP runtime.  `brew install libomp`, or point
# LIBOMP at any prefix that has include/omp.h and lib/libomp.dylib, e.g.
#   make LIBOMP=$HOME/miniforge3/lib/python3.12/site-packages/torch
# Each binary is then relinked to that libomp.dylib and re-signed ad hoc.

UNAME_S := $(shell uname -s)

ifeq ($(origin CC),default)
  ifeq ($(UNAME_S),Darwin)
    CC = clang
  else
    CC = gcc
  endif
endif

CFLAGS_CAMPAIGN = -Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3
CFLAGS_EXTRA   ?=

ifeq ($(UNAME_S),Darwin)
  ifeq ($(origin LIBOMP),undefined)
    LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
  endif
  OMP_CFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  OMP_LIBS   = -L$(LIBOMP)/lib -lomp -Wl,-headerpad_max_install_names
  # point the binary at $(LIBOMP)'s libomp.dylib, whatever install name it was built with
  POSTLINK   = install_name_tool -change "$$(otool -D $(LIBOMP)/lib/libomp.dylib | tail -n 1)" \
               $(LIBOMP)/lib/libomp.dylib $@ 2>/dev/null && codesign --force --sign - $@ 2>/dev/null \
               || { echo "relinking $@ to $(LIBOMP)/lib/libomp.dylib failed"; exit 1; }
else
  OMP_CFLAGS = -fopenmp
  OMP_LIBS   =
  POSTLINK   = true
endif

SOLVERS = viral_tree_hpc viral_tree_fast viral_tree_fast2

all: $(SOLVERS)

viral_tree_hpc: viral_tree_hpc.c | omp
	$(CC) $(CFLAGS_CAMPAIGN) $(CFLAGS_EXTRA) $(OMP_CFLAGS) $< $(OMP_LIBS) -lm -o $@
	@$(POSTLINK)

viral_tree_fast viral_tree_fast2: %: %.c | omp
	$(CC) $(CFLAGS_CAMPAIGN) -ffp-contract=off $(CFLAGS_EXTRA) $(OMP_CFLAGS) $< $(OMP_LIBS) -lm -o $@
	@$(POSTLINK)

# Fails early, with a hint, when macOS has no OpenMP runtime to build against.
omp:
	@if [ "$(UNAME_S)" = Darwin ] && [ ! -f "$(LIBOMP)/include/omp.h" ]; then \
	    echo "No OpenMP runtime under LIBOMP='$(LIBOMP)'."; \
	    echo "Run 'brew install libomp', or 'make LIBOMP=<prefix with include/omp.h and lib/libomp.dylib>'."; \
	    exit 1; \
	fi

# check.sh uses a fixed scratch directory, so the three runs are sequential.
# VIRAL_PAR_MIN=1 makes fast2 split even small levels between threads, so the
# thread-independence test exercises its threaded kernel.
check: $(SOLVERS)
	./check.sh ./viral_tree_hpc
	./check.sh ./viral_tree_fast
	VIRAL_PAR_MIN=1 ./check.sh ./viral_tree_fast2

check-sources: | omp
	CC="$(CC)" CFLAGS_OMP="$(OMP_CFLAGS)" bash tests/check_sources.sh

table:
	python3 merge_results.py --table results/torch-p*-*/jobs

clean:
	rm -f $(SOLVERS) .build.lock
	rm -rf check_tmp

.PHONY: all omp check check-sources table clean
