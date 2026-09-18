#!/bin/bash
# check.sh -- self-tests for viral_tree_hpc.   Run:  make check
#
# 1. p=2 : Monte Carlo vs the EXACT value from enumerating all 2^14 seed
#          configurations (exact_small_p.py; hard-coded here so that numpy is
#          not required):  c_VIRAL(2) = 12380/16384 = 0.7556152344
# 2. p=12: n_cut for 200000 samples must equal the reference value 178239
#          obtained with the original code (bit-exact on x86-64 + glibc;
#          a difference of a few units means last-bit libm/FMA differences
#          on this machine and is harmless; a large difference is a bug).
# 3. thread independence: p=8, 50000 samples, 1 thread vs several threads
# 4. resume/extension: p=8 job 5, 10000 then extended to 30000, must equal a
#    fresh 30000-sample run.
set -u
cd "$(dirname "$0")"
BIN=./viral_tree_hpc
T=check_tmp
rm -rf $T && mkdir -p $T/a $T/b $T/c $T/d $T/e
NT=${OMP_NUM_THREADS:-4}
fail=0
field() { awk -v k="$2" '$1==k {print $2}' "$1"; }

echo "== test 1: p=2, 2e6 samples vs exact 0.7556152344"
OMP_NUM_THREADS=$NT $BIN 2 2000000 0 params/angles_p2.txt $T/a > $T/a/log 2>&1 || { echo "   run failed"; cat $T/a/log; fail=1; }
N=$(field $T/a/viral_tree_p2_job0.txt n_done); K=$(field $T/a/viral_tree_p2_job0.txt n_cut)
python3 - "$N" "$K" <<'EOF' || fail=1
import sys, math
n, k = int(sys.argv[1]), int(sys.argv[2]); x0 = 12380/16384
x = k/n; se = math.sqrt(x*(1-x)/n); dev = (x-x0)/se
print(f"   x = {x:.6f} +- {se:.6f}   exact {x0:.7f}   deviation {dev:+.2f} sigma")
ok = abs(dev) < 5
print("   PASS" if ok else "   FAIL: more than 5 sigma from the exact value")
sys.exit(0 if ok else 1)
EOF

echo "== test 2: p=12, 200000 samples, n_cut vs reference 178239"
OMP_NUM_THREADS=$NT $BIN 12 200000 0 params/angles_p12.txt $T/b > $T/b/log 2>&1 || { echo "   run failed"; cat $T/b/log; fail=1; }
K=$(field $T/b/viral_tree_p12_job0.txt n_cut)
D=$(( K - 178239 )); [ $D -lt 0 ] && D=$(( -D ))
if [ "$K" = "178239" ]; then echo "   n_cut = $K   PASS (bit-exact with the reference code)"
elif [ $D -le 30 ];      then echo "   n_cut = $K   PASS (differs by $D: last-bit libm/FMA differences on this machine; harmless)"
else echo "   n_cut = $K   FAIL (differs by $D from 178239)"; fail=1; fi

echo "== test 3: thread independence, p=8, 50000 samples, 1 thread vs $NT threads"
OMP_NUM_THREADS=1   $BIN 8 50000 0 params/angles_p8.txt $T/c > $T/c/log1 2>&1 || fail=1
OMP_NUM_THREADS=$NT $BIN 8 50000 0 params/angles_p8.txt $T/d > $T/d/logN 2>&1 || fail=1
K1=$(field $T/c/viral_tree_p8_job0.txt n_cut); KN=$(field $T/d/viral_tree_p8_job0.txt n_cut)
if [ "$K1" = "$KN" ] && [ -n "$K1" ]; then echo "   n_cut = $K1 in both   PASS"; else echo "   n_cut = $K1 vs $KN   FAIL"; fail=1; fi

echo "== test 4: resume/extension, p=8 job 5, 10000 -> 30000 vs fresh 30000"
OMP_NUM_THREADS=$NT $BIN 8 10000 5 params/angles_p8.txt $T/e > $T/e/log1 2>&1 || fail=1
OMP_NUM_THREADS=$NT $BIN 8 30000 5 params/angles_p8.txt $T/e > $T/e/log2 2>&1 || fail=1
OMP_NUM_THREADS=$NT $BIN 8 30000 5 params/angles_p8.txt $T/d > $T/d/log5 2>&1 || fail=1
KE=$(field $T/e/viral_tree_p8_job5.txt n_cut); KF=$(field $T/d/viral_tree_p8_job5.txt n_cut)
grep -q RESUMING $T/e/log2 || { echo "   did not resume"; fail=1; }
if [ "$KE" = "$KF" ] && [ -n "$KE" ]; then echo "   n_cut = $KE in both   PASS"; else echo "   n_cut = $KE vs $KF   FAIL"; fail=1; fi

echo "== merge_results.py on the test outputs"
python3 merge_results.py $T/a $T/b -p 12 | sed 's/^/   /' | head -12 || fail=1

rm -rf $T
if [ $fail = 0 ]; then echo; echo "ALL CHECKS PASSED"; else echo; echo "SOME CHECKS FAILED"; exit 1; fi
