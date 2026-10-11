#!/bin/bash
# check_sources.sh -- prove that the documented C sources compile to the code
# that produced the published results.
#
#   make check-sources                               (sets CC and CFLAGS_OMP)
#   CC=gcc CFLAGS_OMP=-fopenmp bash tests/check_sources.sh
#
# The campaigns ran the sources of commit d578d35 (tag results-10M).  Since
# then only comments have changed in viral_tree_hpc.c, viral_tree_fast.c,
# viral_tree_fast2.c and tests/seed_audit.c.  For each of them this checks
#
#   comments  With comments removed and whitespace runs collapsed (string and
#             character literals untouched), the file equals its results-10M
#             version line for line.  Covers every #if branch, including the
#             glibc sincos() code the campaigns ran on Linux.
#   assembly  Both versions compile to identical assembly with the campaign
#             flags (same compiler, same file name).  With clang on macOS this
#             covers the non-glibc branches; with gcc on Linux, the campaign path.
#   hash      (solvers only) scripts/viral_campaign.py maps the sha256 of the
#             current file to the campaign sha256 recorded in the manifests.
#
# Prints one line per file and exits 1 if any check fails.
set -euo pipefail
cd "$(dirname "$0")/.."

REF=d578d35d80d22aef19900f51c8310809828862a5          # tag results-10M
if tag=$(git rev-parse -q --verify 'results-10M^{commit}'); then
    [ "$tag" = "$REF" ] || { echo "tag results-10M points to $tag, not $REF"; exit 1; }
fi
git cat-file -e "$REF^{commit}" 2>/dev/null \
    || { echo "commit $REF is not in this clone; run: git fetch origin --tags"; exit 1; }

CC=${CC:-cc}
CFLAGS_OMP=${CFLAGS_OMP:--fopenmp}
CAMPAIGN="-Wall -Wextra -DBATCH_MAX_PER_THREAD=1 -O3"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0

printf '%-22s %-9s %-9s %s\n' file comments assembly hash
for f in viral_tree_hpc.c viral_tree_fast.c viral_tree_fast2.c tests/seed_audit.c; do
    b=$(basename "$f")
    rm -rf "$tmp/old" "$tmp/new"
    mkdir "$tmp/old" "$tmp/new"
    git show "$REF:$f" > "$tmp/old/$b"
    cp "$f" "$tmp/new/$b"

    if python3 -B - "$tmp/old/$b" "$tmp/new/$b" <<'PY'
import re, sys
# comments -> one space; literals kept verbatim; runs of blanks -> one space
TOKEN = re.compile(rb'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|[ \t]+', re.S)
def code(path):
    text = TOKEN.sub(lambda m: m.group(0) if m.group(0)[:1] in b"\"'" else b" ",
                     open(path, "rb").read())
    return [line.strip() for line in text.splitlines() if line.strip()]
sys.exit(code(sys.argv[1]) != code(sys.argv[2]))
PY
    then comments=ok; else comments=FAIL; fail=1; fi

    case $b in
        seed_audit.c)     flags="-O2" ;;
        viral_tree_hpc.c) flags="$CAMPAIGN $CFLAGS_OMP" ;;
        *)                flags="$CAMPAIGN -ffp-contract=off $CFLAGS_OMP" ;;
    esac
    for v in old new; do
        # shellcheck disable=SC2086  (flags is a word list)
        (cd "$tmp/$v" && $CC $flags -S -o "../$v.s" "$b")
    done
    if cmp -s "$tmp/old.s" "$tmp/new.s"; then assembly=ok; else assembly=FAIL; fail=1; fi

    hash=-
    if [ "$b" != seed_audit.c ]; then
        if python3 -B - "$f" "$tmp/old/$b" <<'PY'
import hashlib, importlib.util, sys
spec = importlib.util.spec_from_file_location("viral_campaign", "scripts/viral_campaign.py")
vc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vc)
current, campaign = (hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sys.argv[1:3])
sys.exit(vc.campaign_source_sha(current) != campaign)
PY
        then hash=ok; else hash=FAIL; fail=1; fi
    fi
    printf '%-22s %-9s %-9s %s\n' "$f" "$comments" "$assembly" "$hash"
done

if [ $fail = 0 ]; then
    echo "SOURCES OK: only comments differ from results-10M"
else
    echo "SOURCES CHANGED: see FAIL above"
    exit 1
fi
