#!/usr/bin/env python3
"""viral_campaign.py for campaigns run with the validated fast solver.

Same commands and checks (init / run / status / verify / pending, frozen
manifest, provenance records, one CPU and one thread per task).  The only
difference: the build sidecar must name viral_tree_fast.c as its source,
whose results are byte-identical to viral_tree_hpc.c
(tests/test_fast_equivalence.py, FAST_SOLVER_CHANGES.md).

viral_campaign.py itself is loaded unchanged and not edited, because every
task of the running p=30 campaign executes it at start-up.
"""

import importlib.util
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location("viral_campaign", Path(__file__).with_name("viral_campaign.py"))
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)

# sha256 of the validated viral_tree_fast.c (R1 job 18517761, R2 job 18517764)
FAST_SOURCE_SHA256 = "f1efdb7dd443bef0a043791a5232678c9659bb2c3be6fab6c959698541b034f8"
vc.SOURCE_SHA256 = FAST_SOURCE_SHA256

if __name__ == "__main__":
    sys.exit(vc.main())
