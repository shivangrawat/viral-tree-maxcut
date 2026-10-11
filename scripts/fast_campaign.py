#!/usr/bin/env python3
"""viral_campaign.py for campaigns run with the validated fast solver.

Same commands and checks (init / run / status / verify / pending, frozen
manifest, provenance records, one CPU and one thread per task).  The only
difference: the build sidecar must name viral_tree_fast.c as its source,
whose results are byte-identical to viral_tree_hpc.c (docs/validation.md).
Used for the p = 32 campaign results/torch-p32-million-v1 (docs/campaigns.md).

viral_campaign.py is loaded as a module and only its SOURCE_SHA256 is replaced
(it was kept unedited while campaigns were running it).
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
