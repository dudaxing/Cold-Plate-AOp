"""Deprecated: superseded by scripts/zhao2d_optimise.py.

This script had the terminal-pairing defect -- it saved the design MMA produced
after the last evaluated iterate, so the saved design and the last recorded
metrics belonged to different designs. The fix lives in tfopus/zhao2d_driver.py,
which both the mechanism check and the main run now share, so this entry point
is kept only as a pointer rather than as a second loop to maintain.

    python scripts/zhao2d_optimise.py --coarse --budget 25   # mechanism check
    python scripts/zhao2d_optimise.py --budget 300           # the main case
"""

import sys

sys.exit(__doc__)
