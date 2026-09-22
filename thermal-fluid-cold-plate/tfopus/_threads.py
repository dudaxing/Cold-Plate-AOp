"""Cap the BLAS thread count before NumPy loads. Import this FIRST.

On a 32-core Windows machine, repeated large sparse solves through
`jax.pure_callback` crash the process with heap corruption
(Windows 0xC0000374), immediately after OpenBLAS prints

    precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata

The crash is silent from Python's side: no traceback, no non-zero exit from the
shell pipeline, just a truncated log. It killed an R1f run after the first of
four analyses had already printed, which is exactly the failure mode that looks
like a hang or a clean stop.

Capping the thread count is the remedy OpenBLAS itself suggests, and it removes
the crash. The solves here are one sparse factorisation at a time, so the lost
parallelism costs little.

`setdefault` means an explicit value in the environment still wins.
"""

from __future__ import annotations

import os

DEFAULT_THREADS = "8"

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, DEFAULT_THREADS)
