"""Default the BLAS thread count before NumPy loads. Import this FIRST.

On a 32-core Windows machine, repeated large sparse solves through
`jax.pure_callback` crashed the process with heap corruption
(Windows 0xC0000374), immediately after OpenBLAS printed

    precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata

The crash is silent from Python's side: no traceback, no non-zero exit from the
shell pipeline, just a truncated log. It killed an R1f run after the first of
four analyses had already printed, which is exactly the failure mode that looks
like a hang or a clean stop. Setting a thread count removed it on that machine;
that is a local mitigation that has been observed to work, not a root-cause
proof, and it has not been reproduced elsewhere.

What this module does, and what it does not:

* It sets a DEFAULT, not a cap. `setdefault` semantics: a value already in the
  environment is left alone, so OPENBLAS_NUM_THREADS=4 stays 4.
* It has an effect only if it runs before the BLAS library loads. OpenBLAS
  reads these variables once, at startup; setting them afterwards does not
  resize a thread pool that already exists. If NumPy was imported first and
  this module had to set a variable, it warns instead of pretending.
* Which variable applies depends on how the BLAS was built: OpenBLAS reads
  OPENBLAS_NUM_THREADS in a pthreads build and OMP_NUM_THREADS in an OpenMP
  one, MKL reads MKL_NUM_THREADS. All are set, since the build is not known.

The solves here are one sparse factorisation at a time, so the lost
parallelism costs little.
"""

from __future__ import annotations

import os
import sys
import warnings

DEFAULT_THREADS = "8"
VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_set_here = [v for v in VARIABLES if v not in os.environ]
for _var in _set_here:
    os.environ[_var] = DEFAULT_THREADS

if _set_here and "numpy" in sys.modules:
    warnings.warn(
        f"tfopus set {', '.join(_set_here)} after NumPy was already imported; "
        "a BLAS library that has already started will not pick this up. Import "
        "tfopus before NumPy, or set the variables in the environment.",
        RuntimeWarning,
        stacklevel=2,
    )
