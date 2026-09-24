"""Default the BLAS thread count before NumPy loads. Import this FIRST.

On a 32-core Windows machine, a run of repeated large sparse solves through
`jax.pure_callback` crashed with heap corruption (Windows 0xC0000374),
immediately after OpenBLAS printed

    precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata

The crash is silent from Python's side: no traceback, no non-zero exit from the
shell pipeline, just a truncated log. It killed an R1f run after the first of
four analyses had already printed, which is exactly the failure mode that looks
like a hang or a clean stop.

Measured here (SciPy 1.17.1, jaxlib 0.11.0, 32 CPUs). SciPy bundles OpenBLAS
0.3.30 built with MAX_THREADS=24, so its thread pool is 24 by default; this
module makes it 8. XLA's threads call that OpenBLAS two ways: jaxlib's CPU
LAPACK is bound to it through `scipy.linalg.cython_lapack` (the `jnp.linalg.inv`
in tfopus/elements.py runs at every Gauss point), and upstream's spsolve runs
inside `jax.pure_callback` -- over the R1d and R1h anchors, 34 solves landed on
21 different threads. On that anchor workload a pool of 8 printed no warning in
three runs; a pool of 24 printed it in both of two, and one of them, after
writing its results, segfaulted at exit. The numbers were identical throughout.

Reported by the conformal-cooling repository after it adopted this default, not
re-run here: re-gating a saved 5200-element state -- residual assembly only, no
sparse solve -- had died in three of three runs with a pool of 24; with 8 it ran
clean (exit 0, no warning), and its coupled check, a mutation run and its full
suite (193 passed) printed no warning either.

Read from OpenBLAS 0.3.30's source (driver/others/memory.c, the allocator that
prints the warning), which accounts for that: a table of 50 buffer slots. Each
call to an optimised LAPACK routine such as dgetrf holds one while it runs, and
each worker of OpenBLAS's own pool holds one for life -- 23 at a pool of 24, 7
at 8. The warning means no slot was free. After it, every call that finds the
table full maps a new buffer and appends a record to a 512-entry heap array
with no bound check, so a process that keeps overflowing writes past it. Slots
belong to calls, not threads: what counts is how many calls are in flight at
once, not which threads make them.

Not established: that this is what crashed -- no crash has been caught with a
native stack; and that 8 is always enough -- it leaves 43 slots for calls in
flight, which more concurrent XLA work could still exceed, and a larger explicit
OPENBLAS_NUM_THREADS, which wins over this default, shrinks.

Evaluated and not adopted: sending upstream's spsolve to one dedicated thread,
as proposed in the conformal-cooling repository. It is bit-identical on the
anchors (C, Psi, the flow and temperature states, the design gradient), but the
solves never overlap one another, so it does not change how many slots are in
use, and with it a pool of 24 still printed the warning. It also moves the solve from XLA's threads, which
flush subnormals, to one that does not: identical here, not guaranteed.

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
