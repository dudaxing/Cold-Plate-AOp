"""Density-method reproduction of Zhou et al. conformal cooling cases on TOFLUX.

Upstream TOFLUX supplies the density-optimization organisation, MMA, the JAX
residual/tangent architecture and implicit differentiation. This package supplies
what the 3D curved-shell cases need and upstream does not have:

  elements   corrected Hex8/Quad4 (see the module docstring for what was wrong)
"""

# FIRST, before anything pulls in NumPy: default the BLAS thread count (an
# explicit value in the environment still wins). Without it, repeated large
# sparse solves through jax.pure_callback crashed the process with Windows heap
# corruption and no traceback. Setting it in the entry scripts alone was not
# enough -- pytest imports this package directly, so the test suite ran with
# the BLAS default and died the same way. See tfopus/_threads.py.
from tfopus import _threads as _threads  # noqa: F401,E402
