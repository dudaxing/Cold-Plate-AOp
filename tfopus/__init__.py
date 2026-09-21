"""Density-method reproduction of Zhou et al. conformal cooling cases on TOFLUX.

Upstream TOFLUX supplies the density-optimization organisation, MMA, the JAX
residual/tangent architecture and implicit differentiation. This package supplies
what the 3D curved-shell cases need and upstream does not have:

  elements   corrected Hex8/Quad4 (see the module docstring for what was wrong)
"""
