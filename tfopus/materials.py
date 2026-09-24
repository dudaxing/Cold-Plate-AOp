"""Material data and the Zhou interpolations, in TOFLUX's density convention.

Zhou's pseudo-density has gamma = 1 for FLUID; TOFLUX's has s = 1 for SOLID.
Everything here is written in s, with gamma = 1 - s substituted in, so the
convention flip happens once, here, and nowhere else in the code.

Zhou equation 15:  alpha(gamma) = a_min + (a_max - a_min) * q_a(1-gamma)/(q_a+gamma)
Zhou equation 17:  k(gamma)     = k_f   + (k_s   - k_f)   * q_k(1-gamma)/(q_k+gamma)

Substituting gamma = 1 - s gives the form used below, q*s/(q + 1 - s), which is
TOFLUX's convex RAMP with the reciprocal parameter: q_toflux = 1/q_zhou. The
sphere's reported q_alpha = q_k = 0.2 is therefore q_toflux = 5, not 0.2.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Phase:
    """One material phase. Zhou table 2."""

    name: str
    density: float  # kg/m^3
    conductivity: float  # W/(m K)
    specific_heat: float  # J/(kg K)
    viscosity: float | None = None  # Pa s, fluid only

    @property
    def volumetric_heat_capacity(self) -> float:
        """rho * c_p, J/(m^3 K)."""
        return self.density * self.specific_heat


WATER = Phase("water", 1000.0, 0.6, 4187.0, 0.001)
ALUMINIUM = Phase("aluminium alloy", 2700.0, 237.0, 880.0)


@dataclasses.dataclass(frozen=True)
class ThermoFluidMaterial:
    """The two phases plus the interpolation parameters.

    Attributes:
      advection_rho_cp_policy: how rho*c_p in the advection term is treated.
        "fluid_constant" uses the fluid value over the whole domain.

        Zhou specifies the RAMP interpolations for alpha and k but never states
        a two-phase rule or grey-region treatment for rho*c_p, so this is a
        reproduction choice, not a recovered author setting. It is defensible
        because in the ideal zero-velocity solid limit the advection term drops
        out entirely and the solid region satisfies -div(k_s grad T) = Q
        regardless of which c_p the (vanishing) advection carried. At finite
        Brinkman penalty and in grey regions the two extensions differ, and by
        how much has NOT been verified here; nor does agreement on a final fixed
        geometry imply two extensions would follow the same optimisation path to
        the same local optimum.
    """

    fluid: Phase = WATER
    solid: Phase = ALUMINIUM
    alpha_min: float = 0.0
    alpha_max: float = 1.0e6
    q_alpha_zhou: float = 0.2
    q_k_zhou: float = 0.2
    advection_rho_cp_policy: str = "fluid_constant"

    def __post_init__(self):
        if self.advection_rho_cp_policy != "fluid_constant":
            raise ValueError(
                "only 'fluid_constant' is implemented; adding a two-phase rho*c_p "
                "interpolation is a separate modelling decision"
            )

    @property
    def b_f(self) -> float:
        """Advective volumetric heat capacity, J/(m^3 K).

        The single number used by the thermal residual, the thermal
        stabilisation parameter, the advection term of the thermal compliance
        and the port enthalpy accounting -- so that all four cannot drift apart.
        """
        return self.fluid.volumetric_heat_capacity

    @property
    def viscosity(self) -> float:
        return self.fluid.viscosity

    @property
    def density(self) -> float:
        return self.fluid.density


def ramp_zhou(s, prop_fluid: float, prop_solid: float, q_zhou: float):
    """Zhou's RAMP in TOFLUX's solid-fraction variable.

        prop(s) = prop_fluid + (prop_solid - prop_fluid) * q*s / (q + 1 - s)

    s = 0 gives prop_fluid, s = 1 gives prop_solid, for any q > 0.
    """
    return prop_fluid + (prop_solid - prop_fluid) * (q_zhou * s) / (q_zhou + 1.0 - s)


def ramp_toflux_convex(s, prop_min: float, prop_max: float, q_toflux: float):
    """TOFLUX's convex RAMP, for the equivalence check only.

        prop(s) = prop_min + (prop_max - prop_min) * s / (1 + q*(1 - s))

    Identical to `ramp_zhou` when q_toflux = 1 / q_zhou.
    """
    return prop_min + (prop_max - prop_min) * s / (1.0 + q_toflux * (1.0 - s))


def brinkman_penalty(s, material: ThermoFluidMaterial, alpha_max: float | None = None):
    """alpha(s), Zhou equation 15. `alpha_max` overrides for continuation."""
    a_max = material.alpha_max if alpha_max is None else alpha_max
    return ramp_zhou(s, material.alpha_min, a_max, material.q_alpha_zhou)


def conductivity(s, material: ThermoFluidMaterial):
    """k(s), Zhou equation 17."""
    return ramp_zhou(
        s, material.fluid.conductivity, material.solid.conductivity, material.q_k_zhou
    )
