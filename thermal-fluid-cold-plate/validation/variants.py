"""Build corrected variants of upstream `toflux.src.fe_fluid`.

Two defects in `_compute_elem_residual` / `_compute_elem_stabilization` are
encoded below as exact source substitutions. Keeping them as substitutions (not
a forked copy) means the suite fails loudly with `SourceChanged` if upstream
edits those lines, instead of silently testing stale code.

    TAU1  `_compute_elem_stabilization` contracts the shape functions with the
          nodal velocities but never sums over the node index, so `ue` is not the
          squared speed at the element centroid. For a Q1 element (N_n = 1/4 at
          the centroid) a uniform flow gives ue = |u|^2 / 4, i.e. tau_1 is 2x too
          large and the convective limit damps too little.

    SUPG  The SUPG weight is tau * (u_k d/dx_k w_i), multiplied by component i of
          the strong-form residual. Two of the three momentum terms swap the free
          index i with the contracted index k. `res_press_stab`, in the same
          block, has it right -- as does the whole PSPG block -- which is what
          makes these two look like slips rather than an alternative formulation.

Variants:
    v0  upstream, unmodified
    v1  tau_1 fixed only
    v2  SUPG indices fixed only
    v3  both fixed  (the reference the tests compare against)
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CACHE = REPO / ".variants"


class SourceChanged(RuntimeError):
    """Upstream no longer contains a line this module patches."""


TAU1 = (
    '    u0 = jnp.einsum("n, nd -> nd", shp_fn, velocity.reshape(-1, self.mesh.num_dim))\n'
    '    ue = jnp.einsum("nd, nd -> ", u0, u0)',
    '    u0 = jnp.einsum("n, nd -> d", shp_fn, velocity.reshape(-1, self.mesh.num_dim))\n'
    '    ue = jnp.einsum("d, d -> ", u0, u0)',
)

SUPG_BRINKMAN = (
    'jnp.einsum("gnd, gd, gd -> gnd", grad_shp_fn, vel_gauss, vel_gauss)',
    'jnp.einsum("gnk, gk, gi -> gni", grad_shp_fn, vel_gauss, vel_gauss)',
)

SUPG_CONVECTION = (
    'jnp.einsum(\n'
    '        "gnd, gi, gm, gdm -> gni", grad_shp_fn, vel_gauss, vel_gauss, dvel_xy\n'
    '      )',
    'jnp.einsum(\n'
    '        "gnk, gk, gm, gim -> gni", grad_shp_fn, vel_gauss, vel_gauss, dvel_xy\n'
    '      )',
)

VARIANTS: dict[str, tuple[tuple[str, str], ...]] = {
    "v0": (),
    "v1": (TAU1,),
    "v2": (SUPG_BRINKMAN, SUPG_CONVECTION),
    "v3": (TAU1, SUPG_BRINKMAN, SUPG_CONVECTION),
}

LABELS = {
    "v0": "V0 upstream",
    "v1": "V1 tau_1 fixed",
    "v2": "V2 SUPG fixed",
    "v3": "V3 both fixed",
}


def _upstream_source() -> str:
    spec = importlib.util.find_spec("toflux.src.fe_fluid")
    if spec is None or spec.origin is None:
        raise SourceChanged("cannot locate toflux.src.fe_fluid")
    return pathlib.Path(spec.origin).read_text(encoding="utf-8")


def build(name: str):
    """Import (generating on first use) the fe_fluid variant called `name`."""
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; expected one of {list(VARIANTS)}")

    source = _upstream_source()
    for old, new in VARIANTS[name]:
        if old not in source:
            raise SourceChanged(
                f"variant {name}: upstream no longer contains:\n{old}\n"
                "Re-check the defect against the current upstream and update "
                "validation/variants.py."
            )
        source = source.replace(old, new)

    CACHE.mkdir(exist_ok=True)
    module_path = CACHE / f"fe_fluid_{name}.py"
    if not module_path.is_file() or module_path.read_text(encoding="utf-8") != source:
        module_path.write_text(source, encoding="utf-8")

    return _import_from_path(f"fe_fluid_{name}", module_path)


def _import_from_path(name: str, path: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
