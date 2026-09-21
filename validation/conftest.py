"""Locate upstream TOFLUX and put it on sys.path, and pin JAX to float64.

TOFLUX_ROOT overrides the default checkout at external/TOFLUX.

x64 is enabled here rather than in each test module because JAX latches the
setting at first array creation: any module that builds an array before the flag
is set silently runs the whole suite in float32, where element volumes and patch
tests fail on precision alone.
"""

import os
import pathlib
import sys

import jax
import pytest

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
TOFLUX_ROOT = pathlib.Path(os.environ.get("TOFLUX_ROOT", REPO / "external" / "TOFLUX"))

if (TOFLUX_ROOT / "toflux" / "src" / "fe_fluid.py").is_file():
    sys.path.insert(0, str(TOFLUX_ROOT))
    sys.path.insert(0, str(REPO))  # our own tfopus package
else:
    pytest.skip(
        f"upstream TOFLUX not found at {TOFLUX_ROOT}. "
        "Run `python scripts/setup_toflux.py`, or set TOFLUX_ROOT.",
        allow_module_level=True,
    )
