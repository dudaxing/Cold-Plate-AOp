"""The BLAS thread default: what it sets, what it leaves alone, when it is late.

Each case runs in a fresh interpreter. The variables are read once, when the
BLAS library loads, and this process loaded NumPy long ago -- so checking
os.environ here would test nothing about a clean start, and asserting a fixed
value here would fail for anyone who legitimately set their own.
"""

import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS")


def _fresh(code: str, **env_overrides: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in VARS}
    env.update(env_overrides)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=REPO,
        capture_output=True, text=True, timeout=120,
    )


PRINT_VARS = (
    "import os, tfopus; "
    f"print(' '.join(os.environ[v] for v in {VARS!r}))"
)


def test_an_empty_environment_gets_the_default():
    out = _fresh(PRINT_VARS)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["8"] * len(VARS)


def test_an_explicit_setting_is_left_alone():
    """setdefault semantics: this is a default, not a cap."""
    out = _fresh(PRINT_VARS, OPENBLAS_NUM_THREADS="4")
    assert out.returncode == 0, out.stderr
    values = dict(zip(VARS, out.stdout.split()))
    assert values["OPENBLAS_NUM_THREADS"] == "4"
    assert values["OMP_NUM_THREADS"] == "8"


def test_setting_it_after_numpy_warns_instead_of_pretending():
    """Too late to affect a BLAS that has already started, so say so."""
    code = (
        "import warnings\n"
        "warnings.simplefilter('always')\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    import numpy\n"
        "    import tfopus\n"
        "print(sum('after NumPy' in str(w.message) for w in caught))\n"
    )
    out = _fresh(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


def test_no_warning_when_it_runs_first():
    code = (
        "import warnings\n"
        "warnings.simplefilter('always')\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    import tfopus\n"
        "    import numpy\n"
        "print(sum('after NumPy' in str(w.message) for w in caught))\n"
    )
    out = _fresh(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0"
