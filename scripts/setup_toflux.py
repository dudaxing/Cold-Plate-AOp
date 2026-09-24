"""Check out upstream TOFLUX into external/ and make it importable without PETSc.

TOFLUX ships no LICENSE file, so it is not vendored into this repository.
This script extracts it from the supplementary-material zip you already have.

    python scripts/setup_toflux.py [--zip PATH] [--dest PATH]

The archive is located from --zip, then $TOFLUX_ZIP, then the repository root
and ~/Downloads. It is idempotent: re-running re-extracts and re-applies the
import guard.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO / "external" / "TOFLUX"

# The supplementary-material zip is not redistributed here. Point at your own
# copy with --zip, or set TOFLUX_ZIP. As a convenience the script also looks for
# the archive's published name in the usual download locations.
ZIP_NAME = "158_2026_4252_MOESM1_ESM.zip"
ZIP_ENV_VAR = "TOFLUX_ZIP"

# Upstream solver.py imports petsc4py at module scope and pyproject.toml does not
# declare it, so `import toflux.src.solver` fails on a clean install. Guard it the
# same way upstream already guards pypardiso.
GUARD_BEFORE = "import pyamg\nimport petsc4py.PETSc as PETSc\n"
GUARD_AFTER = """import pyamg

try:
  import petsc4py.PETSc as PETSc
except ImportError:
  PETSc = None
  warnings.warn("petsc4py not found; the PETSC linear solver is unavailable.")
"""


def find_zip() -> pathlib.Path | None:
    """Locate the supplementary zip without hard-coding anyone's filesystem."""
    env = os.environ.get(ZIP_ENV_VAR)
    if env:
        return pathlib.Path(env)
    candidates = [
        REPO / ZIP_NAME,
        pathlib.Path.home() / "Downloads" / ZIP_NAME,
        pathlib.Path.cwd() / ZIP_NAME,
    ]
    return next((c for c in candidates if c.is_file()), None)


def extract(zip_path: pathlib.Path, dest: pathlib.Path) -> None:
    if not zip_path.is_file():
        sys.exit(f"zip not found: {zip_path}\nPass --zip PATH.")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        roots = {n.split("/")[0] for n in names if "/" in n}
        if len(roots) != 1:
            sys.exit(f"expected a single top-level directory in the zip, found {roots}")
        root = roots.pop()
        staging = dest.parent / f".{dest.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        zf.extractall(staging)
    (staging / root).rename(dest)
    shutil.rmtree(staging, ignore_errors=True)
    print(f"extracted {root} -> {dest.relative_to(REPO)}")


def guard_petsc_import(dest: pathlib.Path) -> None:
    solver = dest / "toflux" / "src" / "solver.py"
    text = solver.read_text(encoding="utf-8")
    if GUARD_AFTER in text:
        print("petsc4py import already guarded")
        return
    if GUARD_BEFORE not in text:
        sys.exit(
            f"could not find the petsc4py import in {solver}.\n"
            "Upstream layout changed; update GUARD_BEFORE in this script."
        )
    solver.write_text(text.replace(GUARD_BEFORE, GUARD_AFTER), encoding="utf-8")
    print("patched toflux/src/solver.py: petsc4py import is now optional")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=pathlib.Path, default=None)
    ap.add_argument("--dest", type=pathlib.Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    zip_path = args.zip or find_zip()
    if zip_path is None:
        sys.exit(
            f"could not find {ZIP_NAME}.\n"
            f"Pass --zip PATH, or set {ZIP_ENV_VAR}, or drop the archive in "
            f"{REPO} or ~/Downloads.\n"
            "It is the supplementary material of the TOFLUX paper "
            "(doi:10.1007/s00158-026-04252-7); the upstream source is also at "
            "https://github.com/UW-ERSL/TOFLUX."
        )

    extract(zip_path, args.dest)
    guard_petsc_import(args.dest)
    print(f"\nready. run:  pytest")


if __name__ == "__main__":
    main()
