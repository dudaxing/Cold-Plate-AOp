"""R1v, part 1: R1t's MMA checkpoint, re-signed with today's binding. Nothing is solved.

The contract proposed in the review of 7786ce7, its first part. R1t saved MMA's
state after its twenty updates, signed with the binding of its day: the model,
the thermal path and the scale, but not the optimisation problem. c31b3c2 added
the optimisation problem to the binding, so every run today refuses R1t's
checkpoint. This script re-signs it with
`zhao2d_fineflow.migrate_legacy_checkpoint`, and changes nothing else.

The new block is rebuilt from R1t's own record and the source R1t was run
with, not from today's defaults:

* the source the rebuild runs through -- the spec and its defaults, the
  configuration, the design mesh, the filter and the projection -- and R1t's
  own script, which built the spec and the configuration, must hash as R1t's
  record has them: byte for byte, or, for these named text files only, up to
  their line endings (which one is recorded per file);
* the configuration is rebuilt field by field from the fingerprint R1t
  recorded, and must give that fingerprint back exactly. It is also what
  R1t's script builds;
* the spec is what R1t's script built, `Zhao2DSpec()`, from the locked
  source. Its sixteen fields in R1t's model identity are compared with the
  record; the nine others are listed as the locked source's;
* the design side -- `Zhao2DProblem` on h, the constructor D runs for its own
  design side -- is built from those two. Its mesh identity and its number of
  variables are R1t's recorded ones. No fine mesh is built.

Checked in stages. A failed stage writes the record and exits non-zero before
the next:

* the inputs: the checkpoint in R1t's fields is the one its record describes
  (the state's hash, the KKT residual, the updates, the binding), its design
  is R1t's terminal design, and its binding is a legacy one;
* the locked source; the configuration and the spec; the design side;
* the migration: the state array, the KKT residual, the updates and the
  number of variables are unchanged, bit for bit; the binding changes only in
  its entry, which gains the optimisation block;
* dry, for part 2, which is not authorised (nothing is run or saved for it):
  appending twenty updates (20 -> 40) is accepted and clears the budget's flag
  only; the driver's own resume, on a stand-in with D's identity and design
  side, accepts the appended checkpoint and stops before its first
  evaluation, which would be at R1t's raw terminal design;
* R1t's files: read, never written; their hashes after the run are the ones
  before it.

On a checkout with LF line endings the dry stage stops at the reference
file's bytes until its recorded CRLF bytes are restored, as for R1u's test:
the scale's binding names those bytes, and part 2's run would too.

Writes zhao2d_r1v_migration.json and zhao2d_r1v_mma_migrated.npz, new files.

    python scripts/zhao2d_r1v_migrate.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import dataclasses
import hashlib
import importlib
import json
import time
from types import SimpleNamespace

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

import toflux.src.mma as tf_mma  # noqa: E402

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_driver as drv  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
from zhao2d_r1m_flow_check import mesh_identity  # noqa: E402

ALPHA_MAX, BETA = 1.0e7, 32.0
FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE = 2, 4, 3
THERMAL_PATH = "linear"
MOVE_LIMIT = 0.1
R1T_UPDATES, APPENDED = 20, 20
APPENDED_PHASE = "r1v-d-vp32-linear"
MIGRATION_VERSION = "r1v-1"
RECORD = "zhao2d_r1v_migration.json"
FIELDS = "zhao2d_r1v_mma_migrated.npz"
INPUTS = ("zhao2d_r1t.json", "zhao2d_r1t_fields.npz")
# What the rebuild runs through: the spec and its defaults, the configuration
# and its forms, the design mesh, the filter and the projection -- and R1t's
# script, which built the spec and the configuration.
LOCKED_SOURCE = ("tfopus/zhao2d.py", "tfopus/zhao2d_r1.py", "tfopus/fe_flow.py",
                 "tfopus/fe_thermal.py", "tfopus/mesh.py", "tfopus/geometry.py",
                 "tfopus/design.py", "tfopus/projection.py", "scripts/zhao2d_r1t_d_continue.py")
R1T_BUILDS = ("spec = z.Zhao2DSpec()",
              "config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)")
STAGES = ("the inputs", "the locked source", "the configuration and the spec", "the design side",
          "the migration", "dry, for part 2", "R1t's files")


class StopBeforeSolving(Exception):
    """Raised by the dry stage's evaluator: the resume was accepted, and nothing is solved."""


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def locked_match(path: pathlib.Path, recorded: str) -> str | None:
    """How a file matches the hash a run recorded for it, if it does.

    Its bytes; or, for the named text files this is used on, its text up to
    line endings, since a checkout may convert them. Nothing looser.
    """
    data = pathlib.Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() == recorded:
        return "bytes"
    lf = data.replace(b"\r\n", b"\n")
    if hashlib.sha256(lf).hexdigest() == recorded:
        return "up to line endings: recorded with LF"
    if hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest() == recorded:
        return "up to line endings: recorded with CRLF"
    return None


def config_from_fingerprint(text: str) -> r1.R1Config:
    """The R1Config a fingerprint records, field by field; refused unless it gives it back."""
    payload = json.loads(text)
    fields = {f.name: f for f in dataclasses.fields(r1.R1Config)}
    if set(payload) != set(fields):
        raise ValueError(f"the fingerprint names {sorted(payload)}; R1Config has {sorted(fields)}")
    config = r1.R1Config(**{
        name: (type(fields[name].default)(**value)
               if dataclasses.is_dataclass(fields[name].default) else value)
        for name, value in payload.items()})
    if config.fingerprint() != text:
        raise ValueError("the configuration rebuilt from the fingerprint does not give it back")
    return config


def flags(checkpoint: drv.MMACheckpoint) -> dict:
    state = tf_mma.MMAState.from_array(checkpoint.state_array.copy(), checkpoint.num_design_var)
    return {"is_converged": bool(state.is_converged), "epoch": int(state.epoch),
            "change_design_var": float(state.change_design_var),
            "kkt_norm_declared": float(state.kkt_norm), "kktnorm": checkpoint.kktnorm}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    inputs, out = args.inputs, args.out
    existing = [n for n in (RECORD, FIELDS) if (out / n).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{out} already holds {', '.join(existing)}, which is cited evidence; write "
                 "elsewhere with --out DIR, or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    inputs_before = {n: sha256_file(inputs / n) for n in INPUTS}
    t_rec = load_json(inputs / "zhao2d_r1t.json")
    with np.load(inputs / "zhao2d_r1t_fields.npz") as f:
        legacy = drv.MMACheckpoint.from_npz(f)
        x_terminal = np.asarray(f["design"], dtype=np.float64)
    cp_rec = t_rec["mma_checkpoint"]
    old = json.loads(legacy.binding)

    record = {
        "note": ("R1v, part 1: R1t's MMA checkpoint re-signed with today's binding, which also "
                 "names the optimisation problem; the state is carried over untouched. The "
                 "optimisation block is rebuilt from R1t's record and the source R1t was run "
                 "with. Nothing is solved, no MMA update is made, no fine mesh is built, and "
                 "nothing is appended to a file."),
        "provenance": provenance(),
        "inputs_sha256": inputs_before,
        "contract": {"part": 1, "migrates": "results/zhao2d_r1t_fields.npz, mma_*",
                     "solves": 0, "mma_updates": 0, "fine_meshes_built": False,
                     "part_2": "not authorised; its append is checked dry, nothing saved",
                     "authorised": ("by the user, part 1 only, on the local CPU, as the review "
                                    "of 7786ce7 states it")},
    }
    for script in (pathlib.Path(__file__).resolve(), REPO / "scripts" / "zhao2d_r1m_flow_check.py",
                   REPO / "scripts" / "zhao2d_r1t_d_continue.py"):
        record["provenance"]["source_sha256"][script.relative_to(REPO).as_posix()] = sha256_file(script)
    failures, checkpoints, cost, saved = [], [], {}, {}

    def require(ok, what: str) -> bool:
        if not ok:
            failures.append(what)
        return bool(ok)

    def finish() -> None:
        record["checkpoints"] = checkpoints
        if saved:
            np.savez_compressed(out / FIELDS, **saved)
            record["fields"] = {"file": FIELDS, "arrays": sorted(saved)}
        record["cost"] = {**cost, "peak_working_set_mb": peak_working_set_mb(),
                          "peak_working_set_note": "the whole process's, cumulative",
                          "wall_clock_total_s": time.perf_counter() - t_start}
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    def checkpoint(stage: str) -> None:
        """Stop here, with the record written, if anything so far has failed."""
        checkpoints.append({"stage": stage, "failures": list(failures)})
        if failures:
            record["stopped"] = f"{stage}: checks failed; nothing after it was built or written"
            finish()
            sys.exit(f"CHECKS FAILED at {stage}; nothing after it was built or written: {failures}")
        print(f"checkpoint passed: {stage}", flush=True)

    # -- the inputs ---------------------------------------------------------------------------
    n = legacy.num_design_var
    require(fs.digest(legacy.state_array) == cp_rec["state_array_sha256"],
            "the state array in R1t's fields is not the one its record describes")
    require(legacy.kktnorm == cp_rec["kktnorm"], "the KKT residual is not R1t's recorded one")
    require(legacy.updates_done == cp_rec["updates_done"] == t_rec["stop"]["mma_updates"]
            == R1T_UPDATES, "the checkpoint is not after R1t's twenty updates")
    require(n == t_rec["identities"]["num_design"], "the checkpoint's variables are not R1t's")
    require(np.array_equal(legacy.state_array[:n], x_terminal),
            "the checkpoint's design is not R1t's saved terminal design")
    require(old == cp_rec["binding"], "the binding in R1t's fields is not the one its record has")
    require(sorted(old["entry"]) == sorted(ff.LEGACY_ENTRY_KEYS),
            f"R1t's binding is not a legacy one: its entry names {sorted(old['entry'])}")
    require(old["entry"]["model"] == t_rec["identities"]["model_identity"],
            "R1t's binding does not name the model its record identifies")
    require(old["entry"]["thermal_path"] == t_rec["contract"]["model"]["thermal_path"] == THERMAL_PATH,
            "R1t's binding does not name the linear thermal path")
    require(old["budget"] == old["mma"]["max_iter"] == len(old["schedule"]) == R1T_UPDATES,
            "R1t's binding is not a budget of twenty over a schedule of twenty")
    require(not any(c["failures"] for c in t_rec["checkpoints"]), "R1t's record has failed checks")
    record["legacy"] = {"file": "results/zhao2d_r1t_fields.npz", "keys": "mma_*",
                        "sha256": inputs_before["zhao2d_r1t_fields.npz"],
                        "binding_sha256": hashlib.sha256(legacy.binding.encode()).hexdigest(),
                        "entry_keys": sorted(old["entry"]),
                        "state_array_sha256": fs.digest(legacy.state_array),
                        "updates_done": legacy.updates_done, **flags(legacy),
                        "budget": old["budget"], "schedule_steps": len(old["schedule"])}
    checkpoint("the inputs")

    # -- the locked source ---------------------------------------------------------------------
    recorded = t_rec["provenance"]["source_sha256"]
    locked = {}
    for name in LOCKED_SOURCE:
        mode = locked_match(REPO / name, recorded[name]) if name in recorded else None
        locked[name] = {"recorded_sha256": recorded.get(name),
                        "sha256_now": sha256_file(REPO / name), "matches": mode}
        require(mode is not None, f"{name} is not the source R1t was run with")
        if name.startswith("tfopus/"):
            module = importlib.import_module(name[:-3].replace("/", "."))
            require(pathlib.Path(module.__file__).resolve() == (REPO / name).resolve(),
                    f"the module imported for {name} is not that file")
    r1t_script = (REPO / "scripts" / "zhao2d_r1t_d_continue.py").read_text(encoding="utf-8")
    for line in R1T_BUILDS:
        require(line in r1t_script, f"R1t's script does not build `{line}`")
    record["locked_source"] = {"files": locked, "r1t_script_builds": list(R1T_BUILDS),
                               "git_head_at_r1t": t_rec["provenance"]["git_head"],
                               "rule": ("bytes, or for these named text files the same text up "
                                        "to line endings; nothing looser")}
    checkpoint("the locked source")

    # -- the configuration and the spec ------------------------------------------------------------
    fingerprint = t_rec["contract"]["fingerprint"]
    try:
        config = config_from_fingerprint(fingerprint)
    except ValueError as exc:
        require(False, f"the configuration: {exc}")
        checkpoint("the configuration and the spec")
    require(config == r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING),
            "R1t's recorded configuration is not the one its script builds")
    spec = z.Zhao2DSpec()  # as R1t's script built it, from the locked tfopus/zhao2d.py
    identity_spec = json.loads(r1.reference_identity(spec, config))["spec"]
    require(identity_spec == t_rec["identities"]["model_identity"]["spec"],
            "the spec's fields in the model identity are not R1t's recorded ones")
    require(json.loads(ff.fine_flow_identity(spec, config, FLOW_REFINEMENT, THERMAL_REFINEMENT,
                                             QUADRATURE)) == t_rec["identities"]["model_identity"],
            "the model identity rebuilt from the spec and the configuration is not R1t's")
    record["rebuilt"] = {
        "config": {"from": "results/zhao2d_r1t.json: contract.fingerprint, field by field",
                   "gives_the_fingerprint_back": True,
                   "equals_what_r1t_script_builds": R1T_BUILDS[1]},
        "spec": {"from": f"{R1T_BUILDS[0]}, as R1t's script built it, from the locked "
                         "tfopus/zhao2d.py",
                 "fields_checked_against_r1t_model_identity": sorted(identity_spec),
                 "fields_from_the_locked_source": {k: v for k, v in dataclasses.asdict(spec).items()
                                                   if k not in identity_spec}},
    }
    checkpoint("the configuration and the spec")

    # -- the design side ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    design = r1.Zhao2DProblem(spec, config)  # the constructor D runs for its design side, on h
    cost["design_side_build_s"] = time.perf_counter() - t0
    require(mesh_identity(design.flow_mesh) == t_rec["identities"]["design_mesh"],
            "the design mesh rebuilt is not R1t's")
    require(design.num_design == n, f"the design side has {design.num_design} variables, not {n}")
    block = ff.optimisation_identity(design)
    record["design_side"] = {"built": "tfopus.zhao2d_r1.Zhao2DProblem(spec, config), on h",
                             "design_mesh": mesh_identity(design.flow_mesh),
                             "num_design": design.num_design, "optimisation_block": block}
    checkpoint("the design side")

    # -- the migration -----------------------------------------------------------------------------
    try:
        migrated = ff.migrate_legacy_checkpoint(legacy, design, fingerprint)
    except ValueError as exc:
        require(False, f"the migration was refused: {exc}")
        checkpoint("the migration")
    new = json.loads(migrated.binding)
    require(np.array_equal(migrated.state_array, legacy.state_array)
            and fs.digest(migrated.state_array) == cp_rec["state_array_sha256"],
            "the migration changed the state array")
    require((migrated.kktnorm, migrated.updates_done, migrated.num_design_var)
            == (legacy.kktnorm, legacy.updates_done, legacy.num_design_var),
            "the migration changed the KKT residual, the updates or the variables")
    require({k: v for k, v in new.items() if k != "entry"} == {k: v for k, v in old.items() if k != "entry"},
            "the migration changed more of the binding than its entry")
    require({k: v for k, v in new["entry"].items() if k != "optimisation"} == old["entry"],
            "the migration changed the entry's model, scale or thermal path")
    require(new["entry"]["optimisation"] == json.loads(json.dumps(block)),
            "the entry's optimisation block is not the one rebuilt")
    record["migration"] = {
        "version": MIGRATION_VERSION,
        "function": "tfopus.zhao2d_fineflow.migrate_legacy_checkpoint",
        "from": {"file": "results/zhao2d_r1t_fields.npz", "keys": "mma_*",
                 "sha256": inputs_before["zhao2d_r1t_fields.npz"],
                 "binding_sha256": record["legacy"]["binding_sha256"]},
        "to": {"file": FIELDS, "keys": "mma_*",
               "binding_sha256": hashlib.sha256(migrated.binding.encode()).hexdigest()},
        "changed": ["the binding's entry gains `optimisation`"],
        "unchanged": {"state_array_sha256": fs.digest(migrated.state_array),
                      "kktnorm": migrated.kktnorm, "updates_done": migrated.updates_done,
                      "num_design_var": migrated.num_design_var,
                      "binding": ["entry.model", "entry.scale", "entry.thermal_path", "mma",
                                  "schedule", "budget"]},
        "binding": new,
    }
    saved.update(**migrated.to_npz(), legacy_binding=np.str_(legacy.binding),
                 migration_version=np.str_(MIGRATION_VERSION),
                 migrated_from=np.str_("results/zhao2d_r1t_fields.npz"),
                 migrated_from_sha256=np.str_(inputs_before["zhao2d_r1t_fields.npz"]))
    checkpoint("the migration")

    # -- dry, for part 2: nothing is run or saved for it ----------------------------------------
    step = old["schedule"][0]
    require(all(s == step for s in old["schedule"]) and step["beta"] == BETA
            and step["alpha_max"] == ALPHA_MAX, "R1t's schedule is not one phase at beta 32, 1e7")
    phases = [drv.Phase(step["phase"], R1T_UPDATES, beta=BETA, alpha_max=ALPHA_MAX),
              drv.Phase(APPENDED_PHASE, APPENDED, beta=BETA, alpha_max=ALPHA_MAX)]
    try:
        appended = migrated.append_budget(phases, R1T_UPDATES + APPENDED)
    except ValueError as exc:
        require(False, f"appending twenty updates was refused: {exc}")
        checkpoint("dry, for part 2")
    grown = json.loads(appended.binding)
    changed = np.flatnonzero(appended.state_array != migrated.state_array)
    require(changed.size == 1 and flags(migrated)["is_converged"]
            and not flags(appended)["is_converged"],
            "appending changed more of the state than the budget's flag")
    require(grown["budget"] == grown["mma"]["max_iter"] == R1T_UPDATES + APPENDED
            and grown["schedule"][:R1T_UPDATES] == old["schedule"]
            and len(grown["schedule"]) == R1T_UPDATES + APPENDED,
            "appending did not add twenty steps and budget after R1t's twenty")
    require(grown["entry"] == new["entry"]
            and {k: v for k, v in grown["mma"].items() if k != "max_iter"}
            == {k: v for k, v in new["mma"].items() if k != "max_iter"},
            "appending changed the entry or an MMA parameter other than max_iter")

    stand_in = SimpleNamespace(
        spec=spec, config=config, design_elements=design.design_elements,
        num_design=design.num_design, thermal_path=THERMAL_PATH,
        model_identity=lambda: ff.fine_flow_identity(spec, config, FLOW_REFINEMENT,
                                                     THERMAL_REFINEMENT, QUADRATURE))
    ref_path = dual.reference_file(THERMAL_REFINEMENT, QUADRATURE)
    scale = ff.common_scale(stand_in, ref_path, THERMAL_REFINEMENT, QUADRATURE)
    require(scale.source_sha256 == old["entry"]["scale"]["source_sha256"],
            f"{ref_path.relative_to(REPO).as_posix()}'s bytes are not the ones R1t's binding names "
            "(on an LF checkout, restore its recorded CRLF bytes first)")
    first = {}

    def stop_before_solving(x, alpha_max, beta, gradient=True):
        first.update(design=np.array(x), alpha_max=float(alpha_max), beta=float(beta))
        raise StopBeforeSolving

    accepted = False
    try:
        drv.run_loop(stand_in, stop_before_solving, phases, MOVE_LIMIT, R1T_UPDATES + APPENDED,
                     binding=ff.run_binding(stand_in, scale), resume=appended)
        require(False, "the resumed loop returned without asking for an evaluation")
    except StopBeforeSolving:
        accepted = True
    except ValueError as exc:
        require(False, f"the driver refused the appended checkpoint: {exc}")
    require(accepted and np.array_equal(first.get("design"), x_terminal),
            "the first design a resumed run would evaluate is not R1t's raw terminal design")
    require(accepted and (first["alpha_max"], first["beta"]) == (ALPHA_MAX, BETA),
            "the first resumed evaluation would not be at beta 32, alpha_max 1e7")
    record["dry_part_2"] = {
        "note": ("part 2 is not authorised: this checks that its start would be accepted, and "
                 "saves nothing. Part 2's script appends again at its start, and its run checks "
                 "the binding on the real D before anything is solved."),
        "phases": [{"name": p.name, "iterations": p.iterations, "beta": p.beta,
                    "alpha_max": p.alpha_max} for p in phases],
        "budget": R1T_UPDATES + APPENDED,
        "append": {"binding_sha256": hashlib.sha256(appended.binding.encode()).hexdigest(),
                   "state_entries_changed": changed.tolist(),
                   "is_converged": [flags(migrated)["is_converged"],
                                    flags(appended)["is_converged"]],
                   "kept": ["x", "x_old_1", "x_old_2", "low", "upp", "epoch", "kktnorm",
                            "change_design_var", "kkt_norm"]},
        "resume": {"stand_in": ("D's model identity through fine_flow_identity, its design side "
                                "from Zhao2DProblem on h, the scale from the reference file; "
                                "D's fine meshes are not built"),
                   "reference_file": ref_path.relative_to(REPO).as_posix(),
                   "accepted_by_run_loop": accepted,
                   "first_evaluation": {"iteration": R1T_UPDATES, "phase": APPENDED_PHASE,
                                        "alpha_max": first.get("alpha_max"),
                                        "beta": first.get("beta"),
                                        "design_is_r1t_raw_terminal": bool(
                                            np.array_equal(first.get("design"), x_terminal)),
                                        "solved": False}},
    }
    checkpoint("dry, for part 2")

    # -- R1t's files ---------------------------------------------------------------------------------
    inputs_after = {n_: sha256_file(inputs / n_) for n_ in INPUTS}
    require(inputs_after == inputs_before, "R1t's files changed during the migration")
    record["inputs_unchanged"] = {"before": inputs_before, "after": inputs_after}
    checkpoint("R1t's files")
    finish()

    print(f"migrated: state {record['migration']['unchanged']['state_array_sha256'][:12]}.. "
          f"unchanged, binding {record['legacy']['binding_sha256'][:12]}.. -> "
          f"{record['migration']['to']['binding_sha256'][:12]}..; dry append to "
          f"{R1T_UPDATES + APPENDED} accepted, first evaluation at R1t's raw terminal")
    print(f"wrote {out / RECORD} and {out / FIELDS}; total "
          f"{record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
