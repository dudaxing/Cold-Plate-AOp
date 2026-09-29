"""R1w: the binary designs along R1v's saved trajectory, each evaluated once on D. No MMA.

The contract proposed in the review of 03ccf6c, its plan accepted in the
review of 45fa8b0, authorised by the user on the local CPU. R1v saved its 21
raw designs, iterates 20 to 40, but exported and evaluated only the first
(R1t's) and the last (R1v's). This exports all 21 by R1l's rule, unchanged,
deduplicates them by the binary design itself, and reuses the D states already
evaluated. Each of the others is solved once on D: one flow solve and one
linear thermal solve on R1m's check-layer route. The whole pool is then
ranked once, and the script stops. It answers one question: under the
original rule, did choosing the terminal miss a better binary design that the
segment already produced?

Checked in stages. A failed stage writes the record and exits non-zero before
the next build or solve:

* inputs: R1v's record and saved designs, and the six earlier candidates'
  records, binary designs and D values, are the recorded ones; the scale is
  R1q's;
* the pool, geometry only. Each design is mapped by the frozen filter and
  projection at beta = 32 and exported by R1l's rule. R1t's and R1v's designs
  map to their saved densities and export to their saved binary designs.
  Before any solve, the manifest -- each iterate, its binary design's digest,
  and whether that design is reused or solved -- is written to a file;
* the check-layer route: R1m's meshes;
* the solves, one design at a time. The copy to h/2 must be a pure
  refinement and the h/8 material the parents'; then one flow solve and one
  thermal solve, each gated. A design whose solve fails -- an error, or a
  state that misses the gate -- is recorded as having no valid performance,
  and the next one is solved. There is no retry, and no gate is changed;
* the ranking, once: J at w = 0.5 on D at the common scale. If any design
  failed, the result is the best of those that passed; it does not rule out
  a better one among the others.

Not done: no MMA update, no AD, no other stage's designs, no change of beta,
q, w, the scale, the meshes or the gates, no cell filled, no volume made up.

    python scripts/zhao2d_r1w_pool_screen.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import faulthandler
import gc
import json
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

import toflux.src.solver as tf_solver  # noqa: E402

from tfopus import affine_solve as affine  # noqa: E402
from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402
from tfopus import zhao2d_refine as ref  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
# R1m's report, so each design on D is summarised exactly as the candidates were
from zhao2d_r1m_flow_check import flat, mesh_identity, thermal_identity  # noqa: E402
# R1v's components by shared edges and its weight envelope, unchanged
import zhao2d_r1v_d_append as rv  # noqa: E402

ALPHA_MAX, BETA = 1.0e7, 32.0
FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE = 2, 4, 3
THERMAL_PATH = "linear"
GATE = 1e-8
POOL_ITERATIONS = tuple(range(20, 41))  # R1v's saved designs, rows 0..20
MAX_SOLVES = 16
RECORD = "zhao2d_r1w.json"
FIELDS = "zhao2d_r1w_fields.npz"
MANIFEST = "zhao2d_r1w_manifest.json"
PROGRESS = "zhao2d_r1w_progress.json"
STACKS = "zhao2d_r1w_stacks.log"
LOCK = "zhao2d_r1w.lock"
INPUTS = ("zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_vp_pilot_fields.npz",
          "zhao2d_r1m_flow_check.json", "zhao2d_r1n_beta32.json", "zhao2d_r1n_fields.npz",
          "zhao2d_r1o.json", "zhao2d_r1o_fields.npz", "zhao2d_r1q_fineflow_check.json",
          "zhao2d_r1r.json", "zhao2d_r1r_fields.npz", "zhao2d_r1t.json", "zhao2d_r1t_fields.npz",
          "zhao2d_r1u.json", "zhao2d_r1u_fields.npz", "zhao2d_r1v.json", "zhao2d_r1v_fields.npz")
NAMES = ("r1t", "r1v", "r1u", "r1r", "r1o", "r1n", "pilot")  # evaluated on D before R1w
COMPARED = ("J", "J_star", "psi", "compliance", "T_max", "D_T_over_Q")


class SolveFailed(RuntimeError):
    """A design whose route check, solve or gate failed: it has no valid performance."""

    def __init__(self, message: str, partial: dict | None = None):
        super().__init__(message)
        self.partial = partial or {}


def rel(b, a) -> float:
    return b / a - 1.0


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def plan_pool(iterations, digests, known: dict, unqualified=()) -> list[dict]:
    """The pool's distinct binary designs, and what is done with each.

    Iterates are grouped by their binary design's digest; a group is labelled
    by its earliest iterate, but its identity is the digest. A digest in
    `known` (digest -> candidate) is reused; one in `unqualified` is not a
    candidate and is not solved; every other one is solved, once.
    """
    groups: dict[str, list[int]] = {}
    for it, d in zip(iterations, digests):
        groups.setdefault(d, []).append(int(it))
    out = []
    for d, its in sorted(groups.items(), key=lambda kv: min(kv[1])):
        if d in unqualified:  # not a candidate, whatever else it matches
            action, name = "not qualified", None
        else:
            name = known.get(d)
            action = "reuse" if name else "solve"
        out.append({"label": min(its), "iterations": sorted(its), "sha256": d,
                    "action": action, "reuse_of": name})
    return out


def screen(groups: list[dict], solve_one, on_result=None) -> list[dict]:
    """Solve each group once, in order. A failure is recorded, never retried, and the next is solved.

    `solve_one(group)` returns the design's D cell or raises. Whatever it
    raises means no valid performance was obtained for that design: it is not
    read as a worse design, or as an infeasible one.
    """
    results = []
    for g in groups:
        try:
            result = {"label": g["label"], "valid": True, "cell": solve_one(g)}
        except Exception as exc:  # noqa: BLE001 -- recorded; the design has no valid performance
            result = {"label": g["label"], "valid": False,
                      "failure": f"{type(exc).__name__}: {exc}",
                      "partial": getattr(exc, "partial", {}),
                      "traceback": traceback.format_exc(limit=-8)}
        results.append(result)
        if on_result:
            try:
                on_result(result)
            except Exception as exc:  # noqa: BLE001 -- reporting must not stop the screen
                result["reporting_error"] = f"{type(exc).__name__}: {exc}"
    return results


def rank_pool(entries: list[dict]) -> dict:
    """Rank the valid designs by J, once.

    A design with no valid performance, or one the rule does not qualify, is
    listed apart and not ranked -- neither as a worse design nor at all.
    `entries` holds every distinct design of the pool.
    """
    valid = sorted((e for e in entries if e["valid"]), key=lambda e: e["J"])
    unqualified = [e["label"] for e in entries if e.get("not_qualified")]
    failed = [e["label"] for e in entries if not e["valid"] and not e.get("not_qualified")]
    n = len(entries)
    if not failed and not unqualified:
        scope = f"all {n} designs of the pool"
    elif failed:
        scope = (f"the {len(valid)} of {n} designs that passed; it does not rule out a better "
                 "one among the others")
    else:
        scope = f"the {len(valid)} qualified designs of the {n}"
    return {"order": [e["label"] for e in valid],
            "lowest": valid[0]["label"] if valid else None,
            "complete": not failed and not unqualified,
            "no_valid_performance": failed,
            "not_qualified": unqualified,
            "scope": scope}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    inputs, out = args.inputs, args.out
    existing = [n for n in (RECORD, FIELDS, MANIFEST) if (out / n).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{out} already holds {', '.join(existing)}, which is cited evidence; write "
                 "elsewhere with --out DIR, or pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK
    try:
        lock.open("x").close()  # one R1w run per output folder at a time
    except FileExistsError:
        sys.exit(f"{lock} exists: another R1w run is using {out}, or one was killed; delete it only "
                 "if no R1w process is running")
    if args.overwrite:  # every R1w file present afterwards belongs to this run
        for n in (RECORD, FIELDS, MANIFEST, PROGRESS, STACKS):
            (out / n).unlink(missing_ok=True)
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.enable(file=stacks, all_threads=True)
    faulthandler.dump_traceback_later(7200, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec = z.Zhao2DSpec()
    config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
    w = config.weight
    material = za.build_material(spec, ALPHA_MAX)
    single = r1.load_reference(spec, config)  # J* only
    p_rec = load_json(inputs / "zhao2d_r1l_vp_pilot.json")
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    n_rec = load_json(inputs / "zhao2d_r1n_beta32.json")
    o_rec = load_json(inputs / "zhao2d_r1o.json")
    q_rec = load_json(inputs / "zhao2d_r1q_fineflow_check.json")
    r_rec = load_json(inputs / "zhao2d_r1r.json")
    t_rec = load_json(inputs / "zhao2d_r1t.json")
    u_rec = load_json(inputs / "zhao2d_r1u.json")
    v_rec = load_json(inputs / "zhao2d_r1v.json")
    fp = np.load(inputs / "zhao2d_r1l_vp_pilot_fields.npz")
    fn = np.load(inputs / "zhao2d_r1n_fields.npz")
    fo = np.load(inputs / "zhao2d_r1o_fields.npz")
    fr = np.load(inputs / "zhao2d_r1r_fields.npz")
    ft = np.load(inputs / "zhao2d_r1t_fields.npz")
    fu = np.load(inputs / "zhao2d_r1u_fields.npz")
    fv = np.load(inputs / "zhao2d_r1v_fields.npz")
    designs = np.asarray(fv["designs"], dtype=np.float64)
    rows = m_rec["rows"]
    base = {"r1t": t_rec["cells"]["new/check"], "r1v": v_rec["cells"]["new/check"],
            "r1u": u_rec["cells"]["filled/check"], "r1r": r_rec["cells"]["new/check"],
            "r1o": o_rec["cells"]["new/check"], "r1n": n_rec["cells"]["new/check"],
            "pilot": m_rec["cells"]["pilot/flow_h2"]}
    binaries = {"r1t": ft["solid_fraction_binary"], "r1v": fv["solid_fraction_binary"],
                "r1u": fu["solid_fraction_filled"], "r1r": fr["solid_fraction_binary"],
                "r1o": fo["solid_fraction_binary"], "r1n": fn["solid_fraction_binary"],
                "pilot": fp["solid_fraction_binary"]}
    binary_sha = {"r1t": t_rec["export"]["binary_sha256"], "r1v": v_rec["export"]["binary_sha256"],
                  "r1u": u_rec["geometry"]["sha256"], "r1r": r_rec["export"]["binary_sha256"],
                  "r1o": o_rec["export"]["binary_sha256"], "r1n": n_rec["export"]["binary_sha256"],
                  "pilot": p_rec["export"]["binary_sha256"]}
    continuous_j = {h["iteration"]: h["J_common_scale"] for h in v_rec["history"]}
    continuous_j[v_rec["terminal"]["iteration"]] = v_rec["terminal"]["J_common_scale"]
    recorded_g = {h["iteration"]: h["constraint_g"] for h in v_rec["history"]}
    recorded_g[v_rec["terminal"]["iteration"]] = v_rec["terminal"]["constraint_g"]

    record = {
        "note": ("R1w: R1v's 21 saved raw designs (iterates 20 to 40) exported by R1l's rule, "
                 "deduplicated by the binary design itself; the designs already evaluated on D "
                 "(R1t's and R1v's) reused, each other one evaluated once on the check layer D "
                 "(flow h/2, thermal h/8; the flow on the Newton path, the temperature by one "
                 "linear solve), and the pool ranked once by J at w = 0.5 on D at the common "
                 "scale. No MMA, no AD. A design whose solve fails has no valid performance."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"pool": "R1v's saved raw designs, iterates 20 to 40",
                     "export": "the frozen configuration, beta 32, R1l's volume rule; no cell "
                               "filled, no re-threshold",
                     "dedup": "by the binary design's digest and the model identity",
                     "model": {"flow_refinement": FLOW_REFINEMENT,
                               "thermal_refinement": THERMAL_REFINEMENT,
                               "thermal_quadrature": QUADRATURE, "thermal_path": THERMAL_PATH},
                     "alpha_max": ALPHA_MAX, "beta": BETA, "gate": GATE, "weight": w,
                     "solves_at_most": {"flow": MAX_SOLVES, "thermal": MAX_SOLVES},
                     "mma_updates": 0, "ad": 0,
                     "failure": "no valid performance: not a worse design, not an infeasible "
                                "one; no retry, no gate changed",
                     "ranking": "J at w = 0.5 on D at the common scale, once, when the pool is "
                                "complete",
                     "authorised": ("by the user on 2026-09-30, on the local CPU; proposed in the "
                                    "review of 03ccf6c, its plan accepted in the review of 45fa8b0")},
    }
    for script in (pathlib.Path(__file__).resolve(), REPO / "scripts" / "zhao2d_r1m_flow_check.py",
                   REPO / "scripts" / "zhao2d_r1v_d_append.py"):
        record["provenance"]["source_sha256"][script.relative_to(REPO).as_posix()] = sha256_file(script)
    failures, checkpoints, cost, cells, saved, progress = [], [], {}, {}, {}, []

    def require(ok, what: str) -> bool:
        if not ok:
            failures.append(what)
        return bool(ok)

    finished = []

    def finish() -> None:
        if finished:  # written once; a later error only re-raises
            return
        finished.append(True)
        record.update(cells=cells, checkpoints=checkpoints)
        if saved:
            record["fields"] = {"file": FIELDS, "arrays": sorted(saved)}
        record["cost"] = {**cost, "peak_working_set_mb": peak_working_set_mb(),
                          "peak_working_set_note": "the whole process's, cumulative",
                          "wall_clock_total_s": time.perf_counter() - t_start}
        try:  # the record's text first, so a serialisation error never leaves fields without it
            text = json.dumps(record, indent=2, default=float)
        except TypeError:
            text = json.dumps(record, indent=2, default=repr)
        if saved:
            np.savez_compressed(out / FIELDS, **saved)
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        stacks.close()
        if (out / STACKS).stat().st_size == 0:
            (out / STACKS).unlink()
        (out / RECORD).write_text(text, encoding="utf-8")
        if not failures:
            (out / PROGRESS).unlink(missing_ok=True)
        lock.unlink(missing_ok=True)

    def checkpoint(stage: str) -> None:
        """Stop here, with the record written, if anything so far has failed."""
        checkpoints.append({"stage": stage, "failures": list(failures)})
        if failures:
            record["stopped"] = f"{stage}: checks failed; nothing after it was built or solved"
            finish()
            sys.exit(f"CHECKS FAILED at {stage}; nothing after it was built or solved: {failures}")
        print(f"checkpoint passed: {stage}", flush=True)

    try:
        # -- inputs ---------------------------------------------------------------------------
        for name, rec_ in (("R1m", m_rec), ("R1n", n_rec), ("R1o", o_rec), ("R1q", q_rec),
                           ("R1r", r_rec), ("R1t", t_rec), ("R1u", u_rec), ("R1v", v_rec)):
            require(not any(c["failures"] for c in rec_["checkpoints"]), f"{name}'s record has failed checks")
        # R1l's record predates checkpoints: its checks, and its export's own verdict
        require(not any(p_rec["checks"].values()) and p_rec["export"]["qualified"],
                "the R1l pilot's record has failed checks or an unqualified export")
        require(designs.shape == (len(POOL_ITERATIONS), 5000),
                f"R1v's saved designs have shape {designs.shape}, not (21, 5000)")
        require(sorted(continuous_j) == list(POOL_ITERATIONS),
                "R1v's record does not hold the continuous J of iterates 20 to 40")
        require(np.array_equal(designs[0], ft["design"]), "R1v's first saved design is not R1t's raw terminal")
        require(np.array_equal(designs[-1], fv["design"]), "R1v's last saved design is not its terminal")
        for name in NAMES:
            require(fs.digest(binaries[name]) == binary_sha[name],
                    f"{name}: the saved binary design is not the one its record has")
            require(zb.cell_usable(base[name]) and base[name]["volume_feasible"],
                    f"{name}: its D cell is not a usable, gated, feasible state in its record")
        for name, rec_ in (("R1t", t_rec), ("R1v", v_rec)):
            require(rec_["check_layer"]["flow_mesh"] == rows["flow_h2"]["flow_mesh"]
                    and rec_["check_layer"]["thermal_mesh"] == rows["flow_h2"]["thermal_mesh"],
                    f"{name}'s D state was not solved on R1m's check-layer route")
            require(rec_["cells"]["new/check"]["thermal_path"] == THERMAL_PATH,
                    f"{name}'s D state was not solved on the linear thermal path")
        ref_path = dual.reference_file(4, QUADRATURE)
        values = r1.ReferenceValues.from_json(ref_path.read_text(encoding="utf-8"))
        yard = {"psi_0": values.psi_0, "c_0": values.c_0, "file": ref_path.relative_to(REPO).as_posix(),
                "file_sha256": sha256_file(ref_path), "declared": "the common scale"}
        require(all(yard[k] == q_rec["scale"][k] for k in ("psi_0", "c_0"))
                and yard["file_sha256"] == q_rec["scale"]["source_sha256"],
                "the development reference is not the common scale R1q to R1v used")

        def common_j(psi: float, c: float) -> float:
            return w * psi / yard["psi_0"] + (1 - w) * c / yard["c_0"]

        for name in NAMES:
            require(base[name]["J"] == common_j(base[name]["psi"], base[name]["compliance"]),
                    f"{name}: its recorded D value of J is not on this scale")
        record["scale"] = yard
        checkpoint("inputs")

        # -- the pool, geometry only ----------------------------------------------------------------
        t0 = time.perf_counter()
        design_side = r1.Zhao2DProblem(spec, config)  # the design side D runs, on h: nothing solved
        cost["design_side_build_s"] = time.perf_counter() - t0
        planar = design_side.flow_mesh
        design_mask = np.asarray(planar.design_mask)
        areas = np.asarray(planar.elem_area)
        pool, masks, densities, digests, qualifies = [], {}, {}, [], {}
        for k, it in enumerate(POOL_ITERATIONS):
            x = jnp.asarray(designs[k])
            s = np.asarray(design_side.solid_fraction(x, BETA))
            g = float(design_side.fluid_fraction(x, BETA)) / config.max_fluid_fraction - 1.0
            v_f = z.fluid_fractions(planar, jnp.asarray(s))["v_f_design_domain"]
            exp = zb.volume_threshold(s, design_mask, areas, config.max_fluid_fraction)
            s_bin = exp.pop("s_binary")
            conn = zb.connectivity(planar, s, (exp["t"],))[f"{exp['t']:g}"]
            comps = rv.component_summary(rv.fluid_components(planar, s_bin))
            d = fs.digest(s_bin)
            qualified = bool(zb.volume_feasible(exp["fluid_fraction"], config.max_fluid_fraction)
                             and conn["inlet_outlet_connected"])
            feasible = bool(zb.volume_feasible(v_f, config.max_fluid_fraction))
            # a binary design qualifies if any iterate exports it, qualified, from a feasible design
            qualifies[d] = qualifies.get(d, False) or (qualified and feasible)
            require(abs(g - recorded_g[it]) <= 1e-12,
                    f"iterate {it}: the constraint {g!r} is not R1v's recorded {recorded_g[it]!r}; the "
                    "saved rows do not line up with the iterates")
            pool.append({"iteration": it, "sha256": d, "t": exp["t"], "fluid_cells": exp["fluid_cells"],
                         "fluid_fraction": exp["fluid_fraction"],
                         "cells_changed_from_0.5": exp["cells_changed_from_0.5"],
                         "continuous_volume_feasible": feasible, "constraint_g": g,
                         "connected": bool(conn["inlet_outlet_connected"]), "qualified": qualified,
                         "components": comps["sizes"],
                         "isolated_cells": [c["cells"] for c in comps["isolated"]],
                         "continuous_J": continuous_j[it],
                         "cells_differing_from_r1t": int(np.sum(s_bin != np.asarray(binaries["r1t"])))})
            masks.setdefault(d, s_bin)
            densities[it] = s
            digests.append(d)
        record["pool"] = pool
        require(np.array_equal(densities[POOL_ITERATIONS[0]], ft["solid_fraction"]),
                "the map does not give R1t's saved density for its raw terminal")
        require(np.array_equal(densities[POOL_ITERATIONS[-1]], fv["solid_fraction"]),
                "the map does not give R1v's saved density for its raw terminal")
        require(digests[0] == binary_sha["r1t"], "iterate 20 does not export to R1t's binary design")
        require(digests[-1] == binary_sha["r1v"], "iterate 40 does not export to R1v's binary design")
        known = {binary_sha[n]: n for n in NAMES}
        unqualified = {d for d, ok in qualifies.items() if not ok}
        groups = plan_pool(POOL_ITERATIONS, digests, known, unqualified)
        to_solve = [g for g in groups if g["action"] == "solve"]
        require(len(to_solve) <= MAX_SOLVES,
                f"{len(to_solve)} designs to solve, more than the {MAX_SOLVES} the contract allows")
        manifest = {"note": ("written before any solve: each iterate's binary design, and whether it "
                             "is reused (a candidate's D state) or solved once; a label is the "
                             "group's earliest iterate, the identity its digest and the model"),
                    "model_identity": {"flow_mesh": rows["flow_h2"]["flow_mesh"],
                                       "thermal_mesh": rows["flow_h2"]["thermal_mesh"],
                                       "thermal_path": THERMAL_PATH, "alpha_max": ALPHA_MAX},
                    "iterates": [{"iteration": p["iteration"], "sha256": p["sha256"],
                                  **{k: next(g_ for g_ in groups if g_["sha256"] == p["sha256"])[k]
                                     for k in ("label", "action", "reuse_of")}} for p in pool],
                    "groups": groups,
                    "counts": {"iterates": len(pool), "distinct": len(groups),
                               "reuse": sum(g["action"] == "reuse" for g in groups),
                               "solve": len(to_solve),
                               "not_qualified": sum(g["action"] == "not qualified" for g in groups)}}
        record["manifest"] = manifest
        for g in groups:
            saved[f"binary_{g['label']}"] = masks[g["sha256"]]
        print(f"pool: {len(pool)} designs, {len(groups)} distinct binary designs "
              f"({manifest['counts']['reuse']} reused, {len(to_solve)} to solve, "
              f"{manifest['counts']['not_qualified']} not qualified); labels "
              f"{[g['label'] for g in groups]}", flush=True)
        if not failures:
            (out / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        checkpoint("the pool and the manifest")

        # -- the check-layer route -------------------------------------------------------------------
        t0 = time.perf_counter()
        fine = dual.Zhao2DDualProblem(ref.refine_spec(spec, FLOW_REFINEMENT), config,
                                      thermal_refinement=THERMAL_REFINEMENT,
                                      thermal_quadrature=QUADRATURE)
        cost["build_check_s"] = time.perf_counter() - t0
        print(f"built the check-layer route: flow {fine.flow_mesh.num_elems} el, thermal "
              f"{fine.thermal_mesh.num_elems} el ({cost['build_check_s']:.1f} s)", flush=True)
        record["check_layer"] = {"flow_mesh": mesh_identity(fine.flow_mesh),
                                 "thermal_mesh": thermal_identity(fine)}
        require(record["check_layer"]["flow_mesh"] == rows["flow_h2"]["flow_mesh"],
                "the h/2 flow mesh is not R1m's")
        require(record["check_layer"]["thermal_mesh"] == rows["flow_h2"]["thermal_mesh"],
                "the h/8 thermal side is not R1m's")
        parents_t = ref.parent_of_each_fine_element(planar, fine.thermal_mesh)
        # the route's own checks, once, on R1t's binary design: a route defect stops the run here
        s_f = np.asarray(ref.refine_design(planar, fine.flow_mesh, binaries["r1t"]))
        try:
            record["check_layer"]["transfer_r1t"] = ref.check_transfer(planar, fine.flow_mesh,
                                                                       binaries["r1t"], s_f)
        except RuntimeError as exc:
            require(False, f"the copy to h/2 is not a pure refinement: {exc}")
        require(np.array_equal(np.asarray(fine.maps.density(jnp.asarray(s_f))),
                               np.asarray(binaries["r1t"])[parents_t]),
                "the h/8 material is not the parents' material")
        # a reused D state must be this route's flow model at that design (no solve needed)
        record["check_layer"]["reused_identity"] = {}
        for g in groups:
            if g["action"] != "reuse":
                continue
            name = g["reuse_of"]
            s_n = np.asarray(ref.refine_design(planar, fine.flow_mesh, masks[g["sha256"]]))
            same = fs.flow_state_identity(fine, s_n, ALPHA_MAX) == base[name].get("flow_identity")
            record["check_layer"]["reused_identity"][name] = bool(same)
            require(same, f"{name}'s recorded D state is not this route's flow model at its design")
            require(base[name].get("thermal_path") == THERMAL_PATH,
                    f"{name}'s recorded D state is not on the linear thermal path")
        checkpoint("the check-layer route")

        # -- the solves, one design at a time ---------------------------------------------------------
        def solve_one(group: dict) -> dict:
            label, s_bin = group["label"], masks[group["sha256"]]
            partial = {"design": f"i{label}", "layer": "check", "thermal_path": THERMAL_PATH,
                       "binary_sha256": group["sha256"], "iterations": group["iterations"],
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            try:
                s_f = np.asarray(ref.refine_design(planar, fine.flow_mesh, s_bin))
                try:
                    partial["transfer"] = ref.check_transfer(planar, fine.flow_mesh, s_bin, s_f)
                except RuntimeError as exc:
                    raise SolveFailed(f"the copy to h/2 is not a pure refinement: {exc}", partial) from exc
                if not np.array_equal(np.asarray(fine.maps.density(jnp.asarray(s_f))),
                                      np.asarray(s_bin)[parents_t]):
                    raise SolveFailed("the h/8 material is not the parents' material", partial)
                s = jnp.asarray(s_f)
                fluid = z.fluid_fractions(fine.flow_mesh, s)
                partial.update(fluid_fractions=fluid, volume_feasible=zb.volume_feasible(
                    fluid["v_f_design_domain"], config.max_fluid_fraction))
                if not partial["volume_feasible"]:
                    raise SolveFailed(f"over the fluid bound on h/2: {fluid}", partial)
                alpha = materials.brinkman_penalty(s, material)
                pv, t_flow = timed(tf_solver.modified_newton_raphson_solve, fine.flow, fine.flow_x0, alpha)
                partial["t_flow_solve_s"] = t_flow
                try:
                    partial["flow_verification"] = fs.verify_flow_state(fine, pv, s, ALPHA_MAX, GATE)
                except (ValueError, r1.NotConverged) as exc:
                    raise SolveFailed(f"the flow solve fails its gate: {exc}", partial) from exc
                partial["flow_identity"] = fs.flow_state_identity(fine, s, ALPHA_MAX)
                temp, t_thermal = timed(
                    lambda: jax.block_until_ready(affine.affine_solve(
                        fine.thermal, fine.thermal_x0, fine.thermal_velocity(pv),
                        fine.thermal_conductivity(s, ALPHA_MAX), fine.q_source)))
                partial["t_thermal_solve_s"] = t_thermal
                t0_ = time.perf_counter()
                rep = fs.cell_report(fine, s, pv, temp, ALPHA_MAX, single)
                cell = {**partial, "state": "solved here", "analysed": True, **flat(rep, yard, w)}
                cell["gate_passed"] = zb.gate_passed(rep["residual_relative"], GATE)
                cell["t_report_s"] = time.perf_counter() - t0_
                cell["report"] = rep
                cell["state_sha256"] = {"s": fs.digest(np.asarray(s)),
                                        "press_vel": fs.digest(np.asarray(pv)),
                                        "temperature": fs.digest(np.asarray(temp))}
                cell["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                if not cell["gate_passed"]:
                    raise SolveFailed(f"a residual fails the gate: {rep['residual_relative']}", cell)
            except SolveFailed:
                raise
            except Exception as exc:  # keep what was recorded before it: which side failed, and when
                raise SolveFailed(f"{type(exc).__name__}: {exc}", partial) from exc
            saved.update({f"i{label}_solid_fraction_flow_h2": s_f,  # only states that passed the gate
                          f"i{label}_press_vel": np.asarray(pv),
                          f"i{label}_temperature": np.asarray(temp)})
            return cell

        def on_result(result: dict) -> None:
            label = result["label"]
            if result["valid"]:
                c = result["cell"]
                cells[f"i{label}"] = c
                line = (f"i{label}: Psi {c['psi']:.9e}  C {c['compliance']:.6f}  J {c['J']:.9f}  "
                        f"D_T/Q {c['D_T_over_Q']:.4%}  T_max {c['T_max']:.4f}  |R| "
                        f"{c['residual_flow']:.1e}/{c['residual_thermal']:.1e}  flow "
                        f"{c['t_flow_solve_s']:.1f} s, thermal {c['t_thermal_solve_s']:.1f} s, "
                        f"report {c['t_report_s']:.1f} s")
                entry = {k: c.get(k) for k in (*COMPARED, "residual_flow", "residual_thermal",
                                               "state_sha256", "t_flow_solve_s", "t_thermal_solve_s",
                                               "t_report_s", "started", "ended")}
            else:
                cells[f"i{label}"] = {"design": f"i{label}", "valid": False,
                                      "no_valid_performance": result["failure"],
                                      "partial": result["partial"], "traceback": result["traceback"]}
                line = f"i{label}: NO VALID PERFORMANCE ({result['failure']})"
                entry = {"failure": result["failure"]}
            progress.append({"label": label, "valid": result["valid"], **entry})
            (out / PROGRESS).write_text(json.dumps(progress, indent=1, default=float), encoding="utf-8")
            print(line, flush=True)
            gc.collect()

        t0 = time.perf_counter()
        results = screen(to_solve, solve_one, on_result)
        cost["t_solves_s"] = time.perf_counter() - t0

        def made(key: str) -> int:
            return sum(1 for r in results
                       if key in ((r["cell"] if r["valid"] else r["partial"]) or {}))

        cost["solves"] = {"planned": len(to_solve), "flow_made": made("t_flow_solve_s"),
                          "thermal_made": made("t_thermal_solve_s"), "mma": 0, "reverse_passes": 0}
        failed_labels = [r["label"] for r in results if not r["valid"]]
        checkpoints.append({"stage": f"the solves ({len(results) - len(failed_labels)} of "
                                     f"{len(results)} with valid performance)",
                            "failures": [], "no_valid_performance": failed_labels,
                            "reporting_errors": {str(r["label"]): r["reporting_error"] for r in results
                                                 if "reporting_error" in r}})

        # -- the ranking, once -------------------------------------------------------------------------
        entries = []
        for g in groups:
            if g["action"] == "not qualified":
                entries.append({"label": g["label"], "valid": False, "not_qualified": True, "J": None,
                                "cell": None, "source": "not qualified by the rule"})
            elif g["action"] == "reuse":
                c = base[g["reuse_of"]]
                entries.append({"label": g["label"], "valid": True, "J": c["J"], "cell": c,
                                "source": f"{g['reuse_of']}'s recorded D state"})
            else:
                r_ = next(r for r in results if r["label"] == g["label"])
                entries.append({"label": g["label"], "valid": r_["valid"],
                                "J": r_["cell"]["J"] if r_["valid"] else None,
                                "cell": r_.get("cell"), "source": "solved here"})
        ranking = rank_pool(entries)
        r1t = base["r1t"]
        table = []
        for e in entries:
            g = next(g_ for g_ in groups if g_["label"] == e["label"])
            pool_row = next(p_ for p_ in pool if p_["iteration"] == g["label"])
            row = {"label": e["label"], "iterations": g["iterations"], "sha256": g["sha256"],
                   "source": e["source"], "valid": e["valid"],
                   "components": pool_row["components"], "isolated_cells": pool_row["isolated_cells"],
                   "continuous_J": {str(i): continuous_j[i] for i in g["iterations"]}}
            if e["valid"]:
                c = e["cell"]
                row.update({k: c.get(k) for k in COMPARED})
                # the binary J over the continuous J of each iterate that exports this design
                row["export_gap_J"] = {str(i): rel(c["J"], continuous_j[i]) for i in g["iterations"]}
                row["against_r1t"] = {
                    "kind": zb.comparison_kind(c, r1t),
                    **{q: rel(c[q], r1t[q]) for q in ("J", "J_star", "psi", "compliance", "T_max",
                                                      "D_T_over_Q")},
                    "dJ": c["J"] - r1t["J"],
                    "terms": {"dissipation": w * (c["psi"] - r1t["psi"]) / yard["psi_0"],
                              "thermal_compliance": (1 - w) * (c["compliance"] - r1t["compliance"])
                              / yard["c_0"]}}
            table.append(row)
        lowest = next((row for row in table if row["label"] == ranking["lowest"]), None)
        points = {f"i{row['label']}": (row["psi"] / yard["psi_0"], row["compliance"] / yard["c_0"])
                  for row in table if row["valid"]}
        for name in ("r1u", "r1r", "r1o", "r1n", "pilot"):
            points[name] = (base[name]["psi"] / yard["psi_0"], base[name]["compliance"] / yard["c_0"])
        record["ranking"] = {
            **ranking, "measure": "J at w = 0.5 on D at the common scale",
            "lowest_in_pool": lowest, "table": table,
            "lowest_by_weight": {"note": "a fixed set of designs reweighted on the common scale: "
                                         "arithmetic, not a new optimisation or an extra weight sweep",
                                 "designs": sorted(points),
                                 "intervals": rv.lowest_by_weight(points)}}
        finish()

        print("\nranked once, by J at w = 0.5 on D (" + ranking["scope"] + "):")
        for label in ranking["order"]:
            row = next(r for r in table if r["label"] == label)
            a = row["against_r1t"]
            print(f"  i{label:<3d} {row['source']:<26s} J {row['J']:.9f}  vs R1t {a['J']:+.3%}"
                  f"  Psi {a['psi']:+.2%}  C {a['compliance']:+.2%}  T_max {a['T_max']:+.2%}")
        for label in ranking["no_valid_performance"]:
            print(f"  i{label:<3d} no valid performance")
        for label in ranking["not_qualified"]:
            print(f"  i{label:<3d} not qualified by the rule")
        print("lowest by weight: " + ", ".join(
            f"{i['lowest']} {i['from']:.4f}-{i['to']:.4f}"
            for i in record["ranking"]["lowest_by_weight"]["intervals"]))
        print(f"\nwrote {out / RECORD}, {out / FIELDS} and {out / MANIFEST}; total "
              f"{record['cost']['wall_clock_total_s']:.0f} s")
        if to_solve and len(failed_labels) == len(to_solve):
            sys.exit(f"NO SOLVED DESIGN HAS VALID PERFORMANCE ({len(failed_labels)} failed): the "
                     "ranking holds only the reused designs; look for a fault common to all")
        if failed_labels:
            print(f"NOTE: {len(failed_labels)} designs have no valid performance: {failed_labels}")
    except SystemExit:
        raise
    except BaseException as exc:  # write what was done, release the lock, then re-raise
        record["stopped"] = (f"interrupted: {type(exc).__name__}: {exc}; "
                             f"{sum(1 for c in cells.values() if c.get('analysed'))} designs "
                             "had been solved")
        failures.append(record["stopped"])
        finish()
        raise


if __name__ == "__main__":
    main()
