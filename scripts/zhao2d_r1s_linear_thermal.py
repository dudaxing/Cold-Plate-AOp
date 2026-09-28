"""R1s: the temperature by one linear solve, on the main mesh: equivalence and cost, no MMA.

The contract proposed in the review of 266ce00. The model is the check layer D
(`zhao2d_fineflow.Zhao2DFineFlowProblem`: design h, flow h/2, thermal h/8) with
`thermal_path="linear"`. For a given flow and density the temperature comes
from one linear solve (`tfopus.affine_solve`) instead of upstream's Newton
loop; the residual, the gate and the implicit derivative are unchanged.

Checked in stages. A failure writes the record and exits non-zero before the
next solve:

* inputs: R1q's main point (x, its saved states and gradients) and R1r's
  qualified binary design with its saved D states are the recorded ones, and
  R1q's saved main-point states are R1r's saved initial states, bit for bit;
* the model and the scale: R1m's check layer, 5000 variables, R1q's scale;
* A. R1q's main point: the temperature on its saved flow, by the linear path
  (1T), against the saved Newton temperature, its objectives and residuals;
* B. R1r's qualified binary design: the same on its saved flow (1T);
* C. one full D value and gradient at R1q's raw main design (1F + 1T and two
  reverse passes), against R1q's recorded values, its saved gradients and its
  recorded central differences.

The equivalence criteria were fixed before anything was solved on the main
mesh. On the small mesh the two paths agree to 1e-14.

* Every state passes the existing 1e-8 gate.
* A and B: against the Newton path's saved state, max|dT| / max|T| <= 1e-8 and
  |dC / C| <= 1e-8.
* C, against R1q's main point:
  - Psi to 1e-12 relative and g to 1e-12 absolute; C and J to 1e-8 relative;
  - each gradient (J, Psi, C, g) within 1e-6 of R1q's saved one in relative
    L2;
  - along R1q's two directions (seeds 11 and 12), the AD projections agree to
    1e-5 with R1q's recorded central differences at their best step, the
    existing criterion.

At most 1F + 3T on the main mesh, and no MMA. Costs are recorded as measured.
The Newton path's timings from R1q and R1r are not paired with these, so no
speed-up factor is claimed.

    python scripts/zhao2d_r1s_linear_thermal.py [--inputs results] [--out DIR] [--overwrite]
"""

from __future__ import annotations

import pathlib
import sys

# Default the BLAS thread count BEFORE numpy loads; see tfopus/_threads.py.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import tfopus._threads  # noqa: F401,E402

import argparse
import faulthandler
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "TOFLUX"))
sys.path.insert(0, str(REPO / "scripts"))

from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_fineflow as ff  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
from zhao2d_r1m_flow_check import mesh_identity, thermal_identity  # noqa: E402

ALPHA_MAX, BETA = 1.0e7, 32.0
FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE = 2, 4, 3
GATE, ANCHOR_TOL = 1e-8, 1e-12
STATE_TOL, VALUE_TOL, GRADIENT_TOL, FD_TOL = 1e-8, 1e-8, 1e-6, 1e-5
FD_ACTIVE_MARGIN = 1e-4  # R1q's: the variables at least its largest step from both bounds
RECORD = "zhao2d_r1s.json"
FIELDS = "zhao2d_r1s_fields.npz"
STACKS = "zhao2d_r1s_stacks.log"
INPUTS = ("zhao2d_r1m_flow_check.json", "zhao2d_r1q_fineflow_check.json", "zhao2d_r1q_fields.npz",
          "zhao2d_r1r.json", "zhao2d_r1r_fields.npz")
QUANTITIES = {"psi": "psi", "c": "compliance", "g": "constraint_g", "J": "J_common_scale"}


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_fields(path: pathlib.Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def rel(b, a) -> float:
    return b / a - 1.0


def state_difference(new, old) -> dict:
    new, old = np.asarray(new), np.asarray(old)
    d = new - old
    return {"max_abs": float(np.abs(d).max()),
            "max_abs_over_max_abs": float(np.abs(d).max() / np.abs(old).max()),
            "rms": float(np.sqrt(np.mean(d * d))),
            "bitwise_equal": bool(np.array_equal(new, old))}


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
    stacks = open(out / STACKS, "w", encoding="utf-8")
    faulthandler.dump_traceback_later(3600, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec = z.Zhao2DSpec()
    config = r1.R1Config(projection=r1.Projection.VOLUME_PRESERVING)
    w = config.weight
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    q_rec = load_json(inputs / "zhao2d_r1q_fineflow_check.json")
    r_rec = load_json(inputs / "zhao2d_r1r.json")
    fq = load_fields(inputs / "zhao2d_r1q_fields.npz")
    fr = load_fields(inputs / "zhao2d_r1r_fields.npz")
    main_eval = q_rec["main_point"]["evaluation"]
    binary_cell = r_rec["cells"]["new/check"]
    rows = m_rec["rows"]

    record = {
        "note": ("R1s: the check layer D (design h, flow h/2, thermal h/8) with the temperature "
                 "by one linear solve for a given flow and density, against upstream's Newton "
                 "path: two temperatures on saved flows (R1q's main point, R1r's qualified binary "
                 "design) and one full value and gradient at R1q's raw main design. No MMA. "
                 "Costs as measured, not paired with the Newton path's."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "contract": {"model": {"design": "h, 5000 raw variables", "flow_refinement": FLOW_REFINEMENT,
                               "thermal_refinement": THERMAL_REFINEMENT,
                               "thermal_quadrature": QUADRATURE, "thermal_path": "linear"},
                     "alpha_max": ALPHA_MAX, "beta": BETA, "gate": GATE,
                     "main_mesh_solves_at_most": {"flow": 1, "thermal": 3}, "mma_updates": 0,
                     "criteria": {"state_max_abs_over_max_abs": STATE_TOL, "C_relative": STATE_TOL,
                                  "main_point_psi_relative": ANCHOR_TOL,
                                  "main_point_g_absolute": ANCHOR_TOL,
                                  "main_point_C_and_J_relative": VALUE_TOL,
                                  "gradient_relative_l2": GRADIENT_TOL,
                                  "directional_vs_recorded_fd_best_step": FD_TOL,
                                  "fixed_before_the_main_mesh": True},
                     "authorised": "by the user, on the local CPU, as the review of 266ce00 states it"},
    }
    record["provenance"]["source_sha256"]["scripts/" + pathlib.Path(__file__).name] = sha256_file(
        pathlib.Path(__file__).resolve())
    record["provenance"]["source_sha256"]["scripts/zhao2d_r1m_flow_check.py"] = sha256_file(
        REPO / "scripts" / "zhao2d_r1m_flow_check.py")
    failures, checkpoints, cost, saved = [], [], {"peak_working_set_mb": {}}, {}

    def require(ok, what: str) -> bool:
        if not ok:
            failures.append(what)
        return bool(ok)

    def finish() -> None:
        record["checkpoints"] = checkpoints
        if saved:
            np.savez_compressed(out / FIELDS, **saved)
            record["fields"] = {"file": FIELDS, "arrays": sorted(saved)}
        record["cost"] = {**cost, "peak_working_set_mb_final": peak_working_set_mb(),
                          "peak_working_set_note": "the whole process's, cumulative",
                          "wall_clock_total_s": time.perf_counter() - t_start}
        faulthandler.cancel_dump_traceback_later()
        stacks.close()
        if (out / STACKS).stat().st_size == 0:
            (out / STACKS).unlink()
        (out / RECORD).write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")

    def checkpoint(stage: str) -> None:
        """Stop here, with the record written, if anything so far has failed."""
        checkpoints.append({"stage": stage, "failures": list(failures)})
        cost["peak_working_set_mb"][stage] = peak_working_set_mb()
        if failures:
            record["stopped"] = f"{stage}: checks failed; nothing after it was solved"
            finish()
            sys.exit(f"CHECKS FAILED at {stage}; nothing after it was solved: {failures}")
        print(f"checkpoint passed: {stage}", flush=True)

    def fixed_flow_temperature(label: str, s, press_vel, t_saved, c_recorded: float) -> dict:
        """One linear-path temperature on a saved flow, against the saved Newton one."""
        s, pv = jnp.asarray(s), jnp.asarray(press_vel)
        out_ = {}
        try:
            out_["flow_verification"] = fs.verify_flow_state(problem, pv, problem.flow_density(s),
                                                             ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{label}: the saved flow does not verify: {exc}")
            return out_
        alpha = problem.flow_material(s, ALPHA_MAX)
        kappa = problem.thermal_conductivity(s, ALPHA_MAX)
        vel = problem.thermal_velocity(pv)
        t0 = time.perf_counter()
        t_new = jax.block_until_ready(problem.solve_thermal(pv, s, ALPHA_MAX))
        out_["t_thermal_solve_s"] = time.perf_counter() - t0
        t_old = jnp.asarray(t_saved)
        psi = float(problem.flow.dissipated_power(pv, alpha))
        c_new = float(problem.thermal.thermal_compliance(t_new, vel, kappa))
        c_old = float(problem.thermal.thermal_compliance(t_old, vel, kappa))
        out_.update(
            psi=psi, compliance_linear=c_new, compliance_newton_saved=c_old,
            compliance_recorded=c_recorded,
            J_linear=w * psi / scale.psi_0 + (1 - w) * c_new / scale.c_0,
            residual_linear=problem.residual_norms_at(pv, t_new, alpha, kappa),
            residual_newton_saved=problem.residual_norms_at(pv, t_old, alpha, kappa),
            temperature_difference=state_difference(t_new, t_old),
            compliance_relative=rel(c_new, c_old),
            saved_state_reproduces_its_record=rel(c_old, c_recorded),
            t_max_linear=float(jnp.max(t_new)), t_min_linear=float(jnp.min(t_new)))
        saved[f"{label}_temperature_linear"] = np.asarray(t_new)
        require(abs(out_["saved_state_reproduces_its_record"]) <= ANCHOR_TOL,
                f"{label}: the saved Newton state does not reproduce its record's C")
        require(zb.gate_passed(out_["residual_linear"], GATE),
                f"{label}: the linear state fails the gate {out_['residual_linear']}")
        require(out_["temperature_difference"]["max_abs_over_max_abs"] <= STATE_TOL,
                f"{label}: the temperature differs from the Newton path's by "
                f"{out_['temperature_difference']['max_abs_over_max_abs']:.3e}")
        require(abs(out_["compliance_relative"]) <= STATE_TOL,
                f"{label}: C differs from the Newton path's by {out_['compliance_relative']:.3e}")
        print(f"{label}: C {c_new:.10f} (Newton {c_old:.10f}, rel {out_['compliance_relative']:+.2e})  "
              f"max|dT|/max|T| {out_['temperature_difference']['max_abs_over_max_abs']:.2e}  |R_T| "
              f"linear {out_['residual_linear']['thermal']:.2e}, Newton "
              f"{out_['residual_newton_saved']['thermal']:.2e}  ({out_['t_thermal_solve_s']:.1f} s)",
              flush=True)
        return out_

    # -- inputs ---------------------------------------------------------------------------
    x = np.asarray(fq["main_point_x"], dtype=np.float64)
    require(fs.digest(x) == q_rec["main_point"]["sha256_float64"], "R1q's saved x is not its main point")
    for key, digest in q_rec["main_point"]["gradient_sha256"].items():
        require(fs.digest(fq[f"gradient_{key}"]) == digest, f"R1q's saved gradient {key} is not the recorded one")
    for mine, theirs in (("main_point_x", "initial_design"), ("main_point_s", "initial_solid_fraction"),
                         ("main_point_press_vel", "initial_press_vel"),
                         ("main_point_temperature", "initial_temperature")):
        require(np.array_equal(fq[mine], fr[theirs]),
                f"R1q's saved {mine} is not R1r's saved {theirs}")
    require(fs.digest(fr["solid_fraction_binary"]) == r_rec["export"]["binary_sha256"],
            "R1r's saved binary design is not the one it exported")
    sha = binary_cell["state_sha256"]
    for mine, theirs in (("new_solid_fraction_flow_h2", "s"), ("new_press_vel_check", "press_vel"),
                         ("new_temperature_check", "temperature")):
        require(fs.digest(fr[mine]) == sha[theirs], f"R1r's saved {mine} is not the recorded one")
    for name, rec_ in (("R1q", q_rec), ("R1r", r_rec)):
        require(not any(c["failures"] for c in rec_["checkpoints"]), f"{name}'s record has failed checks")
    require(q_rec["verdict"]["all_pass"], "R1q's gradient check did not pass")
    checkpoint("inputs")

    # -- the model and the scale -------------------------------------------------------------
    t0 = time.perf_counter()
    problem = ff.Zhao2DFineFlowProblem(spec, config, FLOW_REFINEMENT, THERMAL_REFINEMENT, QUADRATURE,
                                       thermal_path="linear")
    cost["build_s"] = time.perf_counter() - t0
    print(f"built D, thermal path {problem.thermal_path}: design {problem.design_mesh.num_elems} el "
          f"({problem.num_design} variables), flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({cost['build_s']:.1f} s)", flush=True)
    record["identities"] = {"flow_mesh": mesh_identity(problem.flow_mesh),
                            "thermal_mesh": thermal_identity(problem),
                            "design_mesh": mesh_identity(problem.design_mesh),
                            "num_design": problem.num_design, "thermal_path": problem.thermal_path,
                            "model_identity": json.loads(problem.model_identity())}
    require(record["identities"]["flow_mesh"] == rows["flow_h2"]["flow_mesh"],
            "the flow side is not R1m's check layer")
    require(record["identities"]["thermal_mesh"] == rows["flow_h2"]["thermal_mesh"],
            "the thermal side is not R1m's check layer")
    require(record["identities"]["design_mesh"] == rows["flow_h"]["flow_mesh"],
            "the design mesh is not the development model's h mesh")
    require(problem.num_design == 5000, f"{problem.num_design} design variables, not 5000")
    try:
        scale = ff.common_scale(problem, dual.reference_file(4, QUADRATURE), 4, QUADRATURE)
    except ValueError as exc:
        require(False, f"the common scale: {exc}")
        checkpoint("the model and the common scale")
    record["scale"] = {"psi_0": scale.psi_0, "c_0": scale.c_0, "source_sha256": scale.source_sha256}
    require(all(v == q_rec["scale"][k] for k, v in record["scale"].items()),
            "the scale is not the one R1q used")
    checkpoint("the model and the common scale")

    # -- A and B: two temperatures on saved flows ------------------------------------------------
    record["A_r1q_main_point"] = fixed_flow_temperature(
        "A_r1q_main_point", fq["main_point_s"], fq["main_point_press_vel"],
        fq["main_point_temperature"], main_eval["compliance"])
    checkpoint("A: R1q's main point on its saved flow (1T)")

    s_bin = jnp.asarray(fr["solid_fraction_binary"])
    copy_ok = bool(np.array_equal(np.asarray(problem.flow_density(s_bin)), fr["new_solid_fraction_flow_h2"]))
    require(copy_ok, "R1r's binary design copied to h/2 is not its saved flow-mesh density")
    record["B_r1r_binary"] = {"copy_to_flow_mesh_is_the_saved_one": copy_ok}
    checkpoint("B: before its solve")
    record["B_r1r_binary"].update(fixed_flow_temperature(
        "B_r1r_binary", s_bin, fr["new_press_vel_check"], fr["new_temperature_check"],
        binary_cell["compliance"]))
    checkpoint("B: R1r's qualified binary design on its saved flow (1T)")

    # -- C: one full value and gradient at R1q's raw main design ------------------------------------
    root = problem.projection_root(jnp.asarray(x), BETA)
    require(root["nondegenerate"], f"the main point's projection root is degenerate: {root}")
    checkpoint("C: the main point's root")
    t0 = time.perf_counter()
    try:
        main_rec, state, grads = ff.evaluate(problem, scale, jnp.asarray(x), ALPHA_MAX, BETA, True, GATE)
    except (r1.NotConverged, r1.DegenerateProjection) as exc:
        require(False, f"C: {type(exc).__name__}: {exc}")
        checkpoint("C: the full value and gradient")
    cost["C_evaluate_s"] = time.perf_counter() - t0
    c_out = {"evaluation": main_rec,
             "psi_relative": rel(main_rec["psi"], main_eval["psi"]),
             "g_difference": main_rec["constraint_g"] - main_eval["constraint_g"],
             "compliance_relative": rel(main_rec["compliance"], main_eval["compliance"]),
             "J_relative": rel(main_rec["J_common_scale"], main_eval["J_common_scale"]),
             "flow_state_difference": state_difference(state[1], fq["main_point_press_vel"]),
             "temperature_difference": state_difference(state[2], fq["main_point_temperature"]),
             "gradients": {}}
    for key in ("J", "psi", "c", "g"):
        new, old = grads[key], fq[f"gradient_{key}"]
        c_out["gradients"][key] = {
            "relative_l2": float(np.linalg.norm(new - old) / np.linalg.norm(old)),
            "max_abs": float(np.abs(new - old).max()), "bitwise_equal": bool(np.array_equal(new, old)),
            "norm": float(np.linalg.norm(new))}
    saved.update({f"C_gradient_{k}": v for k, v in grads.items()})
    saved.update(C_temperature=state[2])
    require(abs(c_out["psi_relative"]) <= ANCHOR_TOL, f"C: Psi differs from R1q's by {c_out['psi_relative']:.3e}")
    require(abs(c_out["g_difference"]) <= ANCHOR_TOL, f"C: g differs from R1q's by {c_out['g_difference']:.3e}")
    for q in ("compliance", "J"):
        require(abs(c_out[f"{q}_relative"]) <= VALUE_TOL,
                f"C: {q} differs from R1q's by {c_out[f'{q}_relative']:.3e}")
    for key, v in c_out["gradients"].items():
        require(v["relative_l2"] <= GRADIENT_TOL,
                f"C: the {key} gradient differs from R1q's by {v['relative_l2']:.3e} (relative L2)")

    # along R1q's two directions, against its recorded central differences
    free = (x >= FD_ACTIVE_MARGIN) & (x <= 1.0 - FD_ACTIVE_MARGIN)
    c_out["directions"] = []
    for rec_dir in q_rec["directions"]:
        seed = rec_dir["seed"]
        d = np.random.default_rng(seed).choice([-1.0, 1.0], x.size) * free
        require(int(free.sum()) == rec_dir["active_components"],
                f"seed {seed}: {int(free.sum())} active variables, not R1q's {rec_dir['active_components']}")
        row = {"seed": seed}
        for q in QUANTITIES:
            ad = float(np.dot(grads[q], d))
            best = min(rec_dir["steps"], key=lambda st: st[q]["rel_error"])
            fd = best[q]["fd"]
            row[q] = {"ad_linear": ad, "ad_r1q": rec_dir["ad"][q], "fd_r1q_best_step": fd,
                      "best_step": best["step"], "rel_error_vs_fd": abs(ad - fd) / max(abs(fd), 1e-300),
                      "rel_difference_vs_r1q_ad": abs(ad - rec_dir["ad"][q]) / max(abs(rec_dir["ad"][q]), 1e-300)}
            require(row[q]["rel_error_vs_fd"] <= FD_TOL,
                    f"seed {seed}, {q}: {row[q]['rel_error_vs_fd']:.3e} from R1q's recorded difference")
        c_out["directions"].append(row)
    record["C_r1q_main_point_full"] = c_out
    t = main_rec["timing_s"]
    print(f"C: J {main_rec['J_common_scale']:.12f} (R1q {main_eval['J_common_scale']:.12f}, rel "
          f"{c_out['J_relative']:+.2e})  Psi rel {c_out['psi_relative']:+.1e}  C rel "
          f"{c_out['compliance_relative']:+.2e}  |R| {main_rec['flow_residual_relative']:.1e}/"
          f"{main_rec['thermal_residual_relative']:.1e}; gradients relative L2 " + "  ".join(
              f"{k} {v['relative_l2']:.1e}" for k, v in c_out["gradients"].items())
          + f"; forward {t['forward']:.1f} s, reverse {t['reverse_psi']:.1f} + {t['reverse_c']:.1f} s",
          flush=True)
    for row in c_out["directions"]:
        print(f"  seed {row['seed']}: against R1q's recorded differences " + "  ".join(
            f"{q} {row[q]['rel_error_vs_fd']:.1e}" for q in QUANTITIES), flush=True)
    checkpoint("C: the full value and gradient (1F + 1T)")

    cost["solves"] = {"flow_forward": 1, "thermal_forward": 3, "thermal_linear_solves_forward": 3,
                      "reverse_passes": 2, "mma": 0}
    cost["historical_newton_path_not_paired"] = {
        "r1q_main_point_value_and_gradient_s": q_rec["main_point"]["seconds"],
        "r1r_binary_thermal_solve_s": binary_cell["t_thermal_solve_s"],
        "note": "other runs, other processes and compile states: context, not a benchmark"}
    record["verdict"] = {"equivalent": True}
    finish()
    print(f"\nverdict: equivalent within the fixed criteria; wrote {out / RECORD} and {out / FIELDS}; "
          f"total {record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
