"""R1p: R1n's and R1o's binary designs on the bridge model, flow h / thermal h/8.

The contract proposed in the review of 6bd8cb9. On the development layer
A = (flow h, thermal h/4) R1n's qualified binary design is ahead of R1o's; on
the check layer D = (flow h/2, thermal h/8) R1o's is. The bridge
B = (flow h, thermal h/8) sits between them on one path:

    A = (h, h/4)  ->  B = (h, h/8)  ->  D = (h/2, h/8)

If the order has flipped at B, then along this path it flips in the thermal
refinement; if not, in the flow replacement. That is an attribution along one
path, not the full interaction and not a unique mechanism.

A and D are reused from R1n's and R1o's records, by hash. B is new: for each
design, its saved flow on h -- the flow its A state was solved with -- is
re-verified on the bridge problem, and the Psi it gives must reproduce the A
record's Psi (Psi depends only on the flow and s). Only then is the temperature
solved once on h/8, with the thermal form, 3x3 quadrature, tau rule,
boundaries, source and materials unchanged: at most 2 thermal solves, no flow
solve, no MMA, no AD. The bridge problem is R1m's flow-h row, and its meshes
must be the ones R1m recorded. A failed gate writes the record and exits
non-zero before the next solve.

J uses the h/4 model's constants as the common scale of R1l to R1o; no new
reference.

    python scripts/zhao2d_r1p_bridge.py [--inputs results] [--out DIR] [--overwrite]
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

from tfopus import materials  # noqa: E402
from tfopus import zhao2d as z  # noqa: E402
from tfopus import zhao2d_analysis as za  # noqa: E402
from tfopus import zhao2d_binary as zb  # noqa: E402
from tfopus import zhao2d_dual as dual  # noqa: E402
from tfopus import zhao2d_flow_study as fs  # noqa: E402
from tfopus import zhao2d_r1 as r1  # noqa: E402

from zhao2d_dual_check import peak_working_set_mb, timed  # noqa: E402
from zhao2d_flow_mesh_check import provenance, sha256_file  # noqa: E402
# R1m's report, so the bridge is summarised exactly as the other layers were
from zhao2d_r1m_flow_check import flat, mesh_identity, thermal_identity  # noqa: E402

ALPHA_MAX, QUADRATURE, BRIDGE_REFINEMENT = 1.0e7, 3, 8
GATE, ANCHOR_TOL, FLUID_BOUND = 1e-8, 1e-12, 0.4
RECORD = "zhao2d_r1p_bridge.json"
FIELDS = "zhao2d_r1p_fields.npz"
STACKS = "zhao2d_r1p_stacks.log"
INPUTS = ("zhao2d_r1m_flow_check.json", "zhao2d_r1n_beta32.json", "zhao2d_r1n_fields.npz",
          "zhao2d_r1o.json", "zhao2d_r1o_fields.npz")
NAMES = ("r1n", "r1o")
LAYERS = ("A", "B", "D")


def load_json(path: pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def split(new: dict, base: dict, yard: dict, w: float) -> dict:
    """R1o minus R1n at one layer: J, Psi, C relative, and the two weighted terms of dJ."""
    dp = w * (new["psi"] - base["psi"]) / yard["psi_0"]
    dc = (1 - w) * (new["compliance"] - base["compliance"]) / yard["c_0"]
    return {"J": new["J"] / base["J"] - 1.0, "psi": new["psi"] / base["psi"] - 1.0,
            "compliance": new["compliance"] / base["compliance"] - 1.0,
            "dissipation_term": dp, "thermal_term": dc, "dJ": dp + dc}


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
    faulthandler.dump_traceback_later(2400, repeat=True, file=stacks)  # longer than a healthy run
    t_start = time.perf_counter()

    spec, config = z.Zhao2DSpec(), r1.R1Config()
    w = config.weight
    material = za.build_material(spec, ALPHA_MAX)
    single = r1.load_reference(spec, config)  # J* only
    m_rec = load_json(inputs / "zhao2d_r1m_flow_check.json")
    recs = {"r1n": load_json(inputs / "zhao2d_r1n_beta32.json"),
            "r1o": load_json(inputs / "zhao2d_r1o.json")}
    fields = {"r1n": np.load(inputs / "zhao2d_r1n_fields.npz"),
              "r1o": np.load(inputs / "zhao2d_r1o_fields.npz")}
    # each design's saved states: A = its development cell, D = its check cell
    designs = {}
    for name in NAMES:
        f, cells_ = fields[name], recs[name]["cells"]
        designs[name] = {
            "s": f["solid_fraction_binary"], "press_vel": f["new_press_vel_development"],
            "temperature_A": f["new_temperature_development"],
            "s_h2": f["new_solid_fraction_flow_h2"], "press_vel_h2": f["new_press_vel_check"],
            "temperature_D": f["new_temperature_check"],
            "A": cells_["new/development"], "D": cells_["new/check"],
            "export": recs[name]["export"]}

    record = {
        "note": ("R1p: R1n's and R1o's qualified binary designs on the bridge model B = (flow h, "
                 "thermal h/8), between the development layer A = (h, h/4) and the check layer "
                 "D = (h/2, h/8), which are reused from their records. Each saved h flow is "
                 "re-verified and reused; one thermal solve per design. J on the h/4 model's "
                 "constants. An attribution along one path, not the full interaction."),
        "provenance": provenance(),
        "inputs_sha256": {n: sha256_file(inputs / n) for n in INPUTS},
        "alpha_max": ALPHA_MAX, "thermal_quadrature": QUADRATURE, "gate": GATE,
        "anchor_tolerance": ANCHOR_TOL,
        "budget": {"thermal_solves_at_most": 2, "flow_solves": 0, "mma": 0, "ad": 0},
    }
    for script in (pathlib.Path(__file__).resolve(), REPO / "scripts" / "zhao2d_r1m_flow_check.py"):
        record["provenance"]["source_sha256"][script.relative_to(REPO).as_posix()] = sha256_file(script)
    failures, checkpoints, cost, cells, saved = [], [], {}, {}, {}

    def require(ok, what: str) -> bool:
        if not ok:
            failures.append(what)
        return bool(ok)

    def finish() -> None:
        record.update(cells=cells, checkpoints=checkpoints)
        if saved:
            np.savez_compressed(out / FIELDS, **saved)
            record["fields"] = {"file": FIELDS, "arrays": sorted(saved)}
        record["cost"] = {**cost, "peak_working_set_mb": peak_working_set_mb(),
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
        if failures:
            record["stopped"] = f"{stage}: checks failed; nothing after it was built or solved"
            finish()
            sys.exit(f"CHECKS FAILED at {stage}; nothing after it was built or solved: {failures}")
        print(f"checkpoint passed: {stage}", flush=True)

    # -- inputs: the saved states are the recorded ones ------------------------------------
    for name, d in designs.items():
        rec = recs[name]
        require(not any(c["failures"] for c in rec["checkpoints"]), f"{name}: its record has failed checks")
        require(fs.digest(d["s"]) == d["export"]["binary_sha256"], f"{name}: s is not its exported design")
        for layer, keys in (("A", (("s", "s"), ("press_vel", "press_vel"), ("temperature_A", "temperature"))),
                            ("D", (("s_h2", "s"), ("press_vel_h2", "press_vel"),
                                   ("temperature_D", "temperature")))):
            for mine, theirs in keys:
                require(fs.digest(d[mine]) == d[layer]["state_sha256"][theirs],
                        f"{name}/{layer}: the saved {mine} is not the recorded one")
            require(zb.cell_usable(d[layer]) and d[layer].get("volume_feasible"),
                    f"{name}/{layer}: not a usable, qualified cell in its record")
    yard = {k: recs["r1n"]["yardstick"][k] for k in ("psi_0", "c_0", "file")}
    require(all(recs["r1o"]["yardstick"][k] == yard[k] for k in yard),
            "R1n's and R1o's yardsticks differ")
    record["yardstick"] = yard
    record["reporting_scale"] = {"psi_0": single.psi_0, "c_0": single.c_0,
                                 "identity": "single-mesh h reference; J* only"}
    checkpoint("inputs")

    # -- the bridge problem: R1m's flow-h row, checked before anything is solved -----------
    t0 = time.perf_counter()
    problem = dual.Zhao2DDualProblem(spec, config, thermal_refinement=BRIDGE_REFINEMENT,
                                     thermal_quadrature=QUADRATURE)
    cost["build_bridge_s"] = time.perf_counter() - t0
    print(f"built the bridge: flow {problem.flow_mesh.num_elems} el, thermal "
          f"{problem.thermal_mesh.num_elems} el ({cost['build_bridge_s']:.1f} s)", flush=True)
    rows = m_rec["rows"]["flow_h"]
    record["bridge"] = {"flow_mesh": mesh_identity(problem.flow_mesh),
                        "thermal_mesh": thermal_identity(problem)}
    require(record["bridge"]["flow_mesh"] == rows["flow_mesh"], "the flow mesh is not R1m's flow-h row's")
    require(record["bridge"]["thermal_mesh"] == rows["thermal_mesh"],
            "the h/8 thermal side is not R1m's flow-h row's")
    for name, d in designs.items():
        key = f"{name}/B"
        s, pv = jnp.asarray(d["s"]), jnp.asarray(d["press_vel"])
        fluid = z.fluid_fractions(problem.flow_mesh, s)
        cell = {"design": name, "layer": "B", "flow": "h: the saved state its A cell was solved with",
                "temperature": "h/8: solved here", "fluid_fractions": fluid,
                "volume_feasible": zb.volume_feasible(fluid["v_f_design_domain"], FLUID_BOUND)}
        cells[key] = cell
        require(cell["volume_feasible"], f"{key}: over the fluid bound, {fluid}")
        try:
            cell["flow_verification"] = fs.verify_flow_state(problem, pv, s, ALPHA_MAX, GATE)
        except (ValueError, r1.NotConverged) as exc:
            require(False, f"{key}: the saved flow does not verify: {exc}")
            continue
        alpha = materials.brinkman_penalty(s, material)
        psi = float(problem.flow.dissipated_power(pv, alpha))
        relative = psi / d["A"]["psi"] - 1.0
        cell["psi_anchor"] = {"source": f"{name}'s A cell (Psi depends only on the flow and s)",
                              "psi": psi, "record_psi": d["A"]["psi"], "psi_relative": relative,
                              "reproduced": bool(np.isfinite(relative) and abs(relative) <= ANCHOR_TOL)}
        require(cell["psi_anchor"]["reproduced"],
                f"{key}: Psi does not reproduce the A record: {cell['psi_anchor']}")
        print(f"{key}: flow |R| {cell['flow_verification']['residual_relative']:.1e}, Psi {psi:.9e} "
              f"(A record {d['A']['psi']:.9e})", flush=True)
    checkpoint("bridge: identities, flows and Psi, before any solve")

    # -- one thermal solve per design ---------------------------------------------------------
    for name, d in designs.items():
        key = f"{name}/B"
        s, pv = jnp.asarray(d["s"]), jnp.asarray(d["press_vel"])
        cell = cells[key]
        temp, t_thermal = timed(problem.solve_thermal, pv, s, ALPHA_MAX)
        cell["t_thermal_solve_s"] = t_thermal
        saved[f"{name}_temperature_bridge"] = np.asarray(temp)
        t0 = time.perf_counter()
        rep = fs.cell_report(problem, s, pv, temp, ALPHA_MAX, single)
        cell.update(analysed=True, **flat(rep, yard, w))
        cell["gate_passed"] = zb.gate_passed(rep["residual_relative"], GATE)
        cell["t_report_s"] = time.perf_counter() - t0
        cell["report"] = rep
        cell["state_sha256"] = {"s": fs.digest(np.asarray(s)), "press_vel": fs.digest(np.asarray(pv)),
                                "temperature": fs.digest(np.asarray(temp))}
        print(f"{key}: Psi {cell['psi']:.9e}  C {cell['compliance']:.6f}  J {cell['J']:.9f}  "
              f"D_T/Q {cell['D_T_over_Q']:.4%}  |R| {cell['residual_flow']:.1e}/"
              f"{cell['residual_thermal']:.1e}  thermal {t_thermal:.1f} s", flush=True)
        require(cell["gate_passed"], f"{key}: a residual fails the gate {rep['residual_relative']}")
        checkpoint(f"bridge: {name}'s thermal solve ({t_thermal:.1f} s)")

    # -- the path A -> B -> D ------------------------------------------------------------------
    by = {name: {"A": designs[name]["A"], "B": cells[f"{name}/B"], "D": designs[name]["D"]}
          for name in NAMES}
    record["layers"] = {name: {layer: {q: by[name][layer][q] for q in ("psi", "compliance", "J")}
                               for layer in LAYERS} for name in NAMES}
    diff = {layer: split(by["r1o"][layer], by["r1n"][layer], yard, w) for layer in LAYERS}
    record["r1o_minus_r1n"] = diff
    record["steps"] = {
        step: {q: diff[b][q] - diff[a][q] for q in ("dissipation_term", "thermal_term", "dJ")}
        for step, (a, b) in (("thermal_refinement_A_to_B", ("A", "B")),
                             ("flow_replacement_B_to_D", ("B", "D")))}
    record["each_design"] = {
        name: {"C_A_to_B": by[name]["B"]["compliance"] / by[name]["A"]["compliance"] - 1.0,
               "C_B_to_D": by[name]["D"]["compliance"] / by[name]["B"]["compliance"] - 1.0,
               "J_A_to_B": by[name]["B"]["J"] / by[name]["A"]["J"] - 1.0,
               "J_B_to_D": by[name]["D"]["J"] / by[name]["B"]["J"] - 1.0}
        for name in NAMES}
    ahead = {layer: ("r1o" if diff[layer]["dJ"] < 0 else "r1n") for layer in LAYERS}
    record["ahead_at_w_0_5"] = ahead
    record["flip"] = ("in the thermal refinement A -> B" if ahead["B"] != ahead["A"]
                      else "in the flow replacement B -> D" if ahead["D"] != ahead["B"]
                      else "no flip along this path")
    finish()

    print("\nR1o minus R1n (w = 0.5, the common denominators):")
    for layer in LAYERS:
        v = diff[layer]
        print(f"  {layer}: J {v['J']:+.4%}  Psi {v['psi']:+.4%}  C {v['compliance']:+.4%}   "
              f"terms {v['dissipation_term']:+.9f} {v['thermal_term']:+.9f} = {v['dJ']:+.9f}  "
              f"-> {ahead[layer]} ahead")
    for step, v in record["steps"].items():
        print(f"  {step}: dissipation {v['dissipation_term']:+.9f}  thermal {v['thermal_term']:+.9f}  "
              f"dJ {v['dJ']:+.9f}")
    for name, v in record["each_design"].items():
        print(f"  {name}: C A->B {v['C_A_to_B']:+.3%}, B->D {v['C_B_to_D']:+.3%}")
    print(f"\nthe order flips {record['flip']}")
    print(f"wrote {out / RECORD} and {out / FIELDS}; total {record['cost']['wall_clock_total_s']:.0f} s")


if __name__ == "__main__":
    main()
