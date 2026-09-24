"""Figures of the Zhao 2D heat-sink reproduction, drawn from saved results only.

Nothing is solved or re-optimised here: every field and number is read from
results/ and drawn, so a figure shows exactly the state the records describe
-- including that the R1d run is budget-limited and not converged, and that the
thermal compliance still moves with the thermal mesh. Four figures, written to
docs/figures/:

  zhao2d_r1d_fields.png        the R1d design in the layout of Zhao Figs. 8 and
                               11 (density, velocity, temperature on the half
                               model, symmetry plane on the left), with the
                               temperature also on the finest thermal mesh
                               solved so far (h/8, R1i)
  zhao2d_r1d_history.png       the 300-update history on the frozen self scale
  zhao2d_status.png            where the result sits against Zhao Tables 4 and 7
                               on the paper-interpreted scale, and how C moves
                               with the thermal mesh on the coarse and fine flow
  zhao2d_r1k_warm_start.png    R1k: density and temperature at x_300 and after
                               30 updates on the flow h / thermal h/4 model, the
                               objective's history and the design step

    python scripts/zhao2d_figures.py [--results results] [--out docs/figures]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent

# -- the chart system (see the dataviz reference palette) ----------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, fixed order
PAPER = "#898781"  # the paper's points: neutral, so colour marks this work only
# sequential ramps, one hue each, lightest step at the surface
DENSITY = LinearSegmentedColormap.from_list(
    "fluid", ["#f0efec", "#c3c2b7", "#898781", "#52514e", "#0b0b0b"])
SPEED = LinearSegmentedColormap.from_list(
    "speed", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"])
HEAT = LinearSegmentedColormap.from_list(
    "heat", ["#fcfcfb", "#fbe3d6", "#f7c3a7", "#f19e75", "#eb6834",
             "#c9501f", "#9c3c14", "#6e2a0c"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# Zhao section 4.1, Tables 4 and 7: (case, J, v_f, C/C0, Psi/Psi0), as printed.
# The paper's own scale; J = (Psi/Psi0 + C/C0)/2 holds for every row.
ZHAO_FIXED = [(1, 0.9053, 0.3928, 1.1794, 0.6312), (2, 0.8875, 0.3954, 1.2201, 0.5550),
              (3, 0.8776, 0.3958, 1.2193, 0.5359), (4, 0.8817, 0.3981, 1.2567, 0.5066),
              (5, 0.8804, 0.3982, 1.2051, 0.5556), (6, 0.8711, 0.3982, 1.2190, 0.5232)]
ZHAO_ADAPTIVE = [(7, 0.8745, 0.4000, 1.2396, 0.5095), (8, 0.8812, 0.4000, 1.2658, 0.4965),
                 (9, 0.8919, 0.4000, 1.2653, 0.5186), (10, 0.8739, 0.4000, 1.2345, 0.5133),
                 (11, 0.8761, 0.4000, 1.2392, 0.5130), (12, 0.9097, 0.3999, 1.3487, 0.4707)]
# The paper-interpreted scale: Zhao's Psi_0 and C_0 with their labels swapped
# (R0 finding; not an author-confirmed erratum).
PAPER_PSI_0, PAPER_C_0 = 0.0456, 20816.0


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def geometry(r1h: dict) -> dict:
    """The half model, read from the recorded flow identity rather than restated."""
    return r1h["flows"]["h"]["identity"]["spec"]


def in_domain(cx, cy, g) -> np.ndarray:
    design = (cy > 0.0) & (cy < g["design_height"]) & (cx < g["design_half_width"])
    tabs = (cx < g["inlet_half_width"]) & (
        ((cy > g["design_height"]) & (cy < g["design_height"] + g["tab_length"]))
        | ((cy < 0.0) & (cy > -g["tab_length"])))
    return design | tabs


def triangulation(coords: np.ndarray, h: float, g: dict) -> Triangulation:
    """Two triangles per Q1 cell of the masked lattice the mesh was built on."""
    y0 = -g["tab_length"]
    i = np.rint(coords[:, 0] / h).astype(int)
    j = np.rint((coords[:, 1] - y0) / h).astype(int)
    if not (np.allclose(i * h, coords[:, 0], atol=1e-9 * h)
            and np.allclose(y0 + j * h, coords[:, 1], atol=1e-9 * h)):
        raise RuntimeError("nodes are not on the expected lattice")
    index = -np.ones((i.max() + 1, j.max() + 1), dtype=int)
    index[i, j] = np.arange(len(coords))
    ci, cj = np.meshgrid(np.arange(i.max()), np.arange(j.max()), indexing="ij")
    ci, cj = ci.ravel(), cj.ravel()
    keep = in_domain((ci + 0.5) * h, y0 + (cj + 0.5) * h, g)
    ci, cj = ci[keep], cj[keep]
    corners = np.stack([index[ci, cj], index[ci + 1, cj],
                        index[ci + 1, cj + 1], index[ci, cj + 1]], axis=1)
    if (corners < 0).any():
        raise RuntimeError("a domain cell is missing a node")
    tris = np.concatenate([corners[:, [0, 1, 2]], corners[:, [0, 2, 3]]])
    return Triangulation(coords[:, 0], coords[:, 1], tris)


def outline(ax, g) -> None:
    """The domain boundary, a thin line, so fields read against the geometry."""
    w, H, t, a = (g["design_half_width"], g["design_height"], g["tab_length"],
                  g["inlet_half_width"])
    xs = [0, a, a, w, w, a, a, 0, 0]
    ys = [H + t, H + t, H, H, 0, 0, -t, -t, H + t]
    ax.plot(xs, ys, color=INK2, lw=0.6)
    ax.plot([0, 0], [-t, H + t], color=INK2, lw=0.6, ls=(0, (3, 2)))  # symmetry


def field_axes(ax, g, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-0.0003, g["design_half_width"] + 0.0003)
    ax.set_ylim(-g["tab_length"] - 0.0003, g["design_height"] + g["tab_length"] + 0.0003)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5)
    outline(ax, g)


def colorbar(fig, mappable, ax, label: str, **kw):
    """A horizontal bar in its own inset under the panel, so notes can sit below it."""
    cax = ax.inset_axes([0.04, -0.045, 0.92, 0.02])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", **kw)
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.set_label(label, color=INK2, fontsize=8.5)
    return cb


def note(ax, text: str) -> None:
    ax.text(0.04, -0.125, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=INK2, linespacing=1.35)


def footer(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, fontsize=7, color=MUTED, ha="left", va="bottom")


# -- figure 1: the fields ----------------------------------------------------------


def fields_figure(res: pathlib.Path, out: pathlib.Path, g: dict, sources: list) -> pathlib.Path:
    r1d = np.load(res / "zhao2d_r1d_main_fields.npz")
    r1g = np.load(res / "zhao2d_r1g_fields.npz")
    r1i = np.load(res / "zhao2d_r1i_fields.npz")
    meta = json.loads((res / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    r1i_rec = json.loads((res / "zhao2d_r1i_h8.json").read_text(encoding="utf-8"))
    h = g["element_size"]

    # density: one value per design-mesh element, drawn as the element itself
    centres = r1d["elem_centres"]
    gamma = 1.0 - r1d["solid_fraction"]
    nx = int(round(g["design_half_width"] / h))
    ny = int(round((g["design_height"] + 2 * g["tab_length"]) / h))
    grid = np.full((nx, ny), np.nan)
    grid[np.floor(centres[:, 0] / h).astype(int),
         np.floor((centres[:, 1] + g["tab_length"]) / h).astype(int)] = gamma
    xe = np.arange(nx + 1) * h
    ye = -g["tab_length"] + np.arange(ny + 1) * h

    coords_h = r1g["thermal_node_coords_level1"]  # the h mesh; flow and thermal share it
    tri_h = triangulation(coords_h, h, g)
    uv = r1d["press_vel"].reshape(-1, 3)[:, 1:]
    speed = np.linalg.norm(uv, axis=1)
    t_h = r1d["temperature"]  # the model R1d optimised: thermal h, 2x2
    coords_8 = r1i["thermal_node_coords_h8"]
    t_8 = r1i["temperature_thermal_h8"]
    tri_8 = triangulation(coords_8, h / 8, g)
    t_top = float(max(t_h.max(), t_8.max()))

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 8.0))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.17, wspace=0.10)

    ax = axes[0]
    field_axes(ax, g, "(a) Density field: fluid fraction γ")
    m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY,
                      vmin=0, vmax=1, shading="flat")
    colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
    term = meta["terminal"]
    note(ax, f"per element, design mesh h = {h:g}\n"
             f"v_f (design domain) {term['v_f_design_domain']:.5f}\n"
             f"grey 0.05 < s < 0.95: {term['grey_fraction']:.1%} of cells")

    ax = axes[1]
    field_axes(ax, g, "(b) Velocity field |u|")
    m = ax.tripcolor(tri_h, speed, cmap=SPEED, vmin=0,
                     vmax=max(0.3, float(speed.max())), shading="gouraud")
    colorbar(fig, m, ax, "|u|")
    ax.annotate("", xy=(0.5 * g["inlet_half_width"], g["design_height"] + 0.4 * g["tab_length"]),
                xytext=(0.5 * g["inlet_half_width"], g["design_height"] + 1.25 * g["tab_length"]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
    ax.text(g["inlet_half_width"] + 0.00015, g["design_height"] + 0.9 * g["tab_length"],
            "inlet, v = −0.2", fontsize=7.5, color=INK2, va="center")
    ax.text(g["inlet_half_width"] + 0.00015, -0.5 * g["tab_length"], "outlet",
            fontsize=7.5, color=INK2, va="center")
    note(ax, f"flow mesh h, Q1; max |u| {speed.max():.3f}\n"
             f"Ψ = {term['psi']:.6f}\n"
             "the flow the optimiser used")

    tmin_h = float(t_h.min())
    below_h = int((t_h < -1e-10).sum())
    ax = axes[2]
    field_axes(ax, g, "(c) Temperature, thermal mesh h")
    m = ax.tripcolor(tri_h, t_h, cmap=HEAT, vmin=0, vmax=t_top, shading="gouraud")
    colorbar(fig, m, ax, "T", extend="min" if tmin_h < 0 else "neither")
    note(ax, f"the model R1d optimised (2×2)\n"
             f"T_max {t_h.max():.2f}; T_min {tmin_h:.2f} ({below_h} nodes < 0)\n"
             f"C = {term['compliance']:.0f}")

    cell = r1i_rec["cell"]
    ax = axes[3]
    field_axes(ax, g, "(d) Temperature, thermal mesh h/8")
    m = ax.tripcolor(tri_8, t_8, cmap=HEAT, vmin=0, vmax=t_top, shading="gouraud")
    colorbar(fig, m, ax, "T  (same scale as c)")
    note(ax, f"same design and flow, 3×3 (R1i)\n"
             f"T_max {t_8.max():.2f}; T_min {t_8.min():.2f}\n"
             f"C = {cell['compliance']:.0f}")

    fig.suptitle(
        "Zhao §4.1 heat sink by the density method — the R1d design after 300 MMA "
        "updates (budget-limited, not converged)", x=0.02, ha="left", fontsize=11.5,
        color=INK, y=0.975)
    fig.text(0.02, 0.925,
             "Half model as in Zhao Figs. 7(b), 8 and 11: symmetry plane on the left (dashed), "
             "inlet tab top-left, outlet tab bottom-left. Zhao's figures scale |u| 0–0.3 and T "
             "0–12;\nhere T shares one 0–%.1f scale across (c) and (d), and (c)'s undershoot "
             "below 0 is drawn at the lightest step (colour bar arrow)." % t_top,
             fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from " + ", ".join(sources[:3]) + " — scripts/zhao2d_figures.py; "
           "no state is re-solved.")
    path = out / "zhao2d_r1d_fields.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 2: the optimisation history ------------------------------------------------


def history_figure(res: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    meta = json.loads((res / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    hist = meta["history"]
    it = np.array([r["iteration"] for r in hist])
    series = [("J (self scale)", np.array([r["J_self"] for r in hist])),
              ("Ψ/Ψ₀", np.array([r["psi_over_psi0_self"] for r in hist])),
              ("C/C₀", np.array([r["c_over_c0_self"] for r in hist]))]
    vf = np.array([r["v_f_design_domain"] for r in hist])
    phase = [r["phase"] for r in hist]
    alpha = np.array([r["alpha_max"] for r in hist])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1.3]})
    fig.subplots_adjust(left=0.08, right=0.84, top=0.86, bottom=0.10, hspace=0.12)

    # phase boundaries, as hairlines with names
    starts = [0] + [k for k in range(1, len(phase)) if phase[k] != phase[k - 1]]
    for ax in (top, bottom):
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in starts[1:]:
            ax.axvline(it[s] - 0.5, color=AXIS, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for a, s in enumerate(starts):
        end = starts[a + 1] if a + 1 < len(starts) else len(phase)
        label = phase[s]
        if a == 0:
            k78 = int(np.argmax(alpha >= 1e7))
            label = f"{phase[s]} (α_max 10⁶→10⁷ by n = {it[k78]})"
        top.text((it[s] + it[end - 1]) / 2, 1.02, label, transform=top.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7.5, color=INK2)

    top.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    top.text(it[-1] + 3, 1.0, "reference state = 1", fontsize=7.5, color=MUTED, va="center")
    for (label, y), colour in zip(series, SERIES):
        top.plot(it, y, color=colour, lw=1.6)
        top.text(it[-1] + 3, y[-1], f"{label}  {y[-1]:.4f}", color=INK, fontsize=8,
                 va="center")
        top.plot([it[-1]], [y[-1]], "o", color=colour, ms=5)
    top.set_ylabel("ratio to the frozen reference")
    top.set_title("R1d: 300 MMA updates at h = 10⁻⁴, 5200 elements — stopped at the end of "
                  "the β = 8 phase, not converged", loc="left", pad=22)

    bottom.plot(it, vf, color=INK2, lw=1.4)
    bottom.axhline(0.40, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    bottom.text(it[-1] + 3, 0.40, "bound 0.40", fontsize=7.5, color=MUTED, va="center")
    bottom.text(it[-1] + 3, vf[-1] - 0.02, f"v_f {vf[-1]:.4f}", fontsize=8, color=INK,
                va="center")
    bottom.set_ylabel("fluid fraction,\ndesign domain")
    bottom.set_xlabel("MMA update")
    bottom.set_xlim(it[0], it[-1])

    fig.text(0.08, 0.955, "Optimisation history on the self scale (Ψ₀, C₀ frozen once at "
             "γ = 0.4, α_max = 10⁶ on the design mesh)", fontsize=11, color=INK, ha="left",
             va="top")
    footer(fig, "Drawn from results/zhao2d_r1d_main.json — scripts/zhao2d_figures.py. "
           "Each recorded state passed the residual gate.")
    path = out / "zhao2d_r1d_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 3: where the numbers stand ---------------------------------------------------


def status_figure(res: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    meta = json.loads((res / "zhao2d_r1d_main.json").read_text(encoding="utf-8"))
    r1g = json.loads((res / "zhao2d_r1g_dual.json").read_text(encoding="utf-8"))
    r1h = json.loads((res / "zhao2d_r1h_matrix.json").read_text(encoding="utf-8"))
    r1i = json.loads((res / "zhao2d_r1i_h8.json").read_text(encoding="utf-8"))
    term = meta["terminal"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.14, wspace=0.28)

    # (a) against the paper, on the paper-interpreted scale
    ax = left
    xlo, xhi, ylo, yhi = 0.25, 0.70, 1.05, 1.90
    for j in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05):
        ax.plot([xlo, xhi], [2 * j - xlo, 2 * j - xhi], color=GRID, lw=0.8, zorder=0)
        # label where the line leaves the box at the lower right, pulled inside it
        xl = min(xhi, 2 * j - ylo) - 0.012
        if xlo < xl < xhi and ylo < 2 * j - xl < yhi:
            ax.text(xl, 2 * j - xl + 0.012, f"J = {j:.2f}", fontsize=7, color=MUTED,
                    ha="right", va="bottom")
    for rows, marker, label in ((ZHAO_FIXED, "o", "Zhao, fixed CBS count (Table 4)"),
                                (ZHAO_ADAPTIVE, "^", "Zhao, adaptive CBS (Table 7)")):
        ax.plot([r[4] for r in rows], [r[3] for r in rows], marker, ms=7, color=PAPER,
                mfc=PAPER, mec=SURFACE, mew=1.2, ls="none", label=label)
    ours_x, ours_y = term["psi_over_paper"], term["c_over_paper"]
    c8 = r1i["cell"]["compliance"]
    psi = r1i["cell"]["psi"]
    ax.annotate("", xy=(psi / PAPER_PSI_0, c8 / PAPER_C_0), xytext=(ours_x, ours_y),
                arrowprops=dict(arrowstyle="-|>", color=SERIES[0], lw=1.2,
                                shrinkA=5, shrinkB=5))
    ax.plot([ours_x], [ours_y], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=1.5,
            label="this work, R1d (thermal h, as optimised)")
    ax.plot([psi / PAPER_PSI_0], [c8 / PAPER_C_0], "o", ms=8, mfc=SURFACE,
            mec=SERIES[0], mew=1.6, label="same design, thermal h/8 (R1i)")
    ax.text(ours_x + 0.012, ours_y, f"J {term['J_paper_interpreted']:.3f}", fontsize=8,
            color=INK, va="center")
    j8 = 0.5 * (psi / PAPER_PSI_0 + c8 / PAPER_C_0)
    ax.text(psi / PAPER_PSI_0 + 0.012, c8 / PAPER_C_0, f"J {j8:.3f}", fontsize=8,
            color=INK, va="center")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Ψ/Ψ₀  (paper-interpreted scale, Ψ₀ = 0.0456)")
    ax.set_ylabel("C/C₀  (paper-interpreted scale, C₀ = 20816)")
    ax.grid(color=GRID, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK2)
    ax.set_title("(a) Objective space against Zhao's CBS results", loc="left")

    # (b) C against the thermal mesh, coarse and fine flow
    ax = right
    elems = np.array([5200, 20800, 83200, 332800])
    coarse = [r1g["levels"]["1"]["compliance"], r1g["levels"]["2"]["compliance"],
              r1g["levels"]["4"]["compliance"], r1i["cell"]["compliance"]]
    fine = [r1h["summary"]["flow_h2__thermal_h2"]["compliance"],
            r1h["summary"]["flow_h2__thermal_h4"]["compliance"]]
    for k, f in zip((1, 2), fine):  # the flow replacement at a fixed thermal mesh
        ax.plot([elems[k]] * 2, [coarse[k], f], color=MUTED, lw=0.8, ls=(0, (1, 2)))
    ax.plot(elems, coarse, "-o", color=SERIES[0], lw=1.6, ms=6, mec=SURFACE, mew=1.2,
            label="flow on h (the model R1d optimised)")
    ax.plot(elems[1:3], fine, "-o", color=SERIES[1], lw=1.6, ms=6, mec=SURFACE, mew=1.2,
            label="flow on h/2 (R1h)")
    for a, b in zip(range(3), range(1, 4)):
        step = coarse[b] / coarse[a] - 1
        ax.text(np.sqrt(elems[a] * elems[b]), (coarse[a] + coarse[b]) / 2 + 300,
                f"{step:+.1%}", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.text(elems[2] * 1.12, fine[1], f"{fine[1]:.0f}", fontsize=8, color=INK,
            va="center")
    ax.text(elems[3] / 1.08, coarse[3] + 250, f"{coarse[3]:.0f}", fontsize=8,
            color=INK, ha="right", va="bottom")
    ax.text(elems[1] * 1.08, fine[0] - 350, "flow h → h/2: −2.5%", fontsize=7.5,
            color=INK2, va="top")
    ax.text(elems[2] * 1.08, fine[1] - 350, "−2.7%", fontsize=7.5, color=INK2, va="top")
    ratio = r1i["steps"]["compliance"]["abs_ratio_48_over_24"]
    ax.text(elems[3], coarse[0], f"|Δ₄₈| / |Δ₂₄| = {ratio:.2f}\n(an observation, not an "
            "error estimate)", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(elems)
    ax.set_xticklabels(["h\n5,200", "h/2\n20,800", "h/4\n83,200", "h/8\n332,800"])
    ax.minorticks_off()
    ax.set_xlim(3800, 460000)
    ax.set_ylim(26000, 38400)
    ax.set_xlabel("thermal mesh (elements); design mesh h throughout")
    ax.set_ylabel("thermal compliance C (3×3)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK2)
    ax.set_title("(b) Fixed design: C against the thermal mesh", loc="left")

    fig.text(0.07, 0.965, "Where the reproduction stands", fontsize=11.5, color=INK,
             ha="left", va="top")
    fig.text(0.07, 0.915,
             "(a) The paper's constants read with their labels swapped (R0 finding, not an "
             "author erratum); different parametrisation (per-element density vs CBS) and "
             "stabilisation details,\nand R1d is not converged. Like-for-like is the filled "
             "point, on the paper's 5200-element mesh; the arrow is the same design with a "
             "finer thermal mesh alone. (b) Differences\nbetween meshes, not errors against "
             "an exact solution.", fontsize=8, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1d_main.json, zhao2d_r1g_dual.json, "
           "zhao2d_r1h_matrix.json, zhao2d_r1i_h8.json and Zhao et al. Tables 4 and 7 — "
           "scripts/zhao2d_figures.py")
    path = out / "zhao2d_status.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 4: the R1k warm start -------------------------------------------------------


def density_grid(s: np.ndarray, centres: np.ndarray, h: float, g: dict):
    """gamma = 1 - s per design-mesh element, on the lattice, NaN outside."""
    nx = int(round(g["design_half_width"] / h))
    ny = int(round((g["design_height"] + 2 * g["tab_length"]) / h))
    grid = np.full((nx, ny), np.nan)
    grid[np.floor(centres[:, 0] / h).astype(int),
         np.floor((centres[:, 1] + g["tab_length"]) / h).astype(int)] = 1.0 - s
    return np.arange(nx + 1) * h, -g["tab_length"] + np.arange(ny + 1) * h, grid


def r1k_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1k_warm_start.json").read_text(encoding="utf-8"))
    f = np.load(res / "zhao2d_r1k_fields.npz")
    coords_4 = np.load(res / "zhao2d_r1g_fields.npz")["thermal_node_coords_level4"]
    h = g["element_size"]
    if len(coords_4) != len(f["temperature"]):
        raise RuntimeError("the h/4 node coordinates do not match R1k's temperature")
    tri_4 = triangulation(coords_4, h / 4, g)
    first, term = rec["comparison"]["initial"], rec["comparison"]["terminal"]
    t_start, t_end = f["initial_temperature"], f["temperature"]
    t_top = float(max(t_start.max(), t_end.max()))

    fig = plt.figure(figsize=(11.0, 10.4))
    top = fig.add_gridspec(1, 4, left=0.02, right=0.99, top=0.885, bottom=0.40, wspace=0.10)
    low = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.27, bottom=0.07, wspace=0.62)

    for k, (s, title, r) in enumerate(((f["initial_solid_fraction"],
                                        "(a) Density γ at the start: R1d's x₃₀₀", first),
                                       (f["solid_fraction"],
                                        "(b) Density γ after 30 updates", term))):
        ax = fig.add_subplot(top[0, k])
        field_axes(ax, g, title)
        xe, ye, grid = density_grid(s, f["elem_centres"], h, g)
        m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY,
                          vmin=0, vmax=1, shading="flat")
        colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
        note(ax, f"β = 8, per element on h\nv_f (design domain) {r['v_f_design_domain']:.5f}\n"
                 f"grey 0.05 < s < 0.95: {r['grey_fraction']:.1%} of cells")

    for k, (t, title, r) in enumerate(((t_start, "(c) Temperature at the start", first),
                                       (t_end, "(d) Temperature after 30 updates", term))):
        ax = fig.add_subplot(top[0, 2 + k])
        field_axes(ax, g, title)
        m = ax.tripcolor(tri_4, t, cmap=HEAT, vmin=0, vmax=t_top, shading="gouraud")
        colorbar(fig, m, ax, "T  (one scale for c and d)")
        note(ax, f"thermal mesh h/4, 3×3\nT_max {r['T_max']:.2f}; T_min {r['T_min']:.2f}\n"
                 f"C = {r['compliance']:.0f};  Ψ = {r['psi']:.5f}")

    hist = rec["history"]
    term_it = rec["terminal"]["iteration"]
    it = np.array([r["iteration"] for r in hist] + [term_it])
    ax = fig.add_subplot(low[0, 0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(9, 0.975, "reference state = 1", fontsize=7.5, color=MUTED, va="top")
    for (label, key), colour in zip((("J", "J_self"),
                                     ("Ψ/Ψ₀", "psi_over_psi0_self"),
                                     ("C/C₀", "c_over_c0_self")), SERIES):
        y = np.array([r[key] for r in hist] + [rec["terminal"][key]])
        ax.plot(it, y, color=colour, lw=1.6)
        ax.plot([term_it], [y[-1]], "o", color=colour, ms=5)
        ax.text(term_it + 1.2, y[-1], f"{label}  {y[-1]:.4f}", color=INK, fontsize=8,
                va="center")
    ax.set_xlim(0, term_it)
    ax.set_xlabel("MMA update (30 = terminal, re-evaluated)")
    ax.set_ylabel("ratio to this model's frozen reference")
    ax.set_title("(e) Objective and its two terms, on this model's scale", loc="left",
                 fontsize=9.5)

    ax = fig.add_subplot(low[0, 1])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    two = np.array(rec["steps"]["two_norm"])
    mx = np.array(rec["steps"]["max_abs"])
    ax.plot(np.arange(1, len(two) + 1), two, color=INK2, lw=1.4)
    ax.set_xlim(1, len(two))
    ax.set_ylim(0, 1.1 * two.max())
    ax.set_xlabel("update")
    ax.set_ylabel("‖Δx‖₂ per update")
    ax.set_title("(f) Design step", loc="left", fontsize=9.5)
    ax.text(0.03, 0.06, f"max |Δx| {mx[3:].min():.3f}–{mx[3:].max():.3f} from update 4 on:\n"
                        f"the move limit 0.1 binds throughout", transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")

    change = rec["comparison"]["terminal_over_initial_minus_1"]
    fig.suptitle("R1k — warm start from x₃₀₀ on the development model (flow h, thermal h/4), "
                 "α_max = 10⁷, β = 8 fixed",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955,
             "30 MMA updates, the whole budget; not converged. "
             f"J {change['J_self']:+.1%} ({first['J_self']:.4f} → {term['J_self']:.4f}): "
             f"C {change['compliance']:+.1%}, Ψ {change['psi']:+.1%}. J is this model's own "
             "scale, not comparable with R1d's.\nEvery state passed the 10⁻⁸ residual gate; "
             "the lowest feasible J is the terminal design's. No same-point convergence check "
             "was made; upstream's KKT proxy only fell to "
             f"{hist[-1]['kkt_proxy']:.1e}.", fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1k_warm_start.json, results/zhao2d_r1k_fields.npz and "
           "the h/4 node coordinates in results/zhao2d_r1g_fields.npz — "
           "scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1k_warm_start.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=REPO / "results")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "docs" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res = args.results
    r1h = json.loads((res / "zhao2d_r1h_matrix.json").read_text(encoding="utf-8"))
    g = geometry(r1h)
    names = ["zhao2d_r1d_main_fields.npz", "zhao2d_r1g_fields.npz", "zhao2d_r1i_fields.npz",
             "zhao2d_r1d_main.json", "zhao2d_r1g_dual.json", "zhao2d_r1h_matrix.json",
             "zhao2d_r1i_h8.json", "zhao2d_r1k_warm_start.json", "zhao2d_r1k_fields.npz"]
    for n in names:
        print(f"read results/{n}  sha256 {sha(res / n)}")
    sources = [f"results/{n}" for n in names]
    for path in (fields_figure(res, args.out, g, sources), history_figure(res, args.out),
                 status_figure(res, args.out), r1k_figure(res, args.out, g)):
        print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")


if __name__ == "__main__":
    main()
