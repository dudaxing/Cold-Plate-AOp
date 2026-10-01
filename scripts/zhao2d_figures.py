"""Figures of the Zhao 2D heat-sink reproduction, drawn from saved results only.

Nothing is solved or re-optimised here: every field and number is read from
results/ and drawn, so a figure shows exactly the state the records describe
-- including that the R1d run is budget-limited and not converged, and that the
thermal compliance still moves with the thermal mesh. Fifteen figures, written
to docs/figures/:

  zhao2d_r1d_fields.png        the R1d design in the layout of Zhao Figs. 8 and
                               11 (density, velocity, temperature on the half
                               model, symmetry plane on the left), with the
                               temperature also on the finest thermal mesh
                               solved so far (h/8, R1i)
  zhao2d_r1d_history.png       the 300-update history on the frozen self scale
  zhao2d_status.png            where the result sits against Zhao Tables 4 and 7
                               on the paper-interpreted scale (R1d, and the
                               current lead on D), and how C moves with the
                               thermal mesh on the coarse and fine flow
  zhao2d_r1k_warm_start.png    R1k: density and temperature at x_300 and after
                               30 updates on the flow h / thermal h/4 model, the
                               objective's history and the design step
  zhao2d_r1k_terminal_check.png  R1k's terminal design thresholded, its h/8
                               temperature continuous and binary, and J of the
                               start and the terminal under four evaluations
  zhao2d_r1l.png               R1l: the qualified binary baselines, the 30 updates on
                               the volume-preserving projection, and continuous
                               against qualified binary J
  zhao2d_r1m.png               R1m: the two qualified binary designs with the flow
                               on h or h/2, on thermal h/8 -- the pilot's
                               temperature, how each design's temperature moves,
                               J and the heat-balance deficit
  zhao2d_r1n.png               R1n: the pilot's binary design, the beta = 32 stage's
                               continuous and binary designs, its history, and the
                               three binary designs' J on both layers
  zhao2d_r1o.png               R1o: thirty more updates at beta = 32 from R1n, its
                               binary design, and the three candidates' J on both
                               layers
  zhao2d_r1p.png               R1p: R1o's and R1n's difference along the path
                               development -> bridge -> check, split into its terms,
                               and each design's C
  zhao2d_r1r.png               R1r: the updates on the check layer D from R1o's
                               design, the new binary design, and the qualified
                               binary designs' Psi and C on D
  zhao2d_r1t.png               R1t: twenty more updates on D from R1r's design on
                               the linear thermal path, after R1r's own, the new
                               binary design, and the five binary designs on D
  zhao2d_r1v.png               R1v: R1t's run continued with MMA's history, after
                               R1r's and R1t's updates, the new binary design, and
                               the binary designs on D
  zhao2d_r1w.png               R1w: the 18 binary designs along R1v's trajectory,
                               ranked once on D -- where they differ, their J by
                               iterate, and their Psi and C against R1t's design
  zhao2d_lead_fields.png       the lead design on D (R1t's binary design) in the
                               layout of Zhao Figs. 8 and 11: density, |u| on
                               h/2, T on h/8, and R1u's temperature difference

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
from scipy import ndimage  # noqa: E402

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


def note(ax, text: str, y: float = -0.125) -> None:
    ax.text(0.04, y, text, transform=ax.transAxes, ha="left", va="top",
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
    lead = json.loads((res / "zhao2d_r1t.json").read_text(encoding="utf-8"))["cells"]["new/check"]
    filled = json.loads((res / "zhao2d_r1u.json").read_text(encoding="utf-8"))["cells"]["filled/check"]
    term = meta["terminal"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.14, wspace=0.28)

    # (a) against the paper, on the paper-interpreted scale
    ax = left
    xlo, xhi, ylo, yhi = 0.25, 0.70, 1.05, 2.25
    for j in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25):
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
    # the current lead, a binary design on D's finer meshes; R1u's falls on the same point
    lx, ly = lead["psi"] / PAPER_PSI_0, lead["compliance"] / PAPER_C_0
    fx, fy = filled["psi"] / PAPER_PSI_0, filled["compliance"] / PAPER_C_0
    if max(abs(fx - lx), abs(fy - ly)) > 0.005:
        raise RuntimeError("R1u's point no longer falls on R1t's; give it its own marker")
    ax.plot([lx], [ly], "s", ms=7.5, color=INK, mec=SURFACE, mew=1.2,
            label="R1t's binary design on D (flow h/2, thermal h/8); R1u's coincides")
    ax.text(lx + 0.012, ly, f"J {0.5 * (lx + ly):.3f}", fontsize=8, color=INK, va="center")
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
             "finer thermal mesh alone. The square is the current lead,\na binary design on "
             "the check layer's finer meshes: placed, not compared. (b) Differences between "
             "meshes, not errors against an exact solution.", fontsize=8, color=INK2,
             ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1d_main.json, zhao2d_r1g_dual.json, "
           "zhao2d_r1h_matrix.json, zhao2d_r1i_h8.json, zhao2d_r1t.json, zhao2d_r1u.json and "
           "Zhao et al. Tables 4 and 7 — scripts/zhao2d_figures.py")
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


# -- figure 5: the R1k terminal check ---------------------------------------------------


def terminal_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1k_terminal_check.json").read_text(encoding="utf-8"))
    f = np.load(res / "zhao2d_r1k_terminal_fields.npz")
    centres = np.load(res / "zhao2d_r1k_fields.npz")["elem_centres"]
    coords_8 = np.load(res / "zhao2d_r1i_fields.npz")["thermal_node_coords_h8"]
    h = g["element_size"]
    t_cont, t_bin = f["temperature_h8_continuous"], f["temperature_h8_binary"]
    if len(coords_8) != len(t_cont):
        raise RuntimeError("the h/8 node coordinates do not match the terminal check's")
    tri_8 = triangulation(coords_8, h / 8, g)
    cells = rec["cells"]
    t_top = float(max(t_cont.max(), t_bin.max()))

    fig = plt.figure(figsize=(11.0, 7.6))
    top = fig.add_gridspec(1, 3, left=0.02, right=0.62, top=0.86, bottom=0.21, wspace=0.10)
    side = fig.add_gridspec(1, 1, left=0.71, right=0.97, top=0.80, bottom=0.34)

    cb = cells["x30/binary/h8"]
    ax = fig.add_subplot(top[0, 0])
    field_axes(ax, g, "(a) x₃₀ thresholded at s = 0.5")
    xe, ye, grid = density_grid(f["solid_fraction_binary_x30"], centres, h, g)
    m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                      shading="flat")
    colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
    note(ax, f"no repair; one connected fluid domain\nv_f (design domain) "
             f"{cb['v_f_design_domain']:.4f}: {cb['constraint_g']:+.1%} over\nthe 0.40 bound "
             f"(continuous: {cells['x30/continuous/h8']['v_f_design_domain']:.4f})", y=-0.17)
    for k, (t, key, title) in enumerate(((t_cont, "x30/continuous/h8", "(b) x₃₀ continuous, T on h/8"),
                                         (t_bin, "x30/binary/h8", "(c) x₃₀ thresholded, T on h/8"))):
        ax = fig.add_subplot(top[0, 1 + k])
        field_axes(ax, g, title)
        m = ax.tripcolor(tri_8, t, cmap=HEAT, vmin=0, vmax=t_top, shading="gouraud")
        colorbar(fig, m, ax, "T  (one scale for b and c)")
        c = cells[key]
        note(ax, f"flow re-solved on h for this s\nT_max {c['T_max']:.2f}\n"
                 f"C = {c['compliance']:.0f};  Ψ = {c['psi']:.5f}", y=-0.17)

    ax = fig.add_subplot(side[0, 0])
    rows = [("continuous, h/4", "continuous", 4), ("continuous, h/8", "continuous", 8),
            ("thresholded, h/4", "binary", 4), ("thresholded, h/8", "binary", 8)]
    for i, (label, version, level) in enumerate(rows):
        y = len(rows) - 1 - i
        a = cells[f"x300/{version}/h{level}"]["J"]
        b = cells[f"x30/{version}/h{level}"]["J"]
        ax.plot([a, b], [y, y], color=AXIS, lw=1.2, zorder=1)
        ax.plot([a], [y], "o", color=MUTED, ms=8, zorder=2)
        ax.plot([b], [y], "o", color=SERIES[0], ms=8, zorder=3)
        gain = rec["gain_x30_over_x300"][f"{version}/h{level}"]["J"]
        ax.text(max(a, b) + 0.02, y, f"{gain:+.1%}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0.9, 1.7)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("J on R1k's scale (fixed Ψ₀, C₀, w = 0.5)")
    ax.set_title("(d) J of x₃₀₀ and x₃₀, same evaluation", loc="left", fontsize=9.5)
    ax.plot([], [], "o", color=MUTED, ms=7, label="x₃₀₀, the start")
    ax.plot([], [], "o", color=SERIES[0], ms=7, label="x₃₀, after R1k")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=1, frameon=False,
              fontsize=8)

    g_c8 = rec["gain_x30_over_x300"]["continuous/h8"]["J"]
    g_b4 = rec["gain_x30_over_x300"]["binary/h4"]["J"]
    fig.suptitle("R1k terminal check — the gain holds on the finer thermal mesh and does not "
                 "survive thresholding", x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955,
             f"Continuous designs: x₃₀ beats x₃₀₀ by {-g_c8:.1%} in J on h/8 (10.9% on h/4). "
             f"Thresholded at s = 0.5, x₃₀ is {g_b4:+.1%} worse on h/4 — and exceeds the "
             "fluid-fraction bound.\nThe gain came with more grey (6.4% → 9.5% of cells), and "
             "the thresholded design does not keep it. Every state passed the 10⁻⁸ gate; the three "
             "recomputed anchors reproduce R1i, R1j and R1k.", fontsize=8.5, color=INK2,
             ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1k_terminal_check.json, "
           "results/zhao2d_r1k_terminal_fields.npz and the h/8 node coordinates in "
           "results/zhao2d_r1i_fields.npz — scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1k_terminal_check.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 6: R1l, qualified binary designs ---------------------------------------------


def r1l_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    a = json.loads((res / "zhao2d_r1l_baselines.json").read_text(encoding="utf-8"))
    b = json.loads((res / "zhao2d_r1l_vp_pilot.json").read_text(encoding="utf-8"))
    h8 = json.loads((res / "zhao2d_r1l_h8_check.json").read_text(encoding="utf-8"))["cells"]
    r1k = json.loads((res / "zhao2d_r1k_warm_start.json").read_text(encoding="utf-8"))
    fa = np.load(res / "zhao2d_r1l_baselines_fields.npz")
    fb = np.load(res / "zhao2d_r1l_vp_pilot_fields.npz")
    h = g["element_size"]
    ex = b["export"]

    fig = plt.figure(figsize=(11.0, 10.2))
    top = fig.add_gridspec(1, 4, left=0.02, right=0.99, top=0.875, bottom=0.40, wspace=0.10)
    low = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.27, bottom=0.10, wspace=0.55)

    panels = (
        (fa["x300_solid_fraction_binary"], fa["elem_centres"],
         "(a) x₃₀₀ (R1d), qualified binary",
         f"t = {a['designs']['x300']['t']:.4f}, 2000 fluid cells\nJ = {a['designs']['x300']['J']:.4f}"),
        (fa["x30_solid_fraction_binary"], fa["elem_centres"],
         "(b) x₃₀ (R1k), qualified binary",
         f"t = {a['designs']['x30']['t']:.4f}, 2000 fluid cells\nJ = {a['designs']['x30']['J']:.4f}"),
        (fb["solid_fraction"], fb["elem_centres"],
         "(c) R1l B terminal, continuous",
         f"volume-preserving, β = 16\ngrey {b['terminal']['grey_fraction']:.1%}; J = {b['terminal']['J_self']:.4f}"),
        (fb["solid_fraction_binary"], fb["elem_centres"],
         "(d) R1l B terminal, qualified binary",
         f"t = {ex['t']:.4f}, 2000 fluid cells\nJ = {ex['J']:.4f}"),
    )
    for k, (s, centres, title, text) in enumerate(panels):
        ax = fig.add_subplot(top[0, k])
        field_axes(ax, g, title)
        xe, ye, grid = density_grid(s, centres, h, g)
        m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                          shading="flat")
        colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
        note(ax, text + "\nthermal h/4, flow h")

    hist = b["history"]
    it = np.array([r["iteration"] for r in hist] + [b["terminal"]["iteration"]])
    jv = np.array([r["J_self"] for r in hist] + [b["terminal"]["J_self"]])
    gv = np.array([r["constraint_g"] for r in hist] + [b["terminal"]["constraint_g"]])
    sub = low[0, 0].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(it, jv, color=SERIES[0], lw=1.6)
    ax.plot([it[-1]], [jv[-1]], "o", color=SERIES[0], ms=5)
    ax.axhline(r1k["terminal"]["J_self"], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(it[-1] * 0.45, r1k["terminal"]["J_self"] + 0.004, "R1k's tanh terminal, same x₃₀",
            fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xlim(0, it[-1])
    ax.set_ylabel("continuous J")
    ax.set_xticklabels([])
    ax.set_title("(e) R1l B: 30 updates on the new projection", loc="left", fontsize=9.5)
    ax2 = fig.add_subplot(sub[1])
    ax2.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax2.plot(it, gv, color=INK2, lw=1.4)
    ax2.set_xlim(0, it[-1])
    ax2.set_ylabel("volume g")
    ax2.set_xlabel("MMA update (30 = terminal, re-evaluated)")
    ax2.text(1.2, gv[0], f"zero step g = +{gv[0]:.4f}", fontsize=7.5, color=INK2, va="center")

    ax = fig.add_subplot(low[0, 1])
    rows = [("x₃₀₀ (R1d)", 1.116472423534761, a["designs"]["x300"]["J"], h8["x300/h8"]["J"]),
            ("x₃₀ (R1k)", r1k["terminal"]["J_self"], a["designs"]["x30"]["J"], h8["x30/h8"]["J"]),
            ("R1l B terminal", b["terminal"]["J_self"], ex["J"], h8["pilot/h8"]["J"])]
    for i, (label, cont, binj, bin8) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([cont, bin8], [y, y], color=AXIS, lw=1.2, zorder=1)
        ax.plot([cont], [y], "o", color=MUTED, ms=8, zorder=2)
        ax.plot([binj], [y], "o", color=SERIES[0], ms=8, zorder=3)
        ax.plot([bin8], [y], "o", color=SERIES[1], ms=8, zorder=3)
        ax.text(binj, y + 0.2, f"{binj:.4f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
        ax.text(bin8 + 0.012, y, f"{bin8:.4f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(0.95, 1.58)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("J on the h/4 model's scale")
    ax.set_title("(f) Continuous and qualified binary J", loc="left", fontsize=9.5)
    ax.plot([], [], "o", color=MUTED, ms=7, label="continuous, h/4")
    ax.plot([], [], "o", color=SERIES[0], ms=7, label="qualified binary, h/4")
    ax.plot([], [], "o", color=SERIES[1], ms=7, label="qualified binary, h/8")
    ax.legend(loc="upper center", bbox_to_anchor=(0.40, -0.2), ncol=3, frameon=False,
              fontsize=8)

    vs = b["binary_against_baselines"]
    fig.suptitle("R1l — qualified binary baselines, and 30 updates on the volume-preserving "
                 "projection", x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955,
             f"The new terminal's qualified binary design has J {vs['x300']['J']:+.2%} against "
             f"x₃₀₀'s and {vs['x30']['J']:+.2%} against x₃₀'s; x₃₀'s own was "
             f"{a['x30_against_x300']['J']:+.2%} against x₃₀₀'s. On thermal h/8 the ranking holds: "
             f"{h8['pilot/h8']['J'] / h8['x300/h8']['J'] - 1:+.2%} and "
             f"{h8['pilot/h8']['J'] / h8['x30/h8']['J'] - 1:+.2%}.\nBudget used, not converged; "
             "the binary designs are 0/1 material with finite Brinkman resistance, solved as "
             "given s. Every state passed the 10⁻⁸ gate.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1l_baselines.json, zhao2d_r1l_vp_pilot.json, "
           "zhao2d_r1l_h8_check.json, their fields files and zhao2d_r1k_warm_start.json — "
           "scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1l.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 7: R1m, the flow mesh under two qualified binary designs ---------------------

# two hues and a neutral midpoint, for a signed difference
DIVERGE = LinearSegmentedColormap.from_list(
    "diverge", ["#184f95", "#6da7ec", "#ebeae4", "#f19e75", "#9c3c14"])


def dot_rows(ax, cells: dict, key: str, rows: list) -> tuple[float, float]:
    """x₃₀₀ and the pilot on one row per flow mesh; returns the data range."""
    values = []
    for i, (_, row) in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = cells[f"x300/{row}"][key], cells[f"pilot/{row}"][key]
        values += [a, b]
        ax.plot([a, b], [y, y], color=AXIS, lw=1.2, zorder=1)
        ax.plot([a], [y], "o", color=MUTED, ms=8, zorder=2)
        ax.plot([b], [y], "o", color=SERIES[0], ms=8, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return min(values), max(values)


def r1m_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1m_flow_check.json").read_text(encoding="utf-8"))
    f_h = np.load(res / "zhao2d_r1l_h8_fields.npz")
    f_2 = np.load(res / "zhao2d_r1m_fields.npz")
    coords_8 = np.load(res / "zhao2d_r1i_fields.npz")["thermal_node_coords_h8"]
    h = g["element_size"]
    cells, rank = rec["cells"], rec["ranking"]["pilot_against_x300"]
    temps = {name: (f_h[f"{name}_temperature_h8"], f_2[f"{name}_temperature_flow_h2_thermal_h8"])
             for name in ("x300", "pilot")}
    for name, (a, b) in temps.items():
        if not len(a) == len(b) == len(coords_8):
            raise RuntimeError(f"{name}: the temperatures are not on the h/8 mesh")
    tri_8 = triangulation(coords_8, h / 8, g)
    delta = {name: b - a for name, (a, b) in temps.items()}
    lim = float(max(np.abs(d).max() for d in delta.values()))

    fig = plt.figure(figsize=(11.0, 7.6))
    top = fig.add_gridspec(1, 3, left=0.02, right=0.62, top=0.86, bottom=0.21, wspace=0.10)
    side = fig.add_gridspec(2, 1, left=0.71, right=0.97, top=0.82, bottom=0.27, hspace=0.95)

    c = cells["pilot/flow_h2"]
    ax = fig.add_subplot(top[0, 0])
    field_axes(ax, g, "(a) Pilot's terminal, flow h/2")
    m = ax.tripcolor(tri_8, temps["pilot"][1], cmap=HEAT, vmin=0, shading="gouraud")
    colorbar(fig, m, ax, "T on thermal h/8")
    note(ax, f"flow solved on h/2 for the given s\nT_max {c['T_max']:.2f};  C = {c['compliance']:.0f}"
             f"\nΨ = {c['psi']:.5f};  D_T/Q {c['D_T_over_Q']:.1%}", y=-0.17)
    for k, (name, title) in enumerate((("pilot", "(b) Pilot: flow h/2 minus flow h"),
                                       ("x300", "(c) x₃₀₀: flow h/2 minus flow h"))):
        ax = fig.add_subplot(top[0, 1 + k])
        field_axes(ax, g, title)
        m = ax.tripcolor(tri_8, delta[name], cmap=DIVERGE, vmin=-lim, vmax=lim, shading="gouraud")
        colorbar(fig, m, ax, "ΔT  (one scale for b and c)")
        r = rec["flow_replacement"][name]
        note(ax, f"same s, same thermal h/8 mesh\nC {r['compliance']:+.2%};  Ψ {r['psi']:+.2%};  "
                 f"J {r['J']:+.2%}\nT_max {r['T_max']:+.2%}", y=-0.17)

    rows = [("flow h", "flow_h"), ("flow h/2", "flow_h2")]
    ax = fig.add_subplot(side[0, 0])
    lo, hi = dot_rows(ax, cells, "J", rows)
    pad = 0.35 * (hi - lo)
    ax.set_xlim(lo - pad, hi + 1.2 * pad)
    for i, (_, row) in enumerate(rows):
        y = len(rows) - 1 - i
        right = max(cells[f"x300/{row}"]["J"], cells[f"pilot/{row}"]["J"])
        ax.annotate(f"pilot {rank[row]['J']:+.2%}", (right, y), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("J on the h/4 model's scale")
    ax.set_title("(d) J, both on thermal h/8", loc="left", fontsize=9.5)
    ax.plot([], [], "o", color=MUTED, ms=7, label="x₃₀₀'s baseline")
    ax.plot([], [], "o", color=SERIES[0], ms=7, label="pilot's terminal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.42), ncol=2, frameon=False, fontsize=8)

    ax = fig.add_subplot(side[1, 0])
    _, hi = dot_rows(ax, cells, "D_T_over_Q", rows)
    ax.set_xlim(0.0, 1.15 * hi)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("D_T / Q, the discrete heat-balance deficit")
    ax.set_title("(e) Heat-balance deficit, thermal h/8", loc="left", fontsize=9.5)

    fig.suptitle("R1m — the two qualified binary designs with the flow solved on h/2",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955,
             f"Pilot against x₃₀₀, qualified to qualified: J {rank['flow_h']['J']:+.2%} on flow h, "
             f"{rank['flow_h2']['J']:+.2%} on flow h/2 (thermal h/8 both). D_T/Q: x₃₀₀ "
             f"{cells['x300/flow_h']['D_T_over_Q']:.1%} → {cells['x300/flow_h2']['D_T_over_Q']:.1%}, "
             f"pilot {cells['pilot/flow_h']['D_T_over_Q']:.1%} → "
             f"{cells['pilot/flow_h2']['D_T_over_Q']:.1%}.\nThe binary designs are given s, the "
             "material copied from the parent element; two flow meshes are not an exact solution. "
             "Every state passed the 10⁻⁸ gate.", fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1m_flow_check.json, zhao2d_r1m_fields.npz, "
           "zhao2d_r1l_h8_fields.npz and the h/8 node coordinates in zhao2d_r1i_fields.npz — "
           "scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1m.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 8: R1n, one beta = 32 stage and its binary design on two layers ---------------

def r1n_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1n_beta32.json").read_text(encoding="utf-8"))
    pilot = json.loads((res / "zhao2d_r1l_vp_pilot.json").read_text(encoding="utf-8"))
    fb = np.load(res / "zhao2d_r1l_vp_pilot_fields.npz")
    fn = np.load(res / "zhao2d_r1n_fields.npz")
    h = g["element_size"]
    cells, ex, term = rec["cells"], rec["export"], rec["terminal"]
    differ = int(np.sum(fb["solid_fraction_binary"] != fn["solid_fraction_binary"]))

    fig = plt.figure(figsize=(11.0, 10.2))
    top = fig.add_gridspec(1, 3, left=0.03, right=0.95, top=0.875, bottom=0.40, wspace=0.14)
    low = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.27, bottom=0.10, wspace=0.55)

    panels = (
        (fb["solid_fraction_binary"], "(a) The pilot (R1l, β = 16), qualified binary",
         f"t = {pilot['export']['t']:.4f}, 2000 fluid cells\nshown for the start; R1n starts "
         "from its raw x"),
        (fn["solid_fraction"], "(b) After R1n, continuous (β = 32)",
         f"30 updates, not converged\ngrey {term['grey_fraction']:.1%}; J = {term['J_self']:.4f}"),
        (fn["solid_fraction_binary"], "(c) After R1n, qualified binary",
         f"t = {ex['t']:.4f}, 2000 fluid cells, connected\n{differ} cells differ from (a)"),
    )
    for k, (s, title, text) in enumerate(panels):
        ax = fig.add_subplot(top[0, k])
        field_axes(ax, g, title)
        xe, ye, grid = density_grid(s, fn["elem_centres"], h, g)
        m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                          shading="flat")
        colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
        note(ax, text, y=-0.165)

    hist = rec["history"]
    it = np.array([r["iteration"] for r in hist] + [term["iteration"]])
    jv = np.array([r["J_self"] for r in hist] + [term["J_self"]])
    gv = np.array([r["constraint_g"] for r in hist] + [term["constraint_g"]])
    sub = low[0, 0].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(it, jv, color=SERIES[1], lw=1.6)
    ax.plot([it[-1]], [jv[-1]], "o", color=SERIES[1], ms=5)
    ax.axhline(pilot["terminal"]["J_self"], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.98, 0.95, "dashed: the pilot's β = 16 terminal,\nthe same x as update 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, it[-1])
    ax.set_ylabel("continuous J")
    ax.set_xticklabels([])
    ax.set_title("(d) R1n: 30 updates at β = 32", loc="left", fontsize=9.5)
    ax2 = fig.add_subplot(sub[1])
    ax2.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax2.plot(it, gv, color=INK2, lw=1.4)
    ax2.set_xlim(0, it[-1])
    ax2.set_ylabel("volume g")
    ax2.set_xlabel("MMA update (30 = terminal, re-evaluated)")

    ax = fig.add_subplot(low[0, 1])
    rows = [("development\nflow h, thermal h/4", "development"),
            ("check\nflow h/2, thermal h/8", "check")]
    colours = (("x300", MUTED, "x₃₀₀'s baseline"), ("pilot", SERIES[0], "the pilot"),
               ("new", SERIES[1], "after R1n"))
    values = []
    for i, (_, layer) in enumerate(rows):
        y = len(rows) - 1 - i
        js = [cells[f"{name}/{layer}"]["J"] for name, _, _ in colours]
        values += js
        ax.plot([min(js), max(js)], [y, y], color=AXIS, lw=1.2, zorder=1)
        for (name, colour, _), jj in zip(colours, js):
            ax.plot([jj], [y], "o", color=colour, ms=8, zorder=3 if name == "new" else 2)
        v = rec["against"][layer]["pilot"]["J"]
        ax.annotate(f"{v:+.2%} vs the pilot", (min(js), y), xytext=(0, 11),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8, color=INK)
    lo, hi = min(values), max(values)
    pad = 0.18 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("J on the h/4 model's scale, qualified binary designs")
    ax.set_title("(e) The three binary designs on both layers", loc="left", fontsize=9.5)
    for _, colour, label in colours:
        ax.plot([], [], "o", color=colour, ms=7, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.40, -0.22), ncol=3, frameon=False, fontsize=8)

    dev, chk = rec["against"]["development"], rec["against"]["check"]
    fig.suptitle("R1n — one β = 32 stage from the pilot, its binary design checked on two layers",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955,
             f"The new qualified binary design has J {dev['pilot']['J']:+.2%} against the pilot's on "
             f"the development layer and {chk['pilot']['J']:+.2%} on the check layer — lower C, "
             f"higher Ψ — and {dev['x300']['J']:+.2%} / {chk['x300']['J']:+.2%} against x₃₀₀'s, both "
             "objectives lower.\nBudget used, not converged; the continuous–binary gap is "
             f"{rec['responses']['export_gap']['J']:+.1%}. Every state passed the 10⁻⁸ gate.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1n_beta32.json, zhao2d_r1n_fields.npz, "
           "zhao2d_r1l_vp_pilot.json and its fields file — scripts/zhao2d_figures.py; "
           "no state is re-solved.")
    path = out / "zhao2d_r1n.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 9: R1o, thirty more updates at beta = 32 from R1n --------------------------

def r1o_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1o.json").read_text(encoding="utf-8"))
    r1n = json.loads((res / "zhao2d_r1n_beta32.json").read_text(encoding="utf-8"))
    fn = np.load(res / "zhao2d_r1n_fields.npz")
    fo = np.load(res / "zhao2d_r1o_fields.npz")
    h = g["element_size"]
    cells, ex, term = rec["cells"], rec["export"], rec["terminal"]

    fig = plt.figure(figsize=(11.0, 10.2))
    top = fig.add_gridspec(1, 3, left=0.03, right=0.95, top=0.875, bottom=0.40, wspace=0.14)
    low = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.27, bottom=0.10, wspace=0.55)

    panels = (
        (fn["solid_fraction_binary"], "(a) R1n's qualified binary design",
         f"t = {r1n['export']['t']:.4f}, 2000 fluid cells\nshown for the start; R1o starts "
         "from R1n's raw x"),
        (fo["solid_fraction"], "(b) After R1o, continuous (β = 32)",
         f"30 more updates, not converged\ngrey {term['grey_fraction']:.1%}; J = {term['J_self']:.4f}"),
        (fo["solid_fraction_binary"], "(c) After R1o, qualified binary",
         f"t = {ex['t']:.4f}, 2000 fluid cells, connected\n"
         f"{ex['cells_differing_from']['r1n']} cells differ from (a)"),
    )
    for k, (s, title, text) in enumerate(panels):
        ax = fig.add_subplot(top[0, k])
        field_axes(ax, g, title)
        xe, ye, grid = density_grid(s, fo["elem_centres"], h, g)
        m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                          shading="flat")
        colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
        note(ax, text, y=-0.165)

    hist = rec["history"]
    it = np.array([r["iteration"] for r in hist] + [term["iteration"]])
    jv = np.array([r["J_self"] for r in hist] + [term["J_self"]])
    gv = np.array([r["constraint_g"] for r in hist] + [term["constraint_g"]])
    sub = low[0, 0].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(it, jv, color=SERIES[2], lw=1.6)
    ax.plot([it[-1]], [jv[-1]], "o", color=SERIES[2], ms=5)
    ax.axhline(r1n["terminal"]["J_self"], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.98, 0.95, "dashed: R1n's continuous terminal,\nreproduced by update 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, it[-1])
    ax.set_ylabel("continuous J")
    ax.set_xticklabels([])
    ax.set_title("(d) R1o: 30 more updates at β = 32", loc="left", fontsize=9.5)
    ax2 = fig.add_subplot(sub[1])
    ax2.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax2.plot(it, gv, color=INK2, lw=1.4)
    ax2.set_xlim(0, it[-1])
    ax2.set_ylabel("volume g")
    ax2.set_xlabel("MMA update (30 = terminal, re-evaluated)")

    ax = fig.add_subplot(low[0, 1])
    rows = [("development\nflow h, thermal h/4", "development"),
            ("check\nflow h/2, thermal h/8", "check")]
    colours = (("pilot", SERIES[0], "the R1l pilot"), ("r1n", SERIES[1], "R1n's design"),
               ("new", SERIES[2], "after R1o"))
    values = []
    for i, (_, layer) in enumerate(rows):
        y = len(rows) - 1 - i
        js = [cells[f"{name}/{layer}"]["J"] for name, _, _ in colours]
        values += js
        ax.plot([min(js), max(js)], [y, y], color=AXIS, lw=1.2, zorder=1)
        for (name, colour, _), jj in zip(colours, js):
            ax.plot([jj], [y], "o", color=colour, ms=8, zorder=3 if name == "new" else 2)
        v = rec["against"][layer]["r1n"]["J"]
        ax.annotate(f"{v:+.2%} vs R1n's", (cells[f"new/{layer}"]["J"], y), xytext=(0, 11),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    lo, hi = min(values), max(values)
    pad = 0.18 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_ylim(-0.7, len(rows) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("J on the h/4 model's scale, qualified binary designs")
    ax.set_title("(e) The three candidates on both layers", loc="left", fontsize=9.5)
    for _, colour, label in colours:
        ax.plot([], [], "o", color=colour, ms=7, label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.40, -0.22), ncol=3, frameon=False, fontsize=8)

    dev, chk = rec["against"]["development"], rec["against"]["check"]
    fig.suptitle("R1o — thirty more updates at β = 32 from R1n; the two layers disagree",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955,
             f"The new qualified binary design has J {dev['r1n']['J']:+.2%} against R1n's on the "
             f"development layer (Ψ and C both higher) and {chk['r1n']['J']:+.2%} on the check layer "
             f"(lower C, higher Ψ); against the pilot {dev['pilot']['J']:+.2%} / {chk['pilot']['J']:+.2%}."
             f"\nThe continuous J changed {rec['responses']['optimisation']['J']:+.2%}; the export gap is "
             f"{rec['responses']['export_gap']['J']:+.1%}. Budget used, not converged. Every state "
             "passed the 10⁻⁸ gate.", fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1o.json, zhao2d_r1o_fields.npz, zhao2d_r1n_beta32.json "
           "and its fields file — scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1o.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 10: R1p, where along the path R1n's and R1o's order flips ----------------------

def r1p_figure(res: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1p_bridge.json").read_text(encoding="utf-8"))
    layers = ("A", "B", "D")
    labels = ("A\nflow h, thermal h/4", "B\nflow h, thermal h/8", "D\nflow h/2, thermal h/8")
    diff = rec["r1o_minus_r1n"]
    x = np.arange(len(layers))

    fig = plt.figure(figsize=(11.0, 5.2))
    grid = fig.add_gridspec(1, 2, left=0.08, right=0.97, top=0.78, bottom=0.24, wspace=0.32)

    ax = fig.add_subplot(grid[0, 0])
    width = 0.3
    diss = [diff[k]["dissipation_term"] for k in layers]
    therm = [diff[k]["thermal_term"] for k in layers]
    total = [diff[k]["dJ"] for k in layers]
    ax.bar(x - width / 2 - 0.02, diss, width, color=MUTED, label="dissipation term, 0.5 ΔΨ/Ψ₀")
    ax.bar(x + width / 2 + 0.02, therm, width, color=INK2, label="thermal term, 0.5 ΔC/C₀")
    ax.plot(x, total, color=INK, lw=1.2, zorder=3)
    ax.plot(x, total, "o", color=INK, ms=8, zorder=4, label="ΔJ, their sum")
    for xi, t in zip(x, total):
        ax.annotate(f"{t:+.4f}", (xi, t), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK)
    ax.axhline(0.0, color=AXIS, lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.6, len(layers) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("R1o minus R1n, on the h/4 scale")
    ax.text(0.02, 0.03, "above 0: R1n ahead\nbelow 0: R1o ahead", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_title("(a) The difference between the designs, split into its terms", loc="left",
                 fontsize=9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)

    ax = fig.add_subplot(grid[0, 1])
    for name, colour, label in (("r1n", SERIES[1], "R1n's design"), ("r1o", SERIES[2], "R1o's design")):
        c = [rec["layers"][name][k]["compliance"] for k in layers]
        ax.plot(x, c, color=colour, lw=2, zorder=2)
        ax.plot(x, c, "o", color=colour, ms=8, zorder=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.4, len(layers) - 0.6)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("thermal compliance C")
    ax.set_title("(b) Each design's C along the path", loc="left", fontsize=9.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=8)

    steps = rec["steps"]
    fig.suptitle("R1p — along A → B → D, the order of R1n's and R1o's designs flips in the flow "
                 "replacement", x=0.02, ha="left", fontsize=11.5, color=INK, y=0.97)
    fig.text(0.02, 0.905,
             f"Refining the temperature (A → B) moves ΔJ by {steps['thermal_refinement_A_to_B']['dJ']:+.4f}, "
             f"which narrows R1n's lead; replacing the flow (B → D) moves it by "
             f"{steps['flow_replacement_B_to_D']['dJ']:+.4f}, which flips it. One path only, not the "
             "full interaction.\nThe binary designs are given s; every state passed the 10⁻⁸ gate.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1p_bridge.json — scripts/zhao2d_figures.py; "
           "no state is re-solved.")
    path = out / "zhao2d_r1p.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 11: R1r, twenty updates on the check layer D ----------------------------------

def r1r_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1r.json").read_text(encoding="utf-8"))
    fo = np.load(res / "zhao2d_r1o_fields.npz")
    fr = np.load(res / "zhao2d_r1r_fields.npz")
    h = g["element_size"]
    ex, term, hist = rec["export"], rec["terminal"], rec["history"]
    new = rec["cells"].get("new/check")
    scale = rec["scale"]

    fig = plt.figure(figsize=(11.0, 10.2))
    top = fig.add_gridspec(1, 3, left=0.03, right=0.95, top=0.875, bottom=0.40, wspace=0.14)
    low = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.27, bottom=0.10, wspace=0.32)

    panels = [
        (fo["solid_fraction_binary"], fo["elem_centres"], "(a) R1o's qualified binary design",
         "the lead candidate on D; R1r starts\nfrom R1o's raw continuous x"),
        (fr["solid_fraction"], fr["design_elem_centres"], "(b) After R1r, continuous (β = 32)",
         f"{len(hist)} updates on D, not converged\ngrey {term['grey_fraction']:.1%}; "
         f"J = {term['J_common_scale']:.4f} on D"),
    ]
    if "solid_fraction_binary" in fr.files:
        panels.append((fr["solid_fraction_binary"], fr["design_elem_centres"],
                       "(c) After R1r, qualified binary" if ex.get("qualified")
                       else "(c) After R1r, exported, not qualified",
                       f"t = {ex['t']:.4f}, {ex['fluid_cells']} fluid cells, "
                       f"{'connected' if ex['connected'] else 'not connected'}\n"
                       f"{ex['cells_differing_from']['r1o']} cells differ from (a)"))
    for k, (s, centres, title, text) in enumerate(panels):
        ax = fig.add_subplot(top[0, k])
        field_axes(ax, g, title)
        xe, ye, grid = density_grid(s, centres, h, g)
        m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                          shading="flat")
        colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
        note(ax, text, y=-0.165)

    it = np.array([r["iteration"] for r in hist] + [term["iteration"]])
    jv = np.array([r["J_common_scale"] for r in hist] + [term["J_common_scale"]])
    gv = np.array([r["constraint_g"] for r in hist] + [term["constraint_g"]])
    sub = low[0, 0].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(it, jv, color=INK, lw=1.6)
    ax.plot([it[-1]], [jv[-1]], "o", color=INK, ms=5)
    ax.axhline(rec["contract"]["zero_step_anchor"]["J_common_scale"], color=MUTED, lw=0.8,
               ls=(0, (3, 2)))
    ax.text(0.98, 0.95, "dashed: R1o's continuous design on D (R1q),\nreproduced by update 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, it[-1])
    ax.set_ylabel("continuous J on D")
    ax.set_xticks(range(0, int(it[-1]) + 1, 5))
    ax.set_xticklabels([])
    ax.set_title(f"(d) R1r: {len(hist)} updates on D at β = 32", loc="left", fontsize=9.5)
    ax2 = fig.add_subplot(sub[1])
    ax2.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax2.plot(it, gv, color=INK2, lw=1.4)
    ax2.set_xlim(0, it[-1])
    ax2.set_xticks(range(0, int(it[-1]) + 1, 5))
    ax2.set_ylabel("volume g")
    ax2.set_xlabel(f"MMA update ({it[-1]} = terminal, re-evaluated)")

    # the qualified binary designs on D: Psi against C, with J's level lines
    ax = fig.add_subplot(low[0, 1])
    points = [(name, rec["candidates"][name], colour, label) for name, colour, label in (
        ("pilot", SERIES[0], "the R1l pilot"), ("r1n", SERIES[1], "R1n's design"),
        ("r1o", SERIES[2], "R1o's design"))]
    if new is not None:
        points.append(("new", new, INK, "after R1r"))
    psi = np.array([p[1]["psi"] for p in points])
    comp = np.array([p[1]["compliance"] for p in points])
    w = rec["contract"]["weight"]
    span = np.array([psi.min(), psi.max()])
    grid_psi = np.linspace(span[0] - 0.15 * np.ptp(span), span[1] + 0.15 * np.ptp(span), 2)
    for name, cell, colour, label in points:
        if name in ("r1o", "new"):
            level = cell["J"]
            ax.plot(grid_psi, (level - w * grid_psi / scale["psi_0"]) * scale["c_0"] / (1 - w),
                    color=colour, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.plot([cell["psi"]], [cell["compliance"]], "D" if name == "new" else "o", color=colour,
                ms=8, zorder=3, label=f"{label}: J = {cell['J']:.4f}")
    ax.set_xlim(grid_psi[0], grid_psi[1])
    pad = 0.15 * np.ptp(comp)
    ax.set_ylim(comp.min() - pad, comp.max() + pad)
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("dissipated power Ψ")
    ax.set_ylabel("thermal compliance C")
    ax.set_title("(e) The qualified binary designs on D", loc="left", fontsize=9.5)
    ax.text(0.98, 0.95, "dashed: equal J, through R1o's\nand the new design (common scale)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False, fontsize=8)

    opt = rec["responses"]["optimisation"]
    if new is not None:
        vs = rec["against"]["r1o"]
        head = (f"The new qualified binary design has J {vs['J']:+.2%} against R1o's on D "
                f"(Ψ {vs['psi']:+.2%}, C {vs['compliance']:+.2%}); against R1n's "
                f"{rec['against']['r1n']['J']:+.2%}, the pilot's {rec['against']['pilot']['J']:+.2%}."
                f"\nThe continuous J on D changed {opt['J']:+.2%}; the export gap on D is "
                f"{rec['responses']['export_gap']['J']:+.1%}.")
    else:
        head = (f"The continuous J on D changed {opt['J']:+.2%}; no qualified binary design was "
                f"analysed: {rec.get('stopped', '')}.\n")
    fig.suptitle("R1r — updates on the check layer D from R1o's design, at the common scale",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955, f"{head} Stop: {rec['stop']['stop_reason']}, not converged. Every "
             "state passed the 10⁻⁸ gate. D both generated and ranked this design.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1r.json, zhao2d_r1r_fields.npz and "
           "zhao2d_r1o_fields.npz — scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1r.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- figure 12: R1t, twenty more updates on D, on the linear thermal path ------------------

def r1t_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1t.json").read_text(encoding="utf-8"))
    rr = json.loads((res / "zhao2d_r1r.json").read_text(encoding="utf-8"))
    fr = np.load(res / "zhao2d_r1r_fields.npz")
    ft = np.load(res / "zhao2d_r1t_fields.npz")
    h = g["element_size"]
    ex, term, hist = rec["export"], rec["terminal"], rec["history"]
    new = rec["cells"].get("new/check")
    scale = rec["scale"]

    fig = plt.figure(figsize=(11.0, 10.2))
    top = fig.add_gridspec(1, 3, left=0.03, right=0.95, top=0.875, bottom=0.40, wspace=0.14)
    low = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.27, bottom=0.10, wspace=0.32)

    panels = [
        (fr["solid_fraction_binary"], fr["design_elem_centres"], "(a) R1r's qualified binary design",
         "the first choice on D; R1t starts\nfrom R1r's raw continuous x"),
        (ft["solid_fraction"], ft["design_elem_centres"], "(b) After R1t, continuous (β = 32)",
         f"{len(hist)} updates on D, not converged\ngrey {term['grey_fraction']:.1%}; "
         f"J = {term['J_common_scale']:.4f} on D"),
    ]
    if "solid_fraction_binary" in ft.files:
        panels.append((ft["solid_fraction_binary"], ft["design_elem_centres"],
                       "(c) After R1t, qualified binary" if ex.get("qualified")
                       else "(c) After R1t, exported, not qualified",
                       f"t = {ex['t']:.4f}, {ex['fluid_cells']} fluid cells, "
                       f"{'connected' if ex['connected'] else 'not connected'}\n"
                       f"{ex['cells_differing_from']['r1r']} cells differ from (a)"))
    for k, (s, centres, title, text) in enumerate(panels):
        ax = fig.add_subplot(top[0, k])
        field_axes(ax, g, title)
        xe, ye, grid = density_grid(s, centres, h, g)
        m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                          shading="flat")
        colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
        note(ax, text, y=-0.165)
        if k == 2:
            # fluid not joined to the main body: every fluid component but the largest
            labels, count = ndimage.label(np.nan_to_num(grid) > 0.5)
            sizes = np.bincount(labels.ravel())[1:]
            for lab in np.flatnonzero(sizes < sizes.max()) + 1:
                i, j = np.argwhere(labels == lab).mean(axis=0)
                xc, yc = (i + 0.5) * h, ye[0] + (j + 0.5) * h
                ax.plot([xc], [yc], "o", ms=11, mfc="none", mec=SURFACE, mew=2.2, zorder=4)
                ax.annotate("fluid cell isolated by shared edges\n(it touches the channel at a corner)",
                            xy=(xc, yc), xytext=(xc - 0.0010, yc + 0.0023),
                            fontsize=7.5, color=INK, ha="center", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"),
                            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    # R1r's updates, then R1t's: one objective on one model, MMA reinitialised between them
    n_r = len(rr["history"])
    it_r = np.array([r["iteration"] for r in rr["history"]] + [rr["terminal"]["iteration"]])
    j_r = np.array([r["J_common_scale"] for r in rr["history"]] + [rr["terminal"]["J_common_scale"]])
    g_r = np.array([r["constraint_g"] for r in rr["history"]] + [rr["terminal"]["constraint_g"]])
    it_t = n_r + np.array([r["iteration"] for r in hist] + [term["iteration"]])
    j_t = np.array([r["J_common_scale"] for r in hist] + [term["J_common_scale"]])
    g_t = np.array([r["constraint_g"] for r in hist] + [term["constraint_g"]])
    sub = low[0, 0].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(it_r, j_r, color=MUTED, lw=1.4)
    ax.plot(it_t, j_t, color=INK, lw=1.6)
    ax.plot([it_t[-1]], [j_t[-1]], "o", color=INK, ms=5)
    ax.axvline(n_r, color=AXIS, lw=0.8, ls=(0, (3, 2)))
    ax.text(n_r + 4.5, 0.95, "MMA reinitialised at 20;\nthe linear thermal path from there",
            transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=7.5, color=MUTED)
    ax.text(n_r - 0.4, 0.95, "R1r (Newton path)", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, it_t[-1])
    ax.set_ylabel("continuous J on D")
    ax.set_xticks(range(0, int(it_t[-1]) + 1, 5))
    ax.set_xticklabels([])
    ax.set_title(f"(d) R1r's 20 updates, then R1t's {len(hist)}, on D at β = 32", loc="left",
                 fontsize=9.5)
    ax2 = fig.add_subplot(sub[1])
    ax2.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax2.axvline(n_r, color=AXIS, lw=0.8, ls=(0, (3, 2)))
    ax2.plot(it_r, g_r, color=MUTED, lw=1.2)
    ax2.plot(it_t, g_t, color=INK2, lw=1.4)
    ax2.set_xlim(0, it_t[-1])
    ax2.set_xticks(range(0, int(it_t[-1]) + 1, 5))
    ax2.set_ylabel("volume g")
    ax2.set_xlabel("MMA update, counted across both runs (each terminal re-evaluated)")

    # the qualified binary designs on D: Psi against C, with J's level lines
    ax = fig.add_subplot(low[0, 1])
    points = [(name, rec["candidates"][name], colour, marker, label) for name, colour, marker, label in (
        ("pilot", SERIES[0], "o", "the R1l pilot"), ("r1n", SERIES[1], "o", "R1n's design"),
        ("r1o", SERIES[2], "o", "R1o's design"), ("r1r", INK, "D", "R1r's design"))]
    if new is not None:
        points.append(("new", new, INK, "s", "after R1t"))
    psi = np.array([p[1]["psi"] for p in points])
    comp = np.array([p[1]["compliance"] for p in points])
    w = rec["contract"]["weight"]
    span = np.array([psi.min(), psi.max()])
    grid_psi = np.linspace(span[0] - 0.15 * np.ptp(span), span[1] + 0.15 * np.ptp(span), 2)
    for name, cell, colour, marker, label in points:
        if name in ("r1r", "new"):
            level = cell["J"]
            ax.plot(grid_psi, (level - w * grid_psi / scale["psi_0"]) * scale["c_0"] / (1 - w),
                    color=colour, lw=0.8, ls=(0, (3, 2)) if name == "r1r" else (0, (1, 1.5)),
                    zorder=1)
        face = SURFACE if name == "new" else colour
        ax.plot([cell["psi"]], [cell["compliance"]], marker, color=colour, mfc=face, mew=1.6,
                ms=8, zorder=3, label=f"{label}: J = {cell['J']:.4f}")
    ax.set_xlim(grid_psi[0], grid_psi[1])
    pad = 0.15 * np.ptp(comp)
    ax.set_ylim(comp.min() - pad, comp.max() + pad)
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("dissipated power Ψ")
    ax.set_ylabel("thermal compliance C")
    ax.set_title("(e) The qualified binary designs on D", loc="left", fontsize=9.5)
    ax.text(0.98, 0.95, "equal J through R1r's design (dashed)\nand the new one (dotted)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, frameon=False, fontsize=8)

    opt = rec["responses"]["optimisation"]
    if new is not None:
        vs = rec["against"]["r1r"]
        head = (f"The new qualified binary design has J {vs['J']:+.2%} against R1r's on D "
                f"(Ψ {vs['psi']:+.2%}, C {vs['compliance']:+.2%}, T_max {vs['T_max']:+.2%}); "
                f"against R1o's {rec['against']['r1o']['J']:+.2%}."
                f"\nThe continuous J on D changed {opt['J']:+.2%}; the export gap on D is "
                f"{rec['responses']['export_gap']['J']:+.1%}.")
    else:
        head = (f"The continuous J on D changed {opt['J']:+.2%}; no qualified binary design was "
                f"analysed: {rec.get('stopped', '')}.\n")
    fig.suptitle("R1t — twenty more updates on D from R1r's design, on the linear thermal path",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.955, f"{head} Stop: {rec['stop']['stop_reason']}, not converged. Every "
             "state passed the 10⁻⁸ gate; MMA's state is saved to resume from.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1t.json, zhao2d_r1t_fields.npz, zhao2d_r1r.json and "
           "zhao2d_r1r_fields.npz — scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1t.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def mark_isolated(ax, grid: np.ndarray, ye: np.ndarray, h: float, text: str) -> int:
    """Circle every fluid component but the largest, by shared edges; returns how many."""
    labels, _ = ndimage.label(np.nan_to_num(grid) > 0.5)  # 4-connectivity: shared edges
    sizes = np.bincount(labels.ravel())[1:]
    small = np.flatnonzero(sizes < sizes.max()) + 1
    for k, lab in enumerate(small):
        i, j = np.argwhere(labels == lab).mean(axis=0)
        xc, yc = (i + 0.5) * h, ye[0] + (j + 0.5) * h
        ax.plot([xc], [yc], "o", ms=11, mfc="none", mec=SURFACE, mew=2.2, zorder=4)
        if k == 0:
            ax.annotate(text, xy=(xc, yc), xytext=(xc - 0.0010, yc + 0.0023), fontsize=7.5,
                        color=INK, ha="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"),
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    return len(small)


def r1v_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1v.json").read_text(encoding="utf-8"))
    rt = json.loads((res / "zhao2d_r1t.json").read_text(encoding="utf-8"))
    rr = json.loads((res / "zhao2d_r1r.json").read_text(encoding="utf-8"))
    ft = np.load(res / "zhao2d_r1t_fields.npz")
    fv = np.load(res / "zhao2d_r1v_fields.npz")
    h = g["element_size"]
    ex, term, hist = rec["export"], rec["terminal"], rec["history"]
    new = rec["cells"].get("new/check")
    scale = rec["scale"]

    fig = plt.figure(figsize=(11.0, 11.0))
    top = fig.add_gridspec(1, 3, left=0.03, right=0.95, top=0.865, bottom=0.425, wspace=0.14)
    low = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.295, bottom=0.135, wspace=0.32)

    panels = [
        (ft["solid_fraction_binary"], ft["design_elem_centres"], "(a) R1t's qualified binary design",
         "the numerical first of the evaluated designs; R1v\nresumes from R1t's raw x and MMA's state"),
        (fv["solid_fraction"], fv["design_elem_centres"], "(b) After R1v, continuous (β = 32)",
         f"{len(hist)} more updates on D, MMA's history kept\ngrey {term['grey_fraction']:.1%}; "
         f"J = {term['J_common_scale']:.4f} on D"),
    ]
    if "solid_fraction_binary" in fv.files:
        panels.append((fv["solid_fraction_binary"], fv["design_elem_centres"],
                       "(c) After R1v, qualified binary" if ex.get("qualified")
                       else "(c) After R1v, exported, not qualified",
                       f"t = {ex['t']:.4f}, {ex['fluid_cells']} fluid cells, "
                       f"{'connected' if ex['connected'] else 'not connected'}\n"
                       f"{ex['cells_differing_from']['r1t']} cells differ from (a)"))
    for k, (s, centres, title, text) in enumerate(panels):
        ax = fig.add_subplot(top[0, k])
        field_axes(ax, g, title)
        xe, ye, grid = density_grid(s, centres, h, g)
        m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                          shading="flat")
        colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
        note(ax, text, y=-0.165)
        if k == 0:
            mark_isolated(ax, grid, ye, h, "isolated by shared edges\n(R1u fills it)")
        elif k == 2:
            mark_isolated(ax, grid, ye, h, "isolated by shared edges\n(reported, not filled)")

    # R1r's updates, R1t's, then R1v's: MMA reinitialised at 20, carried over at 40
    n_r = len(rr["history"])
    runs = []
    for run, offset in ((rr, 0), (rt, n_r), (rec, n_r)):
        recs = run["history"] + [run["terminal"]]
        runs.append((offset + np.array([r["iteration"] for r in recs]),
                     np.array([r["J_common_scale"] for r in recs]),
                     np.array([r["constraint_g"] for r in recs])))
    (it_r, j_r, g_r), (it_t, j_t, g_t), (it_v, j_v, g_v) = runs
    carried = n_r + rec["history"][0]["iteration"]
    sub = low[0, 0].subgridspec(2, 1, height_ratios=[3, 1.2], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.plot(it_r, j_r, color=MUTED, lw=1.4)
    ax.plot(it_t, j_t, color=INK2, lw=1.4)
    ax.plot(it_v, j_v, color=INK, lw=1.6)
    ax.plot([it_v[-1]], [j_v[-1]], "o", color=INK, ms=5)
    for x in (n_r, carried):
        ax.axvline(x, color=AXIS, lw=0.8, ls=(0, (3, 2)))
    for x, text in ((n_r / 2, "R1r"), ((n_r + carried) / 2, "R1t"), ((carried + it_v[-1]) / 2, "R1v")):
        ax.text(x, 0.97, text, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color=INK2)
    peak = int(np.argmax(j_t))
    ax.annotate("J rose after MMA\nwas reinitialised", xy=(it_t[peak], j_t[peak]),
                xytext=(it_t[peak] + 2.2, j_t[peak] - 0.22 * (j_t[peak] - j_v.min())), fontsize=7.5,
                color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    # R1v's updates are small on this scale: zoomed, with R1t's last ones, in an inset
    zoom = ax.inset_axes([0.63, 0.24, 0.35, 0.50])
    keep = it_t >= carried - 4
    zoom.plot(it_t[keep], j_t[keep], color=INK2, lw=1.2)
    zoom.plot(it_v, j_v, color=INK, lw=1.4)
    zoom.plot([it_v[-1]], [j_v[-1]], "o", color=INK, ms=4)
    zoom.axvline(carried, color=AXIS, lw=0.8, ls=(0, (3, 2)))
    zoom.set_xlim(carried - 4, it_v[-1])
    zoom.yaxis.tick_right()
    zoom.tick_params(labelsize=6.5, length=2)
    zoom.grid(axis="y", color=GRID, lw=0.5)
    zoom.set_title("no jump in J after the append", fontsize=7, color=MUTED, pad=2)
    for side in ("top", "left"):
        zoom.spines[side].set_visible(False)
    ax.set_xlim(0, it_v[-1])
    ax.set_ylabel("continuous J on D")
    ax.set_xticks(range(0, int(it_v[-1]) + 1, 10))
    ax.set_xticklabels([])
    ax.set_title(f"(d) R1r's 20 updates, R1t's 20, then R1v's {len(hist)}, on D at β = 32",
                 loc="left", fontsize=9.5)
    ax2 = fig.add_subplot(sub[1])
    ax2.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    for x in (n_r, carried):
        ax2.axvline(x, color=AXIS, lw=0.8, ls=(0, (3, 2)))
    ax2.plot(it_r, g_r, color=MUTED, lw=1.2)
    ax2.plot(it_t, g_t, color=INK2, lw=1.2)
    ax2.plot(it_v, g_v, color=INK, lw=1.4)
    ax2.set_xlim(0, it_v[-1])
    ax2.set_xticks(range(0, int(it_v[-1]) + 1, 10))
    ax2.set_ylabel("volume g")
    ax2.set_xlabel("MMA update, counted across the three runs (each terminal re-evaluated)")

    # the qualified binary designs on D: Psi against C, with J's level lines
    ax = fig.add_subplot(low[0, 1])
    points = [(name, rec["candidates"][name], colour, marker, face, label)
              for name, colour, marker, face, label in (
                  ("pilot", SERIES[0], "o", None, "the R1l pilot"),
                  ("r1n", SERIES[1], "o", None, "R1n's design"),
                  ("r1o", SERIES[2], "o", None, "R1o's design"),
                  ("r1r", INK, "D", None, "R1r's design"),
                  ("r1t", INK2, "^", None, "R1t's design"),
                  ("r1u", INK2, "v", SURFACE, "R1t's, filled (R1u)"))]
    if new is not None:
        points.append(("new", new, INK, "s", SURFACE, "after R1v"))
    psi = np.array([p[1]["psi"] for p in points])
    comp = np.array([p[1]["compliance"] for p in points])
    w = rec["contract"]["weight"]
    span = np.array([psi.min(), psi.max()])
    grid_psi = np.linspace(span[0] - 0.15 * np.ptp(span), span[1] + 0.15 * np.ptp(span), 2)
    for name, cell, colour, marker, face, label in points:
        if name in ("r1t", "new"):
            ax.plot(grid_psi, (cell["J"] - w * grid_psi / scale["psi_0"]) * scale["c_0"] / (1 - w),
                    color=colour, lw=0.8, ls=(0, (3, 2)) if name == "r1t" else (0, (1, 1.5)),
                    zorder=1)
        ax.plot([cell["psi"]], [cell["compliance"]], marker, color=colour,
                mfc=face if face else colour, mew=1.6, ms=8, zorder=3,
                label=f"{label}: J = {cell['J']:.4f}")
    ax.set_xlim(grid_psi[0], grid_psi[1])
    pad = 0.15 * np.ptp(comp)
    ax.set_ylim(comp.min() - pad, comp.max() + pad)
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("dissipated power Ψ")
    ax.set_ylabel("thermal compliance C")
    ax.set_title("(e) The qualified binary designs on D", loc="left", fontsize=9.5)
    ax.text(0.98, 0.95, "equal J through R1t's design (dashed)\nand the new one (dotted);\n"
            "R1t's and R1u's nearly coincide",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.21), ncol=3, frameon=False,
              fontsize=7.5, columnspacing=1.2, handletextpad=0.4)

    opt = rec["responses"]["optimisation"]
    if new is not None:
        vs, vu, vr = rec["against"]["r1t"], rec["against"]["r1u"], rec["against"]["r1r"]
        head = (f"The terminal's qualified binary design has J {vs['J']:+.2%} against R1t's on D "
                f"(Ψ {vs['psi']:+.2%}, C {vs['compliance']:+.2%}, T_max {vs['T_max']:+.2%}); "
                f"against R1u's {vu['J']:+.2%}, against R1r's {vr['J']:+.2%}."
                f"\nThe continuous J on D changed {opt['J']:+.3%} over R1v's "
                f"{len(hist)} updates; the export gap on D is "
                f"{rec['responses']['export_gap']['J']:+.1%}. ")
    else:
        head = (f"The continuous J on D changed {opt['J']:+.3%}; no qualified binary design was "
                f"analysed: {rec.get('stopped', '')}.\n")
    fig.suptitle("R1v — R1t's run continued on D, MMA's history carried over (20 → "
                 f"{rec['stop']['mma_updates_in_all']} updates)",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.958, f"{head}Stop: {rec['stop']['stop_reason']}, not converged.\nThe first "
             "resumed evaluation reproduced R1t's terminal; every state passed the 10⁻⁸ gate.",
             fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.4)
    footer(fig, "Drawn from results/zhao2d_r1v.json, zhao2d_r1v_fields.npz, zhao2d_r1t.json, "
           "zhao2d_r1t_fields.npz and zhao2d_r1r.json — scripts/zhao2d_figures.py; no state is "
           "re-solved.")
    path = out / "zhao2d_r1v.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def r1w_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    rec = json.loads((res / "zhao2d_r1w.json").read_text(encoding="utf-8"))
    rv = json.loads((res / "zhao2d_r1v.json").read_text(encoding="utf-8"))
    fw = np.load(res / "zhao2d_r1w_fields.npz")
    ft = np.load(res / "zhao2d_r1t_fields.npz")
    h = g["element_size"]
    rank = rec["ranking"]
    rows = {row["label"]: row for row in rank["table"]}
    labels = sorted(rows)
    j_t = rows[20]["J"]  # R1t's binary design
    j_c = {p["iteration"]: p["continuous_J"] for p in rec["pool"]}
    iterates = sorted(j_c)
    masks = np.array([fw[f"binary_{k}"] for k in labels])
    varying = np.flatnonzero(~np.all(masks == masks[0], axis=0))
    # two bands in J, split at the largest gap; then the cells that are the same within each
    # band and differ between them (a description of the saved designs, not a cause)
    order = sorted(labels, key=lambda k: rows[k]["J"])
    cut = int(np.argmax(np.diff([rows[k]["J"] for k in order])))
    near, far = sorted(order[:cut + 1]), sorted(order[cut + 1:])
    m_near = masks[[labels.index(k) for k in near]]
    m_far = masks[[labels.index(k) for k in far]]
    split = np.flatnonzero(np.all(m_near == m_near[0], axis=0) & np.all(m_far == m_far[0], axis=0)
                           & (m_near[0] != m_far[0]))
    if len(split) != 1:
        raise RuntimeError(f"expected one cell to separate the two bands, found {split.tolist()}")
    cell = int(split[0])
    state = {1.0: "solid", 0.0: "fluid"}
    band = {k: SERIES[0] if k in near else SERIES[1] for k in labels}
    reused = {k for k in labels if rows[k]["source"] != "solved here"}
    r1v_label = next(k for k in labels if rows[k]["source"] == "r1v's recorded D state")
    # the text below states these; check them against the record rather than restate them
    fluid_cells = {p["fluid_cells"] for p in rec["pool"]}
    solves = rec["cost"]["solves"]
    if not (rank["lowest"] == 20 and rows[20]["source"] == "r1t's recorded D state"
            and rec["contract"]["weight"] == 0.5
            and solves["flow_made"] == solves["thermal_made"] == len(labels) - len(reused)
            and len(fluid_cells) == 1 and rec["contract"]["gate"] == 1e-8
            and all(r["isolated_cells"] == rows[20]["isolated_cells"] for r in rank["table"])
            and all(c["gate_passed"] for c in rec["cells"].values())
            and not any(c["failures"] for c in rec["checkpoints"])
            and solves["mma"] == 0 and solves["reverse_passes"] == 0):
        raise RuntimeError("the record no longer says what this figure's text states")

    fig = plt.figure(figsize=(11.0, 8.6))
    left = fig.add_gridspec(1, 1, left=0.02, right=0.33, top=0.84, bottom=0.19)
    right = fig.add_gridspec(2, 1, left=0.42, right=0.98, top=0.835, bottom=0.10, hspace=0.42)

    # (a) R1t's design, with the cells that vary across the pool
    ax = fig.add_subplot(left[0, 0])
    field_axes(ax, g, "(a) R1t's qualified binary design (iterate 20)")
    centres = ft["design_elem_centres"]
    xe, ye, grid = density_grid(fw["binary_20"], centres, h, g)
    m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                      shading="flat")
    colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
    for k in varying:
        cx, cy = centres[k]
        ax.add_patch(plt.Rectangle((cx - h / 2, cy - h / 2), h, h, fill=False, ec=SERIES[2],
                                   lw=1.1, zorder=4.5))  # above the rings, below the labels
    cx, cy = centres[cell]
    ax.plot([cx], [cy], "o", ms=12, mfc="none", mec=SURFACE, mew=3.2, zorder=4)
    ax.plot([cx], [cy], "o", ms=12, mfc="none", mec=INK, mew=1.4, zorder=4)
    ax.annotate(f"cell {cell}: {state[m_near[0][cell]]} in the {len(near)}\nlowest-J designs "
                f"(w = 0.5),\n{state[m_far[0][cell]]} in the other {len(far)}", xy=(cx, cy),
                xytext=(cx + 0.0005, cy + 0.0006),
                fontsize=7.5, color=INK, ha="left", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    mark_isolated(ax, grid, ye, h, f"isolated by shared\nedges in all {len(labels)}")
    note(ax, f"green: the {len(varying)} cells whose state differs\nbetween the {len(labels)} "
             f"designs; each design has\n{fluid_cells.pop()} fluid cells in the design domain",
         y=-0.135)

    # (b) J on D by iterate: the binary design each iterate exports, and the iterate itself
    ax = fig.add_subplot(right[0, 0])
    ax.grid(axis="y", color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0.0, color=AXIS, lw=0.8)
    cont = np.array([100 * (j_c[i] / j_c[20] - 1) for i in iterates])
    ax.plot(iterates, cont, color=INK2, lw=1.4, marker="o", ms=3, zorder=2)
    for i in iterates:
        k = next(lab for lab in labels if i in rows[lab]["iterations"])
        ax.plot([i], [100 * (rows[k]["J"] / j_t - 1)], "o", ms=7, color=band[k],
                mfc=SURFACE if k in reused else band[k], mew=1.6, zorder=4 if k in reused else 3)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=7, color=SERIES[0],
                   label=f"exported binary design, cell {cell} {state[m_near[0][cell]]} "
                         f"({len(near)} designs)"),
        plt.Line2D([], [], ls="none", marker="o", ms=7, color=SERIES[1],
                   label=f"exported binary design, cell {cell} {state[m_far[0][cell]]} "
                         f"({len(far)} designs)"),
        plt.Line2D([], [], ls="none", marker="o", ms=7, color=INK2, mfc=SURFACE, mew=1.6,
                   label="open: R1t's or R1v's recorded D state, reused"),
        plt.Line2D([], [], color=INK2, lw=1.4, marker="o", ms=3,
                   label="the continuous design at the iterate"),
    ]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.36), frameon=False,
              fontsize=7.5, handletextpad=0.4)
    second = rows[rank["order"][1]]
    ax.annotate(f"iterate {second['label']}: {100 * (second['J'] / j_t - 1):+.3f}%",
                xy=(second["label"], 100 * (second["J"] / j_t - 1)), xytext=(24.4, 0.40),
                fontsize=7.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate(f"R1t's design: the lowest of the {len(labels)}", xy=(20, 0.0),
                xytext=(20.4, 0.62), fontsize=7.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, relpos=(0, 0)))
    ax.set_xlim(19.4, 40.6)
    ax.set_xticks(range(20, 41, 2))
    ax.set_xlabel("iterate of R1v's run (20 is R1t's terminal)")
    ax.set_ylabel("J on D, change from its own\nvalue at iterate 20 (%)")
    ax.set_title("(b) J at w = 0.5 on D along R1v's trajectory", loc="left", fontsize=9.5)

    # (c) Psi and C against R1t's design, with J's level line through it
    ax = fig.add_subplot(right[1, 0])
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    scale = rec["scale"]
    w = rec["contract"]["weight"]
    t = rows[20]

    def rel(c):
        return 100 * (c["psi"] / t["psi"] - 1), 100 * (c["compliance"] / t["compliance"] - 1)

    slope = -(w * t["psi"] / scale["psi_0"]) / ((1 - w) * t["compliance"] / scale["c_0"])
    xs = np.array([-2.7, 1.35])
    ax.plot(xs, slope * xs, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(-2.6, slope * -2.6 + 0.06, "equal J at w = 0.5 through R1t's design", fontsize=7.5,
            color=MUTED, rotation=np.degrees(np.arctan(slope)), transform_rotates_text=True,
            rotation_mode="anchor", ha="left", va="bottom")
    for k in labels:
        x, y = rel(rows[k])
        ax.plot([x], [y], "o", ms=6.5, color=band[k], mfc=SURFACE if k in reused else band[k],
                mew=1.5, zorder=5 if k in reused else 3)
    cands = rv["candidates"]
    xu, yu = rel(cands["r1u"])
    ax.plot([xu], [yu], "v", color=INK, mfc=SURFACE, mew=1.4, ms=6.5, zorder=4)
    ax.annotate("R1t's, filled (R1u)", xy=(xu, yu), xytext=(xu + 0.02, -0.24), fontsize=7.5,
                color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, relpos=(0, 1)))
    ax.annotate("R1t's (iterate 20)", xy=(0.0, 0.0), xytext=(-0.95, -0.24), fontsize=7.5,
                color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    xr, yr = rel(cands["r1r"])
    ax.plot([xr], [yr], "D", color=INK, ms=6.5, zorder=4)
    ax.text(xr, yr + 0.13, "R1r's", fontsize=7.5, color=INK2, ha="center", va="bottom")
    xv, yv = rel(rows[r1v_label])
    its = ", ".join(str(i) for i in rows[r1v_label]["iterations"])
    ax.annotate(f"R1v's (iterates {its})", xy=(xv, yv), xytext=(xv - 0.05, yv + 0.23),
                fontsize=7.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, relpos=(0, 0)))
    names = {"r1o": "R1o's", "r1n": "R1n's", "pilot": "the pilot's"}
    ax.text(0.99, 0.97, "off the panel (Ψ, C from R1t's):\n"
            + "\n".join(f"{names[n]} {x:+.1f}%, {y:+.1f}%"
                        for n, (x, y) in ((n, rel(cands[n])) for n in names)),
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED,
            linespacing=1.35)
    ax.set_xlim(*xs)
    ax.set_ylim(-0.35, 2.15)
    ax.set_xlabel("dissipated power Ψ, change from R1t's design (%)")
    ax.set_ylabel("thermal compliance C,\nchange from R1t's design (%)")
    ax.set_title(f"(c) The {len(labels)} designs on D, with the candidates nearest R1t's",
                 loc="left", fontsize=9.5)

    env = {i["lowest"]: i for i in rank["lowest_by_weight"]["intervals"]}
    if set(env) != {"r1r", "i20", "pilot"}:
        raise RuntimeError(f"the envelope's designs changed: {sorted(env)}")
    fig.suptitle(f"R1w — the {len(labels)} binary designs along R1v's trajectory, ranked once on D",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.985)
    fig.text(0.02, 0.958,
             f"{rank['scope'].capitalize()}: R1t's design is the lowest in J at w = 0.5; the "
             f"nearest, iterate {second['label']}'s, is {100 * (second['J'] / j_t - 1):+.3f}%. "
             f"{solves['flow_made']} designs solved (1F + 1T each), {len(reused)} reused; every "
             "state passed the 10⁻⁸ gate.\nLowest by weight over these "
             f"{len(labels)} and R1u's, R1r's, R1o's, R1n's and the pilot's: R1r's for w < "
             f"{env['r1r']['to']:.4f}, R1t's for {env['i20']['from']:.4f} < w < "
             f"{env['i20']['to']:.4f}, the pilot's above.\n"
             f"The {len(labels)} come from R1t's and R1v's runs on D; R1w ranked them on D with no "
             "MMA update and no AD.",
             fontsize=8.5, color=INK2, ha="left", va="top", linespacing=1.4)
    footer(fig, "Drawn from results/zhao2d_r1w.json, zhao2d_r1w_fields.npz, zhao2d_r1v.json and "
           "zhao2d_r1t_fields.npz — scripts/zhao2d_figures.py; no state is re-solved.")
    path = out / "zhao2d_r1w.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def digest(*arrays) -> str:
    """sha256 over dtype, shape and bytes: tfopus.zhao2d_flow_study.digest, restated so this
    script still needs no JAX."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def lead_figure(res: pathlib.Path, out: pathlib.Path, g: dict) -> pathlib.Path:
    """R1t's binary design on D in the layout of Zhao Figs. 8 and 11, and R1u's difference."""
    ft = np.load(res / "zhao2d_r1t_fields.npz")
    fu = np.load(res / "zhao2d_r1u_fields.npz")
    rt = json.loads((res / "zhao2d_r1t.json").read_text(encoding="utf-8"))
    ru = json.loads((res / "zhao2d_r1u.json").read_text(encoding="utf-8"))
    coords_2 = np.load(res / "zhao2d_r1m_fields.npz")["flow_node_coords_h2"]
    coords_8 = np.load(res / "zhao2d_r1i_fields.npz")["thermal_node_coords_h8"]
    ct, cu = rt["cells"]["new/check"], ru["cells"]["filled/check"]
    layer = rt["check_layer"]
    pv_t, temp_t = ft["new_press_vel_check"], ft["new_temperature_check"]
    pv_u, temp_u = fu["press_vel"], fu["temperature"]
    # the saved meshes and states are the ones the records describe, by their own hashes
    if not (digest(coords_2) == layer["flow_mesh"]["node_coords_sha256"]
            and digest(coords_8) == layer["thermal_mesh"]["node_coords_sha256"]
            and all(ru["check_layer"][m]["node_coords_sha256"] == layer[m]["node_coords_sha256"]
                    for m in ("flow_mesh", "thermal_mesh"))
            and digest(ft["solid_fraction_binary"]) == rt["export"]["binary_sha256"]
            and digest(pv_t) == ct["state_sha256"]["press_vel"]
            and digest(temp_t) == ct["state_sha256"]["temperature"]
            and digest(pv_u) == cu["state_sha256"]["press_vel"]
            and digest(temp_u) == cu["state_sha256"]["temperature"]):
        raise RuntimeError("the saved meshes or states are not the ones R1t's and R1u's records hash")
    h = g["element_size"]
    tri_2 = triangulation(coords_2, h / 2, g)
    tri_8 = triangulation(coords_8, h / 8, g)
    speed = np.linalg.norm(pv_t.reshape(-1, 3)[:, 1:], axis=1)
    d_temp = temp_u - temp_t
    d_max = float(np.abs(d_temp).max())

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 8.0))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.17, wspace=0.10)

    ax = axes[0]
    field_axes(ax, g, "(a) R1t's qualified binary design")
    xe, ye, grid = density_grid(ft["solid_fraction_binary"], ft["design_elem_centres"], h, g)
    m = ax.pcolormesh(xe, ye, np.ma.masked_invalid(grid).T, cmap=DENSITY, vmin=0, vmax=1,
                      shading="flat")
    colorbar(fig, m, ax, "γ  (0 solid, 1 fluid)")
    mark_isolated(ax, grid, ye, h, "isolated by shared\nedges (R1u fills it)")
    note(ax, f"design mesh h, {int(round((1 - ft['solid_fraction_binary']).sum()))} fluid cells\n"
             f"with the tabs; v_f (design domain) {ct['fluid_fractions']['v_f_design_domain']:.3f}")

    ax = axes[1]
    field_axes(ax, g, "(b) Velocity |u|, flow mesh h/2")
    m = ax.tripcolor(tri_2, speed, cmap=SPEED, vmin=0, vmax=max(0.3, float(speed.max())),
                     shading="gouraud")
    colorbar(fig, m, ax, "|u|")
    note(ax, f"max |u| {speed.max():.3f}\nΨ = {ct['psi']:.6f}")

    ax = axes[2]
    field_axes(ax, g, "(c) Temperature, thermal mesh h/8")
    m = ax.tripcolor(tri_8, temp_t, cmap=HEAT, vmin=0, vmax=float(temp_t.max()),
                     shading="gouraud")
    colorbar(fig, m, ax, "T")
    note(ax, f"T_max {ct['T_max']:.2f}; C = {ct['compliance']:.0f}\n"
             f"D_T/Q {ct['D_T_over_Q']:.2%}")

    ax = axes[3]
    field_axes(ax, g, "(d) R1u − R1t: temperature")
    m = ax.tripcolor(tri_8, d_temp, cmap=DIVERGE, vmin=-d_max, vmax=d_max, shading="gouraud")
    colorbar(fig, m, ax, "ΔT")
    note(ax, (f"cell filled, nothing else changed\nΔT {d_temp.min():+.3f} to {d_temp.max():+.3f}\n"
              f"J {cu['J'] / ct['J'] - 1:+.3%} on the common scale").replace("-", "−"))

    fig.suptitle("The lead designs on the check layer D, in the layout of Zhao Figs. 8 and 11",
                 x=0.02, ha="left", fontsize=11.5, color=INK, y=0.975)
    fig.text(0.02, 0.925,
             "R1t's qualified binary design, the numerical first among the evaluated designs "
             "at w = 0.5 on D (flow h/2, thermal h/8), and R1u's, the same design with its "
             "isolated cell filled.\nZhao's Fig. 11 scales |u| 0–0.3 and T 0–12 on its "
             "5200-element mesh. D has 4 and 64 times its elements (flow, thermal); refining "
             "the thermal mesh raised C, the flow lowered it (R1i; R1h, R1m).",
             fontsize=8.5, color=INK2, ha="left", va="top")
    footer(fig, "Drawn from results/zhao2d_r1t_fields.npz, zhao2d_r1u_fields.npz and their records, "
           "nodes from zhao2d_r1m/r1i_fields.npz; design, states and nodes checked by hash — "
           "scripts/zhao2d_figures.py; nothing re-solved.")
    path = out / "zhao2d_lead_fields.png"
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
             "zhao2d_r1i_h8.json", "zhao2d_r1k_warm_start.json", "zhao2d_r1k_fields.npz",
             "zhao2d_r1k_terminal_check.json", "zhao2d_r1k_terminal_fields.npz",
             "zhao2d_r1l_baselines.json", "zhao2d_r1l_vp_pilot.json", "zhao2d_r1l_h8_check.json",
             "zhao2d_r1m_flow_check.json", "zhao2d_r1m_fields.npz",
             "zhao2d_r1n_beta32.json", "zhao2d_r1n_fields.npz", "zhao2d_r1o.json",
             "zhao2d_r1o_fields.npz", "zhao2d_r1p_bridge.json", "zhao2d_r1r.json",
             "zhao2d_r1r_fields.npz", "zhao2d_r1t.json", "zhao2d_r1t_fields.npz",
             "zhao2d_r1v.json", "zhao2d_r1v_fields.npz", "zhao2d_r1w.json",
             "zhao2d_r1w_fields.npz", "zhao2d_r1u.json", "zhao2d_r1u_fields.npz"]
    for n in names:
        print(f"read results/{n}  sha256 {sha(res / n)}")
    sources = [f"results/{n}" for n in names]
    for path in (fields_figure(res, args.out, g, sources), history_figure(res, args.out),
                 status_figure(res, args.out), r1k_figure(res, args.out, g),
                 terminal_figure(res, args.out, g), r1l_figure(res, args.out, g),
                 r1m_figure(res, args.out, g), r1n_figure(res, args.out, g),
                 r1o_figure(res, args.out, g), r1p_figure(res, args.out),
                 r1r_figure(res, args.out, g), r1t_figure(res, args.out, g),
                 r1v_figure(res, args.out, g), r1w_figure(res, args.out, g),
                 lead_figure(res, args.out, g)):
        print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")


if __name__ == "__main__":
    main()
