"""The two Zhou case geometries, reconstructed as swept shells.

Zhou's text does not pin either geometry down completely, so every quantity the
paper leaves open is a named field here with a recorded default, never a silent
constant. `CylinderSpec.provenance()` and `SphereSpec.provenance()` print which
numbers come from the paper and which are this reproduction's choices.

Layout, in the parametric box (p1 = along the flow, p2 = across it):

    p1:  [-port_extent, 0] | design [0, Arc] | [Arc, Arc + port_extent]
         |<- inlet ext  ->|                  |<- outlet ext ->|

Each extension band is a non-design fluid channel over a p2 interval, embedded
in non-design solid that walls it in. The inlet face is the p1-minus end of that
channel; the rest of that end face is wall. Where the channel sits differs
between the two cases, and both come from the figures:

    cylinder   Figure 3 marks a symmetry BC along the lower edge with the ports
               straddling it, so this is a half model: the channel runs from
               p2 = 0 (the symmetry plane) to half the port width.
    sphere     Figure 14 shows a full patch with a 3.5 mm port centred on each
               of two opposite edges, and section 4.2 never mentions a half
               model, so the channel is centred on p2 = 0 and there is no
               symmetry plane.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from tfopus import mesh as _mesh

Region = _mesh.Region
Face = _mesh.Face


def graded_nodes(breakpoints, target_h: float) -> np.ndarray:
    """Node coordinates hitting every breakpoint, filled at about `target_h`.

    Guarantees a grid line exactly on each breakpoint so that region boundaries
    (design vs port, channel vs wall) fall on element faces rather than cutting
    through elements.
    """
    breakpoints = np.asarray(breakpoints, dtype=float)
    out = [np.array([breakpoints[0]])]
    for lo, hi in zip(breakpoints[:-1], breakpoints[1:]):
        n = max(1, int(round(abs(hi - lo) / target_h)))
        out.append(np.linspace(lo, hi, n + 1)[1:])
    return np.concatenate(out)


@dataclasses.dataclass(frozen=True)
class CylinderSpec:
    """Zhou section 4.1. See `provenance` for what is reported vs assumed."""

    inner_radius: float = 0.0195  # paper
    thickness: float = 0.0005  # paper
    heat_source: float = 3.0e7  # paper, W/m^3, design domain only
    inlet_temperature: float = 293.15  # paper body text (figure 3 says 293)
    inlet_speed: float = 0.1  # paper
    fluid_fraction_max: float = 0.45  # paper
    target_elem_size: float = 0.0003  # paper: inner-surface mean size

    # Not determined by the paper.
    arc_angle: float = np.pi
    axial_span: float = 0.010  # half-model extent along the axis
    axial_scope: str = "half"  # "half": 10 mm is the modelled half; "full": 10 mm is the full length
    port_extent: float = 0.002  # arc length of each non-design extension
    port_half_width: float = 0.00175  # half of a 3.5 mm port
    n_thick: int = 2

    @property
    def arc_length(self) -> float:
        return self.inner_radius * self.arc_angle

    @property
    def modelled_axial_span(self) -> float:
        return self.axial_span if self.axial_scope == "half" else 0.5 * self.axial_span

    def provenance(self) -> dict:
        return {
            "from_paper": {
                "inner_radius": self.inner_radius,
                "thickness": self.thickness,
                "heat_source": self.heat_source,
                "inlet_temperature": self.inlet_temperature,
                "inlet_speed": self.inlet_speed,
                "fluid_fraction_max": self.fluid_fraction_max,
                "target_elem_size": self.target_elem_size,
                "half_model": True,
            },
            "reproduction_choice": {
                "arc_angle": self.arc_angle,
                "axial_span": self.axial_span,
                "axial_scope": self.axial_scope,
                "port_extent": self.port_extent,
                "port_half_width": self.port_half_width,
                "n_thick": self.n_thick,
            },
            "paper_gaps": [
                "circumferential angle never stated",
                "axial 10 mm: body text says design domain, figure 3 labels the "
                "half-model view",
                "inlet/outlet port size and extension never stated for the cylinder",
                "through-thickness layer count never stated",
                "q_alpha / q_k never stated for the cylinder (4.2 gives 0.2 for "
                "the sphere only)",
            ],
        }


@dataclasses.dataclass(frozen=True)
class SphereSpec:
    """Zhou section 4.2. Gnomonic patch; see the mesh module for why."""

    inner_radius: float = 0.0195  # paper
    thickness: float = 0.0005  # paper
    heat_source: float = 2.5e7  # paper, W/m^3
    inlet_temperature: float = 293.0  # paper
    inlet_speed: float = 0.1  # paper
    fluid_fraction_max: float = 0.45  # paper
    angular_span: float = 120.0 * np.pi / 180.0  # paper: "azimuthal angle range of 120 deg"
    ramp_q_alpha: float = 0.2  # paper
    ramp_q_k: float = 0.2  # paper

    # Not determined by the paper. Section 4.2 gives no mesh size at all; the
    # 0.3 mm in section 4.1 is stated for the cylinder only. 0.5 mm keeps the
    # sphere's cost comparable to the cylinder's on a patch of ~2.4x the area.
    target_elem_size: float = 0.0005
    port_extent: float = 0.002  # arc length of each non-design extension
    port_half_width: float = 0.00175  # half-width of the 3.5 mm port (figure 14)
    n_thick: int = 2

    @property
    def half_angle(self) -> float:
        return 0.5 * self.angular_span

    def provenance(self) -> dict:
        return {
            "from_paper": {
                "inner_radius": self.inner_radius,
                "thickness": self.thickness,
                "heat_source": self.heat_source,
                "inlet_temperature": self.inlet_temperature,
                "angular_span_deg": np.degrees(self.angular_span),
                "ramp_q_alpha": self.ramp_q_alpha,
                "ramp_q_k": self.ramp_q_k,
                "port_size_from_figure_14": 0.0035,
            },
            "reproduction_choice": {
                "patch": "gnomonic (cubed-sphere) square, not a polar cap",
                "target_elem_size": self.target_elem_size,
                "port_extent": self.port_extent,
                "port_half_width": self.port_half_width,
                "n_thick": self.n_thick,
            },
            "paper_gaps": [
                "'azimuthal angle range 120 deg' does not fix the patch shape; a "
                "gnomonic square spanning 120 deg along each centre line is one "
                "reading, not the authors' CAD",
                "port cross-section, extension length and connection shape unstated",
                "alpha_max continuation schedule unstated",
                "section 4.2 average temperature 303.98 K exceeds table 6 maximum "
                "300.99 K; both are recorded, neither is treated as the target",
            ],
        }


def _port_region_fn(design_lo, design_hi, channel_lo, channel_hi):
    """Design inside the flow-direction band; outside it, channel or solid.

    The port channel is the p2 interval [channel_lo, channel_hi] inside the
    non-design extension bands; the rest of each band is solid that walls it in.
    """

    def region_fn(c1, c2):
        in_design = (c1 > design_lo) & (c1 < design_hi)
        in_channel = (c2 > channel_lo) & (c2 < channel_hi)
        port = np.where(in_channel, int(Region.PORT_FLUID), int(Region.PORT_SOLID))
        return np.where(in_design, int(Region.DESIGN), port)

    return region_fn


def _port_face_fn(channel_lo, channel_hi, symmetry_at_p2_min: bool):
    """p1 ends are inlet/outlet over the channel; everything else is wall.

    `symmetry_at_p2_min` tags the p2-minus boundary as a symmetry plane. The
    cylinder is a half model (Zhou figure 3 marks the symmetry BC on that edge);
    the sphere is not -- section 4.2 never mentions one, and figure 14 shows a
    full patch with a port centred on each of two opposite edges.
    """

    def face_fn(axis, side, phys_centre, param_centre):
        if axis == 0:
            if channel_lo < param_centre[1] < channel_hi:
                return Face.INLET if side == 0 else Face.OUTLET
            return Face.WALL
        if axis == 1 and side == 0 and symmetry_at_p2_min:
            return Face.SYMMETRY
        return Face.WALL

    return face_fn


def build_cylinder(spec: CylinderSpec, dofs_per_node: int) -> _mesh.ShellMesh:
    """Swept mesh for the cylindrical sector, ports included."""
    R = spec.inner_radius
    arc = spec.arc_angle
    d_theta_port = spec.port_extent / R
    theta = graded_nodes(
        [-d_theta_port, 0.0, arc, arc + d_theta_port], spec.target_elem_size / R
    )
    z = graded_nodes(
        [0.0, spec.port_half_width, spec.modelled_axial_span], spec.target_elem_size
    )
    # The port straddles the symmetry plane, so the half model carries half of it.
    return _mesh.build_shell_mesh(
        param_map=_mesh.cylinder_map(R, spec.thickness),
        p1=theta,
        p2=z,
        n_thick=spec.n_thick,
        dofs_per_node=dofs_per_node,
        region_fn=_port_region_fn(0.0, arc, -1.0, spec.port_half_width),
        face_fn=_port_face_fn(-1.0, spec.port_half_width, symmetry_at_p2_min=True),
    )


def build_sphere(spec: SphereSpec, dofs_per_node: int) -> _mesh.ShellMesh:
    """Swept mesh for the equiangular gnomonic spherical patch, ports included.

    The parametric box is an angle box, so every length converts to an angle by
    dividing by R and element size is uniform in arc length.
    """
    R = spec.inner_radius
    A = spec.half_angle
    h_ang = spec.target_elem_size / R
    port_ang = spec.port_extent / R
    w_ang = spec.port_half_width / R
    alpha = graded_nodes([-A - port_ang, -A, A, A + port_ang], h_ang)
    # Figure 14: a full patch with a 3.5 mm port centred on each of two opposite
    # edges, and no symmetry plane anywhere in section 4.2.
    beta = graded_nodes([-A, -w_ang, w_ang, A], h_ang)
    return _mesh.build_shell_mesh(
        param_map=_mesh.spherical_cap_map(R, spec.thickness),
        p1=alpha,
        p2=beta,
        n_thick=spec.n_thick,
        dofs_per_node=dofs_per_node,
        region_fn=_port_region_fn(-A, A, -w_ang, w_ang),
        face_fn=_port_face_fn(-w_ang, w_ang, symmetry_at_p2_min=False),
    )
