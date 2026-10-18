"""STL geometry builders for tensile specimen and modifier zones.

All units are millimeters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Tri = Tuple[int, int, int]
NamedMesh = Tuple[str, List[Vec3], List[Tri]]


@dataclass(frozen=True)
class SpecimenConfig:
    total_length: float = 165.0
    grip_width: float = 19.0
    gauge_width: float = 13.0
    parallel_length: float = 57.0
    transition_length: float | None = None
    distance_between_grips: float = 115.0
    fillet_radius: float = 76.2
    thickness: float = 3.2
    shoulder_points: int = 36


@dataclass(frozen=True)
class ModifierConfig:
    gauge_length: float = 50.0
    gauge_width: float = 13.0
    thickness: float = 3.2
    # Slight overlap improves reliability when imported as modifier bodies.
    overlap: float = 0.10


@dataclass(frozen=True)
class RectangularGradientConfig:
    specimen_length: float = 100.0
    specimen_width: float = 50.0
    gradient_length: float = 50.0
    gradient_width: float = 50.0
    thickness: float = 3.2
    x_partitions: int = 1
    y_partitions: int = 1
    overlap: float = 0.10


@dataclass(frozen=True)
class SmoothDogboneConfig:
    """Dimensions of a flat dogbone with tangent-continuous shoulders, in mm."""

    total_length: float
    grip_width: float
    gauge_width: float
    parallel_length: float
    grip_length: float
    thickness: float
    shoulder_segments: int = 32


def signed_area(poly: Sequence[Vec2]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def cross2(a: Vec2, b: Vec2, c: Vec2) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def point_in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool:
    c1 = cross2(a, b, p)
    c2 = cross2(b, c, p)
    c3 = cross2(c, a, p)
    has_neg = (c1 < 0.0) or (c2 < 0.0) or (c3 < 0.0)
    has_pos = (c1 > 0.0) or (c2 > 0.0) or (c3 > 0.0)
    return not (has_neg and has_pos)


def triangulate_polygon(poly: Sequence[Vec2]) -> List[Tri]:
    if len(poly) < 3:
        raise ValueError("Polygon needs at least 3 points.")

    points = list(poly)
    if signed_area(points) < 0:
        points.reverse()

    idx = list(range(len(points)))
    tris: List[Tri] = []
    guard = 0

    while len(idx) > 3:
        guard += 1
        if guard > 10_000:
            raise RuntimeError("Triangulation failed; check polygon validity.")

        ear_found = False
        m = len(idx)
        for i in range(m):
            i_prev = idx[(i - 1) % m]
            i_curr = idx[i]
            i_next = idx[(i + 1) % m]

            a = points[i_prev]
            b = points[i_curr]
            c = points[i_next]

            if cross2(a, b, c) <= 0:
                continue

            contains_point = False
            for j in idx:
                if j in (i_prev, i_curr, i_next):
                    continue
                if point_in_triangle(points[j], a, b, c):
                    contains_point = True
                    break
            if contains_point:
                continue

            tris.append((i_prev, i_curr, i_next))
            del idx[i]
            ear_found = True
            break

        if not ear_found:
            raise RuntimeError("No ear found; polygon may be self-intersecting.")

    tris.append((idx[0], idx[1], idx[2]))
    return tris


def build_extruded_mesh(poly: Sequence[Vec2], thickness: float) -> Tuple[List[Vec3], List[Tri]]:
    points = list(poly)
    if signed_area(points) < 0:
        points.reverse()

    n = len(points)
    bottom: List[Vec3] = [(x, y, 0.0) for (x, y) in points]
    top: List[Vec3] = [(x, y, thickness) for (x, y) in points]
    vertices = bottom + top

    faces: List[Tri] = []
    cap = triangulate_polygon(points)

    faces.extend((a + n, b + n, c + n) for (a, b, c) in cap)
    faces.extend((c, b, a) for (a, b, c) in cap)

    for i in range(n):
        j = (i + 1) % n
        i0, j0 = i, j
        i1, j1 = i + n, j + n
        faces.append((i0, j0, j1))
        faces.append((i0, j1, i1))

    return vertices, faces


def tri_normal(v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3:
    ux, uy, uz = (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2])
    vx, vy, vz = (v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2])
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    return (nx, ny, nz)


def write_ascii_stl(path: Path, name: str, vertices: Sequence[Vec3], faces: Iterable[Tri]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f"solid {name}\n")
        for a, b, c in faces:
            n = tri_normal(vertices[a], vertices[b], vertices[c])
            f.write(f"  facet normal {n[0]:.8g} {n[1]:.8g} {n[2]:.8g}\n")
            f.write("    outer loop\n")
            for i in (a, b, c):
                x, y, z = vertices[i]
                f.write(f"      vertex {x:.8g} {y:.8g} {z:.8g}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


def build_specimen_polygon(cfg: SpecimenConfig) -> List[Vec2]:
    half_total = cfg.total_length / 2.0
    half_grip_w = cfg.grip_width / 2.0
    half_gauge_w = cfg.gauge_width / 2.0

    if cfg.parallel_length <= 0 or cfg.distance_between_grips <= 0:
        raise ValueError("parallel_length and distance_between_grips must be > 0.")
    if cfg.distance_between_grips <= cfg.parallel_length:
        raise ValueError("distance_between_grips must be larger than parallel_length.")
    if cfg.total_length <= cfg.distance_between_grips:
        raise ValueError("total_length must be larger than distance_between_grips.")
    if cfg.fillet_radius <= 0:
        raise ValueError("fillet_radius must be > 0.")
    if cfg.shoulder_points < 2:
        raise ValueError("shoulder_points must be >= 2.")

    derived_transition = (cfg.distance_between_grips - cfg.parallel_length) / 2.0
    if cfg.transition_length is not None and abs(cfg.transition_length - derived_transition) > 1e-6:
        raise ValueError(
            "transition_length conflicts with parallel_length and distance_between_grips."
        )

    grip_tab_len = (cfg.total_length - cfg.distance_between_grips) / 2.0
    if grip_tab_len <= 0:
        raise ValueError("Computed grip tab length must be > 0.")

    x0 = -half_total
    x5 = +half_total
    xg_l = x0 + grip_tab_len
    xg_r = x5 - grip_tab_len
    xn_l = -cfg.parallel_length / 2.0
    xn_r = +cfg.parallel_length / 2.0

    if not (x0 < xg_l < xn_l < xn_r < xg_r < x5):
        raise ValueError("Invalid geometry ordering; please verify dimensions.")

    def _norm_angle(a: float) -> float:
        while a <= -math.pi:
            a += 2.0 * math.pi
        while a > math.pi:
            a -= 2.0 * math.pi
        return a

    def _arc_points(p0: Vec2, p1: Vec2, center: Vec2, n: int) -> List[Vec2]:
        cx, cy = center
        t0 = math.atan2(p0[1] - cy, p0[0] - cx)
        t1 = math.atan2(p1[1] - cy, p1[0] - cx)
        d = _norm_angle(t1 - t0)
        pts: List[Vec2] = []
        for i in range(1, n):
            t = i / n
            th = t0 + d * t
            pts.append((cx + cfg.fillet_radius * math.cos(th), cy + cfg.fillet_radius * math.sin(th)))
        return pts

    half_delta_w = half_grip_w - half_gauge_w
    if half_delta_w < 0:
        raise ValueError("grip_width must be >= gauge_width.")
    if half_delta_w > cfg.fillet_radius:
        raise ValueError("fillet_radius is too small for grip/gauge width difference.")

    # ASTM D638 Figure 1: use R as shoulder radius tied to the reduced section.
    # D remains grip-spacing guidance; shoulders can start inside that region.
    dx_to_grip_line = math.sqrt(
        cfg.fillet_radius * cfg.fillet_radius
        - (cfg.fillet_radius - half_delta_w) * (cfg.fillet_radius - half_delta_w)
    )
    x_arc_l = xn_l - dx_to_grip_line
    x_arc_r = xn_r + dx_to_grip_line
    if x_arc_l < xg_l - 1e-6 or x_arc_r > xg_r + 1e-6:
        raise ValueError(
            "distance_between_grips is too small for the current fillet radius and widths."
        )

    p_ll0 = (x_arc_l, -half_grip_w)
    p_ll1 = (xn_l, -half_gauge_w)
    p_lr0 = (xn_r, -half_gauge_w)
    p_lr1 = (x_arc_r, -half_grip_w)
    p_ur0 = (x_arc_r, +half_grip_w)
    p_ur1 = (xn_r, +half_gauge_w)
    p_ul0 = (xn_l, +half_gauge_w)
    p_ul1 = (x_arc_l, +half_grip_w)

    c_ll = (xn_l, -half_gauge_w - cfg.fillet_radius)
    c_lr = (xn_r, -half_gauge_w - cfg.fillet_radius)
    c_ur = (xn_r, +half_gauge_w + cfg.fillet_radius)
    c_ul = (xn_l, +half_gauge_w + cfg.fillet_radius)

    arc_ll = _arc_points(p_ll0, p_ll1, c_ll, cfg.shoulder_points)
    arc_lr = _arc_points(p_lr0, p_lr1, c_lr, cfg.shoulder_points)
    arc_ur = _arc_points(p_ur0, p_ur1, c_ur, cfg.shoulder_points)
    arc_ul = _arc_points(p_ul0, p_ul1, c_ul, cfg.shoulder_points)

    poly: List[Vec2] = [(x0, -half_grip_w), (xg_l, -half_grip_w), p_ll0]
    poly.extend(arc_ll)
    poly.extend([p_ll1, (xn_r, -half_gauge_w)])
    poly.extend(arc_lr)
    poly.extend([p_lr1, (xg_r, -half_grip_w), (x5, -half_grip_w), (x5, +half_grip_w), (xg_r, +half_grip_w), p_ur0])
    poly.extend(arc_ur)
    poly.extend([p_ur1, (xn_l, +half_gauge_w)])
    poly.extend(arc_ul)
    poly.extend([p_ul1, (xg_l, +half_grip_w), (x0, +half_grip_w)])
    return poly


def build_smooth_dogbone_polygon(cfg: SmoothDogboneConfig) -> List[Vec2]:
    """Return a symmetric outline with zero shoulder slope at both ends.

    A quintic smoothstep joins the straight grip and parallel sections. This
    avoids a corner at the 40-to-20 mm shoulder of the long gradient specimen.
    """

    if min(
        cfg.total_length,
        cfg.grip_width,
        cfg.gauge_width,
        cfg.parallel_length,
        cfg.grip_length,
        cfg.thickness,
    ) <= 0:
        raise ValueError("All smooth-dogbone dimensions must be > 0 mm.")
    if cfg.grip_width <= cfg.gauge_width:
        raise ValueError("grip_width must exceed gauge_width.")
    if cfg.shoulder_segments < 2:
        raise ValueError("shoulder_segments must be >= 2.")

    shoulder_length = (cfg.total_length - cfg.parallel_length) / 2 - cfg.grip_length
    if shoulder_length <= 0:
        raise ValueError("Total length must leave positive shoulder length on both ends.")

    half_total = cfg.total_length / 2
    half_parallel = cfg.parallel_length / 2
    shoulder_end = half_total - cfg.grip_length
    half_grip = cfg.grip_width / 2
    half_gauge = cfg.gauge_width / 2

    def smoothstep(t: float) -> float:
        return t * t * t * (10 + t * (-15 + 6 * t))

    lower: List[Vec2] = [(-half_total, -half_grip), (-shoulder_end, -half_grip)]
    for i in range(1, cfg.shoulder_segments + 1):
        t = i / cfg.shoulder_segments
        x = -shoulder_end + t * shoulder_length
        half_width = half_grip + (half_gauge - half_grip) * smoothstep(t)
        lower.append((x, -half_width))

    lower.append((half_parallel, -half_gauge))
    for i in range(1, cfg.shoulder_segments + 1):
        t = i / cfg.shoulder_segments
        x = half_parallel + t * shoulder_length
        half_width = half_gauge + (half_grip - half_gauge) * smoothstep(t)
        lower.append((x, -half_width))
    lower.append((half_total, -half_grip))

    return lower + [(x, -y) for x, y in reversed(lower)]


def make_box_polygon(x_min: float, x_max: float, y_min: float, y_max: float) -> List[Vec2]:
    return [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]


def _clip_polygon_at_x(
    poly: Sequence[Vec2],
    x_boundary: float,
    keep_greater: bool,
    tolerance: float = 1e-10,
) -> List[Vec2]:
    """Clip a polygon by one vertical half-plane using Sutherland--Hodgman."""

    if not poly:
        return []

    def is_inside(point: Vec2) -> bool:
        if keep_greater:
            return point[0] >= x_boundary - tolerance
        return point[0] <= x_boundary + tolerance

    def intersection(start: Vec2, end: Vec2) -> Vec2:
        dx = end[0] - start[0]
        if abs(dx) <= tolerance:
            return (x_boundary, start[1])
        fraction = (x_boundary - start[0]) / dx
        return (x_boundary, start[1] + fraction * (end[1] - start[1]))

    clipped: List[Vec2] = []
    previous = poly[-1]
    previous_inside = is_inside(previous)
    for current in poly:
        current_inside = is_inside(current)
        if current_inside:
            if not previous_inside:
                clipped.append(intersection(previous, current))
            clipped.append(current)
        elif previous_inside:
            clipped.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside

    deduplicated: List[Vec2] = []
    for point in clipped:
        if not deduplicated or math.dist(point, deduplicated[-1]) > tolerance:
            deduplicated.append(point)
    if len(deduplicated) > 1 and math.dist(deduplicated[0], deduplicated[-1]) <= tolerance:
        deduplicated.pop()
    return deduplicated


def clip_polygon_to_x_range(
    poly: Sequence[Vec2], x_min: float, x_max: float
) -> List[Vec2]:
    """Return the in-plane intersection of a polygon and ``x_min <= x <= x_max``."""

    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max.")
    clipped = _clip_polygon_at_x(poly, x_min, keep_greater=True)
    clipped = _clip_polygon_at_x(clipped, x_max, keep_greater=False)
    if len(clipped) < 3 or abs(signed_area(clipped)) <= 1e-12:
        raise ValueError("The x range does not produce a finite polygonal region.")
    return clipped


def _build_modifier_box_mesh(
    cfg: ModifierConfig,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[List[Vec3], List[Tri]]:
    z_thickness = cfg.thickness + 2.0 * cfg.overlap
    poly = make_box_polygon(x_min, x_max, y_min, y_max)
    vertices, faces = build_extruded_mesh(poly, z_thickness)
    shifted = [(x, y, z - cfg.overlap) for (x, y, z) in vertices]
    return shifted, faces


def build_modifier_meshes_1d(cfg: ModifierConfig, x_partitions: int) -> List[NamedMesh]:
    if x_partitions < 1:
        raise ValueError("x_partitions must be >= 1")

    half_gauge_len = cfg.gauge_length / 2.0
    y_min = -cfg.gauge_width / 2.0
    y_max = +cfg.gauge_width / 2.0
    x_step = cfg.gauge_length / x_partitions

    meshes: List[NamedMesh] = []
    for i in range(x_partitions):
        x0 = -half_gauge_len + i * x_step
        x1 = x0 + x_step
        name = f"modifier_X{i + 1:02d}"
        vertices, faces = _build_modifier_box_mesh(cfg, x0, x1, y_min, y_max)
        meshes.append((name, vertices, faces))
    return meshes


def build_edge_extended_dogbone_modifiers(
    specimen_cfg: SmoothDogboneConfig,
    x_partitions: int,
    overlap: float = 0.10,
    parallel_region_lengths: Sequence[float] | None = None,
) -> List[NamedMesh]:
    """Partition the parallel section and continue end densities into grips.

    Each modifier is the exact in-plane intersection between the dogbone outline
    and its x interval. The first and last regions therefore follow the shoulder
    and grip contours. A small z overlap is retained for robust slicer import.
    """

    if x_partitions < 2:
        raise ValueError("x_partitions must be >= 2 for edge-extended modifiers.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0 mm.")
    specimen_polygon = build_smooth_dogbone_polygon(specimen_cfg)

    lengths = (
        tuple(parallel_region_lengths)
        if parallel_region_lengths is not None
        else (specimen_cfg.parallel_length / x_partitions,) * x_partitions
    )
    if len(lengths) != x_partitions or any(length <= 0 for length in lengths):
        raise ValueError("parallel_region_lengths must contain one positive length per region.")
    if abs(sum(lengths) - specimen_cfg.parallel_length) > 1e-6:
        raise ValueError("parallel_region_lengths must sum to parallel_length.")

    half_total = specimen_cfg.total_length / 2
    half_parallel = specimen_cfg.parallel_length / 2
    meshes: List[NamedMesh] = []
    cursor = -half_parallel

    for i in range(x_partitions):
        parallel_left = cursor
        cursor += lengths[i]
        x0 = -half_total if i == 0 else parallel_left
        x1 = half_total if i == x_partitions - 1 else cursor
        region_polygon = clip_polygon_to_x_range(specimen_polygon, x0, x1)
        vertices, faces = build_extruded_mesh(
            region_polygon, specimen_cfg.thickness + 2.0 * overlap
        )
        vertices = [(x, y, z - overlap) for x, y, z in vertices]
        meshes.append((f"modifier_X{i + 1:02d}", vertices, faces))
    return meshes


def build_modifier_meshes_2d(cfg: ModifierConfig, x_partitions: int, y_partitions: int) -> List[NamedMesh]:
    if x_partitions < 1:
        raise ValueError("x_partitions must be >= 1")
    if y_partitions < 1:
        raise ValueError("y_partitions must be >= 1")

    half_gauge_len = cfg.gauge_length / 2.0
    half_gauge_w = cfg.gauge_width / 2.0
    x_step = cfg.gauge_length / x_partitions
    y_step = cfg.gauge_width / y_partitions

    meshes: List[NamedMesh] = []
    for row in range(y_partitions):
        y0 = -half_gauge_w + row * y_step
        y1 = y0 + y_step
        for col in range(x_partitions):
            x0 = -half_gauge_len + col * x_step
            x1 = x0 + x_step
            name = f"modifier_X{col + 1:02d}_Y{row + 1:02d}"
            vertices, faces = _build_modifier_box_mesh(cfg, x0, x1, y0, y1)
            meshes.append((name, vertices, faces))
    return meshes


def write_modifier_meshes(
    output_dir: Path,
    modifier_meshes: Sequence[NamedMesh],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, vertices, faces in modifier_meshes:
        write_ascii_stl(output_dir / f"{name}.stl", name, vertices, faces)


def build_rectangular_specimen_polygon(
    specimen_length: float,
    specimen_width: float,
) -> List[Vec2]:
    if specimen_length <= 0 or specimen_width <= 0:
        raise ValueError("specimen_length and specimen_width must be > 0.")

    half_length = specimen_length / 2.0
    half_width = specimen_width / 2.0
    return make_box_polygon(-half_length, half_length, -half_width, half_width)


def build_rectangular_gradient_meshes(cfg: RectangularGradientConfig) -> List[NamedMesh]:
    if cfg.gradient_length <= 0 or cfg.gradient_width <= 0:
        raise ValueError("gradient_length and gradient_width must be > 0.")
    if cfg.gradient_length > cfg.specimen_length or cfg.gradient_width > cfg.specimen_width:
        raise ValueError("gradient area must stay inside the specimen bounds.")

    modifier_cfg = ModifierConfig(
        gauge_length=cfg.gradient_length,
        gauge_width=cfg.gradient_width,
        thickness=cfg.thickness,
        overlap=cfg.overlap,
    )
    return build_modifier_meshes_2d(
        modifier_cfg,
        x_partitions=cfg.x_partitions,
        y_partitions=cfg.y_partitions,
    )


def build_rectangular_partition_meshes(cfg: RectangularGradientConfig) -> List[NamedMesh]:
    if cfg.gradient_length <= 0 or cfg.gradient_width <= 0:
        raise ValueError("gradient_length and gradient_width must be > 0.")
    if cfg.gradient_length > cfg.specimen_length or cfg.gradient_width > cfg.specimen_width:
        raise ValueError("gradient area must stay inside the specimen bounds.")
    if cfg.x_partitions < 1 or cfg.y_partitions < 1:
        raise ValueError("x_partitions and y_partitions must be >= 1.")

    half_specimen_length = cfg.specimen_length / 2.0
    half_gradient_length = cfg.gradient_length / 2.0
    half_gradient_width = cfg.gradient_width / 2.0
    y_step = cfg.gradient_width / cfg.y_partitions
    x_step = cfg.gradient_length / cfg.x_partitions

    meshes: List[NamedMesh] = []

    for row in range(cfg.y_partitions):
        y0 = -half_gradient_width + row * y_step
        y1 = y0 + y_step

        for col in range(cfg.x_partitions):
            if col == 0:
                x0 = -half_specimen_length
            else:
                x0 = -half_gradient_length + col * x_step

            if col == cfg.x_partitions - 1:
                x1 = half_specimen_length
            else:
                x1 = -half_gradient_length + (col + 1) * x_step

            vertices, faces = _build_modifier_box_mesh(
                ModifierConfig(
                    gauge_length=x1 - x0,
                    gauge_width=y1 - y0,
                    thickness=cfg.thickness,
                    overlap=cfg.overlap,
                ),
                x0,
                x1,
                y0,
                y1,
            )
            meshes.append((f"measure_X{col + 1:02d}_Y{row + 1:02d}", vertices, faces))

    return meshes


def generate_dogbone_stl(
    output_dir: Path,
    specimen_cfg: SpecimenConfig | None = None,
    filename: str = "dogbone_specimen.stl",
) -> None:
    specimen_cfg = specimen_cfg or SpecimenConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    specimen_poly = build_specimen_polygon(specimen_cfg)
    vertices, faces = build_extruded_mesh(specimen_poly, specimen_cfg.thickness)
    solid_name = filename.replace(".stl", "")
    write_ascii_stl(output_dir / filename, solid_name, vertices, faces)


def generate_smooth_dogbone_stl(
    output_dir: Path,
    specimen_cfg: SmoothDogboneConfig,
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specimen_poly = build_smooth_dogbone_polygon(specimen_cfg)
    vertices, faces = build_extruded_mesh(specimen_poly, specimen_cfg.thickness)
    write_ascii_stl(output_dir / filename, filename.removesuffix(".stl"), vertices, faces)


def generate_edge_extended_dogbone_modifiers_stl(
    output_dir: Path,
    specimen_cfg: SmoothDogboneConfig,
    x_partitions: int,
    overlap: float = 0.10,
    parallel_region_lengths: Sequence[float] | None = None,
) -> None:
    meshes = build_edge_extended_dogbone_modifiers(
        specimen_cfg,
        x_partitions=x_partitions,
        overlap=overlap,
        parallel_region_lengths=parallel_region_lengths,
    )
    write_modifier_meshes(output_dir, meshes)


def generate_area1_modifiers_stl(
    output_dir: Path,
    x_partitions: int,
    modifier_cfg: ModifierConfig | None = None,
) -> None:
    modifier_cfg = modifier_cfg or ModifierConfig()
    meshes = build_modifier_meshes_1d(modifier_cfg, x_partitions=x_partitions)
    write_modifier_meshes(output_dir=output_dir, modifier_meshes=meshes)


def generate_area2_modifiers_stl(
    output_dir: Path,
    x_partitions: int,
    y_partitions: int,
    modifier_cfg: ModifierConfig | None = None,
) -> None:
    modifier_cfg = modifier_cfg or ModifierConfig()
    meshes = build_modifier_meshes_2d(
        modifier_cfg,
        x_partitions=x_partitions,
        y_partitions=y_partitions,
    )
    write_modifier_meshes(output_dir=output_dir, modifier_meshes=meshes)


def generate_rectangular_gradient_stl(
    output_dir: Path,
    cfg: RectangularGradientConfig | None = None,
    specimen_filename: str = "rectangular_specimen.stl",
    regions_subdir: str = "regions",
) -> None:
    cfg = cfg or RectangularGradientConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    specimen_poly = build_rectangular_specimen_polygon(
        specimen_length=cfg.specimen_length,
        specimen_width=cfg.specimen_width,
    )
    vertices, faces = build_extruded_mesh(specimen_poly, cfg.thickness)
    solid_name = specimen_filename.replace(".stl", "")
    write_ascii_stl(output_dir / specimen_filename, solid_name, vertices, faces)

    region_meshes = build_rectangular_partition_meshes(cfg)
    write_modifier_meshes(output_dir / regions_subdir, region_meshes)
