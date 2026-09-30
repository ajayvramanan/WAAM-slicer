"""
WAAM Slicer Core
================
Slices an STL mesh into layers with configurable infill strategies.

Design principle:
  - Always compute wall passes first (perimeter offsets inward)
  - Fill whatever interior space remains after walls
  - If nothing remains, walls alone cover the cross section
  - No thin/thick/transitional classification — it is not needed
  - Classification was causing layers to be skipped on tapered geometry

Dependencies:
    pip install trimesh numpy shapely matplotlib scipy
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from shapely.geometry import (Polygon, LineString, MultiLineString,
                               MultiPolygon, Point)
from shapely.affinity import rotate as shapely_rotate
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import math

# Bead geometry models — graceful fallback if file not yet on path
try:
    from waam_bead_models import BeadModel as _BeadModel, HEIGHT_MIN, HEIGHT_MAX
    _BEAD_MODEL_AVAILABLE = True
except ImportError:
    _BEAD_MODEL_AVAILABLE = False
    HEIGHT_MIN = 2.042
    HEIGHT_MAX = 2.565


# ─────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────

class InfillType(Enum):
    RASTER  = "raster"
    CONTOUR = "contour"


class LineOrder(Enum):
    SEQUENTIAL  = "sequential"
    ALTERNATING = "alternating"
    INTERLEAVED = "interleaved"
    RANDOM      = "random"
    CUSTOM      = "custom"


class DirectionMode(Enum):
    SAME          = "same"
    BOUSTROPHEDON = "boustrophedon"
    RANDOM        = "random"


class SeamMode(Enum):
    FIXED  = "fixed"
    ROTATE = "rotate"
    RANDOM = "random"


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

@dataclass
class SlicerConfig:

    # ── Layer ──────────────────────────────────
    layer_height: float           = 2.0
    adaptive_layers: bool         = False

    # Adaptive layer parameters (used when adaptive_layers = True).
    bead_height: float            = 2.565   # h_b: max achievable bead height (mm)
                                            # = HEIGHT_MAX from BeadModel (3.0 mm/sec)
    overhang_limit_deg: float     = 20.0
    # C_max = 1.0 mm — consistent with Dolenc & Makela (1994) reported range.
    # Produces visually distinct layer schedule on curved geometry.
    cusp_height_max: float        = 1.0

    # When True and adaptive_layers=True, inverts the DoE bead model to
    # compute weld travel speed per layer. Speed written as R[weld_speed_reg]=xx
    # at the start of each layer in the LS program.
    use_bead_model: bool          = True

    # ── Bead ───────────────────────────────────
    bead_width: float             = 6.0
    # Overlap ratio: fraction of bead width that adjacent beads share.
    # 30% is literature-recommended for flat GMAW beads (Xiong et al. 2019).
    # 50% causes height accumulation — do not use for solid fills.
    overlap_pct: float            = 30.0

    # ── Walls ──────────────────────────────────
    num_contours: int             = 2       # 0 = no walls, fill full polygon
    seam_mode: SeamMode           = SeamMode.ROTATE
    seam_rotate_deg: float        = 67.0
    seam_fixed_deg: float         = 0.0
    # Thin wall direction alternation (boustrophedon for open paths).
    # For thin-wall cross-sections, the centreline is an OPEN path.
    # Seam rotation makes no sense for open paths — use direction
    # reversal instead. Even layers: start→end. Odd layers: end→start.
    thin_wall_alternating: bool   = True

    # ── Infill ─────────────────────────────────
    infill_type: InfillType       = InfillType.RASTER
    raster_angle: float           = 0.0
    angle_increment: float        = 90.0
    direction_mode: DirectionMode = DirectionMode.BOUSTROPHEDON
    line_order: LineOrder         = LineOrder.INTERLEAVED
    custom_order: List[int]       = field(default_factory=list)
    # Contour infill ring order.
    # False = inside-to-outside (recommended — avoids pyramid accumulation).
    # True  = outside-to-inside (legacy default, causes height buildup).
    contour_inset_start: bool     = False

    # ── Path quality ──────────────────────────
    # Minimum path length as fraction of bead_width.
    # Raster segments shorter than this are discarded.
    # Prevents arc restrikes on tiny edge stubs.
    min_path_length_frac: float    = 0.5

    # ── Fill mismatch ──────────────────────────
    # Gap fraction of bead_width below which Strategy 1 is used.
    # Above this, an extra edge pass is added at adjusted speed.
    mismatch_threshold_frac: float = 0.5
    # Speed multiplier for the extra edge pass (relative to weld speed).
    # >1.0 = faster = narrower bead. Calibrate from your DoE.
    edge_speed_multiplier: float   = 1.5
    # Weld speed (mm/sec) — used only for edge pass calculation.
    # Should match R[2] value you set in the LS generator.
    weld_speed_mms: float          = 4.0

    # ── Torch orientation / tilt ───────────────
    # ── Torch orientation ─────────────────────
    # Applied uniformly to all layers when torch_tilt_enabled = True.
    # Use for parts set up at an angle on the weld table, or for builds
    # where the torch must maintain a specific fixed orientation.
    # For vertical parts built straight up: leave torch_tilt_enabled=False.
    torch_tilt_enabled: bool      = False
    torch_W_base: float           = 0.0     # rotation around robot Z (deg)
    torch_P_base: float           = 0.0     # tilt from vertical (deg)
    torch_R_base: float           = -70.0   # torch spin, fixed by geometry

    # ── Adaptive directional tilt ──────────────
    # When enabled (and torch_tilt_enabled=True), W and P are computed
    # per-layer from the mesh's local overhang direction, instead of
    # using the fixed torch_W_base/torch_P_base for all layers.
    #
    # Only activates when the layer's overhang is concentrated in ONE
    # direction (e.g. hook/eye crown closure). For radially-distributed
    # overhang (annular parts, domes), falls back to vertical torch —
    # tilting toward one side of a symmetric part doesn't help and
    # risks continuous wrist rotation / limit errors.
    adaptive_tilt_enabled: bool    = False
    adaptive_tilt_p_max: float     = 25.0   # max tilt angle from vertical (deg)
    adaptive_tilt_w_max_step: float = 15.0  # max W change between layers (deg)
    # Directional consistency threshold: fraction of overhang area that
    # must point within ±30deg of the dominant direction to trigger tilt.
    # Below this, overhang is too distributed (annular-like) — stay vertical.
    adaptive_tilt_consistency_min: float = 0.65

    # ── Visualisation ──────────────────────────
    show_contours: bool           = True
    show_infill: bool             = True
    colormap: str                 = "plasma"

    # ── Process time / wire calculator ─────────
    # These values feed print_summary() wire and time estimates.
    # Wire properties
    wire_diameter_mm: float       = 1.2     # mm (your ER308L / SS wire)
    material_density: float       = 8.0     # g/cc (SS = 8.0, mild steel = 7.85)
    wire_feed_rate_m_min: float   = 4.0     # m/min WFR (from your WPS)
    deposition_efficiency: float  = 0.95    # fraction (0–1); accounts for spatter
    # Travel (air move) parameters
    travel_speed_mms: float       = 50.0    # mm/sec (R[1] in LS program)
    lift_clearance_mm: float      = 10.0    # mm Z lift between layers
    # Interlayer dwell
    dwell_time_sec: float         = 10.0    # sec (R[3] in LS program)


# ─────────────────────────────────────────────
#  Data containers
# ─────────────────────────────────────────────

@dataclass
class MismatchInfo:
    region_width_mm: float
    n_lines: int
    filled_width_mm: float
    gap_mm: float
    strategy: str        # "adjusted_spacing" or "extra_edge_pass"
    edge_speed: float    # mm/sec of edge pass (0 if strategy 1)


@dataclass
class GeometryProfile:
    """
    Geometry analysis of one cross-section layer.
    Used for parameter suggestions and adaptive slicing decisions.
    """
    z: float
    n_polygons: int
    has_annulus: bool          # any polygon has interior holes
    has_solid: bool            # any polygon is solid (no holes)
    has_multi_body: bool       # more than one disconnected polygon
    min_wall_thickness: float  # mm — thinnest feature across all polygons
    max_wall_thickness: float  # mm — thickest feature
    dominant_type: str         # "solid", "annular", "mixed", "multi_body"


@dataclass
class LayerData:
    index: int
    z: float
    walls: List       = field(default_factory=list)
    infill_paths: List = field(default_factory=list)
    edge_passes: List  = field(default_factory=list)   # (LineString, speed)
    mismatch_info: List = field(default_factory=list)
    wpr: tuple        = (0.0, 0.0, -70.0)  # (W, P, R) torch orientation
    # Per-layer weld speed from bead model inversion.
    # None = use global R[weld_speed_reg]. Float = write R[reg]=xx per layer.
    weld_speed_mms: Optional[float] = None


# ─────────────────────────────────────────────
#  Slicer
# ─────────────────────────────────────────────

class WAAMSlicer:

    def __init__(self, config: SlicerConfig = None):
        self.cfg              = config or SlicerConfig()
        self.mesh             = None
        self.layers           = []
        self.global_bounds    = None
        self._ls_dwell_sec    = 10.0   # interlayer dwell; updated by set_ls_dwell()
        self.geometry_profile = None   # set by analyse_geometry()

    # ── Mesh loading ──────────────────────────

    def load_stl(self, path: str):
        import os
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n  STL file not found: '{path}'\n"
                f"  Check the path and try again.\n"
            )
        loaded = trimesh.load(path)
        if isinstance(loaded, trimesh.Scene):
            print("  Multi-body STL detected — merging into single mesh.")
            meshes = [g for g in loaded.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise ValueError("STL contains no valid mesh geometry.")
            self.mesh = trimesh.util.concatenate(meshes)
        else:
            self.mesh = loaded
        if not self.mesh.is_watertight:
            print("  Warning: mesh is not watertight — slicing may produce gaps.")
        print(f"  Loaded  : {path}")
        b = self.mesh.bounds
        print(f"  Bounds  X:[{b[0,0]:.2f}, {b[1,0]:.2f}]  "
              f"Y:[{b[0,1]:.2f}, {b[1,1]:.2f}]  "
              f"Z:[{b[0,2]:.2f}, {b[1,2]:.2f}] mm")
        print(f"  Faces: {len(self.mesh.faces)}   "
              f"Vertices: {len(self.mesh.vertices)}")
        self._apply_z_offset()
        # Auto-analyse geometry for parameter suggestions
        print("  Analysing geometry...")
        self.analyse_geometry()
        return self

    def load_mesh_object(self, mesh):
        if isinstance(mesh, str):
            raise TypeError(
                f"\n  load_mesh_object() received a string: '{mesh}'\n"
                f"  Use load_stl('{mesh}') instead.\n"
            )
        self.mesh = mesh
        self._apply_z_offset()
        return self

    def _apply_z_offset(self):
        """Translate mesh so its base sits exactly at Z=0."""
        if self.mesh is None:
            return
        z_min = self.mesh.bounds[0, 2]
        if abs(z_min) > 1e-6:
            import trimesh.transformations as _tf
            self.mesh.apply_transform(
                _tf.translation_matrix([0.0, 0.0, -z_min]))
            print(f"  Z-offset applied: {-z_min:+.3f} mm "
                  f"(base translated to Z=0)")

    # ── Z levels — constant or adaptive ──────

    def _compute_z_levels(self) -> List[float]:
        """
        Returns the list of Z heights at which to slice.

        Constant mode:
          Uniform spacing = layer_height from z_min to z_max.

        Adaptive mode (adaptive_layers = True):
          Layer thickness varies using the Dolenc-Makela cusp height formula:

            t(z) = clamp( C_max / n_z_max(z),  t_min,  h_b )

          where:
            C_max   = cusp_height_max (must be < h_b for adaptation to occur)
                      default = 0.3 * h_b
            n_z_max = MAX |n_iz| of facets intersecting Z=z
                      (most horizontal surface = most constraining)
            h_b     = bead_height (maximum allowed layer)
            t_min   = 0.5 * C_max (minimum stable layer)

          KEY CONSTRAINT: C_max must be < h_b.
          If C_max >= h_b: t = C_max/nz >= h_b always → no adaptation.
          Default C_max = 0.3*h_b ensures adaptation occurs from nz > 0.3.

          Behaviour:
            Vertical wall (nz=0):   t → ∞ → h_b  (thick, no cusp constraint)
            45-deg surface (nz=0.5): t = C_max/0.5 → clamped to h_b or thinner
            Horizontal top (nz=1):  t = C_max    (thinnest layer)
        """
        cfg   = self.cfg
        z_min, z_max = self.mesh.bounds[:, 2]

        if not cfg.adaptive_layers:
            return list(np.arange(z_min + cfg.layer_height,
                                  z_max + cfg.layer_height,
                                  cfg.layer_height))

        # ── Adaptive mode ─────────────────────────────────────────
        h_b   = cfg.bead_height
        C_max = cfg.cusp_height_max if cfg.cusp_height_max > 0 else h_b * 0.3
        sin_amax = math.sin(math.radians(cfg.overhang_limit_deg))

        # ── Bead model setup ──────────────────────────────────────
        bead_model = None
        if getattr(cfg, 'use_bead_model', False):
            if _BEAD_MODEL_AVAILABLE:
                bead_model = _BeadModel()
                # Override h_b and t_min from physical bead model bounds
                h_b   = HEIGHT_MAX   # 2.565 mm at 3.0 mm/sec
                t_min = HEIGHT_MIN   # 2.042 mm at 8.0 mm/sec
                print(f"  Bead model loaded — bounds: "
                      f"t_min={t_min:.3f}mm  h_b={h_b:.3f}mm")
            else:
                print("  WARNING: use_bead_model=True but waam_bead_models.py "
                      "not found. Place it alongside waam_slicer.py.")
                t_min = max(h_b * 0.5, C_max * 0.5)
        else:
            t_min = max(h_b * 0.5, C_max * 0.5)

        if C_max >= h_b:
            print(f"  WARNING: C_max ({C_max:.2f}mm) >= h_b ({h_b:.2f}mm). "
                  f"Adaptive slicing will produce constant layers. "
                  f"Set C_max < {h_b:.2f}mm for adaptation.")

        print(f"  Adaptive slicing: h_b={h_b:.3f}mm  C_max={C_max:.3f}mm  "
              f"t_min={t_min:.3f}mm  alpha_max={cfg.overhang_limit_deg:.0f}deg")
        print(f"  Adaptation active for surfaces shallower than "
              f"{math.degrees(math.asin(min(C_max/h_b,1))):.1f}deg from horizontal")

        z_levels = []   # List of (z, weld_speed_mms or None)
        t0 = self._adaptive_layer_thickness_cmax(z_min, h_b, t_min, C_max)

        # Pre-compute width-floor speed once — same for all layers
        speed_width_floor = None
        if bead_model is not None and cfg.bead_width > 0:
            speed_width_floor = bead_model.speed_for_width_ml(cfg.bead_width)

        def _actual_bead_height(t_adaptive, spd):
            """
            Return the actual bead height at the assigned speed.
            When the width constraint dominates (speed > speed_for_height),
            the actual bead height is LESS than h_b — the bead is thinner
            because the torch travels faster. Using h_b here would
            over-estimate the Z step and cause a growing torch-to-deposit
            gap over successive layers.
            """
            if bead_model is not None and spd is not None:
                return bead_model.height_at_speed(spd)
            return t_adaptive

        # Compute speed for first layer
        def _layer_speed(t_val):
            if bead_model is None:
                return None
            spd_h = bead_model.speed_for_height(t_val)
            if speed_width_floor is not None:
                return round(max(spd_h, speed_width_floor), 2)
            return round(spd_h, 2)

        spd0 = _layer_speed(t0)
        bh0  = _actual_bead_height(t0, spd0)

        # Tilt correction for first layer
        dz0 = bh0
        if cfg.torch_tilt_enabled and cfg.adaptive_tilt_enabled:
            r0 = self._overhang_direction_at(z_min)
            if r0 is not None:
                _, sev0, _ = r0
                dz0 = bh0 * math.cos(math.radians(sev0))

        z  = z_min + dz0
        n_overhang = 0
        t_history  = [(z_min, t0)]

        while z < z_max:
            spd = _layer_speed(t0)
            z_levels.append((round(z, 6), spd))

            t = self._adaptive_layer_thickness_cmax(z, h_b, t_min, C_max)
            t_history.append((z, t))

            n_oh = self._overhang_count_at(z, sin_amax)
            if n_oh > 0:
                n_overhang += 1

            # ── Tilt-corrected Z increment ────────────────────────
            # Use ACTUAL bead height at the assigned speed, not the
            # adaptive schedule's t (which is clamped to h_b=2.565mm).
            # When the width constraint dominates at 5.22 mm/sec, the
            # actual bead height is 2.278mm — using h_b instead causes
            # a +0.287mm gap per layer, accumulating to 8mm over 28 layers.
            bh_next = _actual_bead_height(t, _layer_speed(t))
            dz = bh_next
            if cfg.torch_tilt_enabled and cfg.adaptive_tilt_enabled:
                r = self._overhang_direction_at(z)
                if r is not None:
                    _, severity_deg, _ = r
                    alpha_rad = math.radians(severity_deg)
                    dz = bh_next * math.cos(alpha_rad)
                    dz = max(dz, t_min * math.cos(alpha_rad))

            t0 = t
            z  = z + dz

        if n_overhang > 0:
            print(f"  WARNING: {n_overhang} layer(s) have true overhanging "
                  f"surfaces (angle < {cfg.overhang_limit_deg:.0f}deg). "
                  f"Monitor melt pool stability.")

        all_t   = [t for _, t in t_history]
        n_const = int((z_max - z_min) / h_b)
        speeds  = [spd for _, spd in z_levels if spd is not None]
        spd_str = (f"  speed {min(speeds):.1f}–{max(speeds):.1f} mm/sec"
                   if speeds else "")
        # Report actual Z steps (may differ from bead height if tilt-corrected)
        if len(z_levels) > 1:
            z_steps = [z_levels[i+1][0]-z_levels[i][0]
                       for i in range(len(z_levels)-1)]
            step_str = (f"  Z-step {min(z_steps):.3f}–{max(z_steps):.3f}mm"
                        if cfg.torch_tilt_enabled and cfg.adaptive_tilt_enabled
                        else "")
        else:
            step_str = ""
        print(f"  Adaptive: {len(z_levels)} layers  "
              f"(constant h_b gives {n_const})  "
              f"t: {min(all_t):.3f}–{max(all_t):.3f}mm"
              f"{step_str}{spd_str}")
        return z_levels

    def _nz_critical_at(self, z: float) -> Optional[float]:
        """
        Returns the area-weighted 90th-percentile |n_z| of upward-facing
        facets straddling height z — used for the cusp-height formula.

        Only considers upward-facing faces (nz_signed >= 0) because:
        - Downward-facing faces are overhangs, handled separately
        - The cusp-height formula needs the most horizontal UPWARD surface
          (the surface the next layer deposits onto)
        - Area-weighting suppresses tiny mesh artefact faces
        """
        verts   = self.mesh.vertices
        faces   = self.mesh.faces
        normals = self.mesh.face_normals
        areas   = self.mesh.area_faces

        z_verts = verts[:, 2]
        f_z_min = z_verts[faces].min(axis=1)
        f_z_max = z_verts[faces].max(axis=1)
        mask    = (f_z_min < z) & (f_z_max > z)

        if not np.any(mask):
            return None

        nz_signed = normals[mask, 2]
        nz_abs    = np.abs(nz_signed)
        ar_vals   = areas[mask]

        # Only upward-facing faces for cusp-height calculation
        up_mask = nz_signed >= 0
        if not np.any(up_mask):
            return None

        nz_up = nz_abs[up_mask]
        ar_up = ar_vals[up_mask]
        total_a = ar_up.sum()

        if total_a < 1e-9:
            return float(nz_up.max())

        # Area-weighted 90th percentile
        sort_idx = np.argsort(nz_up)
        cum_area = np.cumsum(ar_up[sort_idx])
        pct90_idx = min(np.searchsorted(cum_area, 0.90 * total_a),
                        len(nz_up) - 1)
        return float(nz_up[sort_idx[pct90_idx]])

    def _nz_overhang_at(self, z: float) -> float:
        """
        Returns overhang severity (degrees from horizontal) at height z.
        Only counts real downward-facing geometry — not mesh artefacts.

        Marching cubes on a cubic voxel grid produces artefact faces at
        exactly arcsin(1/sqrt(3)) = 35.26deg and arcsin(1/sqrt(2)) = 45deg.
        These are rejected by angle fingerprinting before reporting.
        """
        import math as _math
        verts   = self.mesh.vertices
        faces   = self.mesh.faces
        normals = self.mesh.face_normals
        areas   = self.mesh.area_faces

        z_verts = verts[:, 2]
        f_z_min = z_verts[faces].min(axis=1)
        f_z_max = z_verts[faces].max(axis=1)
        mask    = (f_z_min < z) & (f_z_max > z)

        if not np.any(mask):
            return 0.0

        nz_signed = normals[mask, 2]
        ar_vals   = areas[mask]
        total_a   = ar_vals.sum()

        down_mask = nz_signed < 0
        if not np.any(down_mask):
            return 0.0

        nz_down = np.abs(nz_signed[down_mask])
        ar_down = ar_vals[down_mask]

        # ── Marching-cubes artefact rejection ────────────────────────
        # MC artefact faces have |nz| at exactly these values (within tol):
        MC_NZ = np.array([
            _math.sin(_math.radians(35.264)),  # space diagonal of cube
            _math.sin(_math.radians(45.000)),  # face diagonal
            _math.sin(_math.radians(54.736)),  # complementary space diag
            1.000,                              # perfectly flat cap
        ])
        TOL = 0.04   # within ~2.3deg
        is_artefact = np.any(
            np.abs(nz_down[:, None] - MC_NZ[None, :]) < TOL, axis=1)
        real_mask   = ~is_artefact
        real_area   = ar_down[real_mask].sum()

        # Only report if real (non-artefact) overhang area > 2% of total
        if total_a > 0 and real_area / total_a < 0.02:
            return 0.0

        # Area-weighted average over real downward faces
        if not np.any(real_mask):
            return 0.0
        oh_deg = _math.degrees(_math.asin(
            min(float(np.average(nz_down[real_mask],
                                 weights=ar_down[real_mask])), 1.0)))
        return round(oh_deg, 1)

    def _overhang_count_at(self, z: float, sin_amax: float) -> int:
        """
        Count facets at height z that represent true overhangs.
        A true overhang has its normal pointing DOWNWARD (n_z_signed < 0)
        AND the surface angle from horizontal < alpha_max.
        This correctly excludes vertical walls (n_z ≈ 0) from overhang flags.
        """
        verts   = self.mesh.vertices
        faces   = self.mesh.faces
        normals = self.mesh.face_normals

        z_verts = verts[:, 2]
        f_z_min = z_verts[faces].min(axis=1)
        f_z_max = z_verts[faces].max(axis=1)
        mask    = (f_z_min < z) & (f_z_max > z)

        if not np.any(mask):
            return 0

        # True overhang: normal has downward component AND is below alpha_max
        # from horizontal (i.e. n_z > 0 when measured as abs value, but the
        # signed value is negative for a downward-facing surface)
        nz_signed = normals[mask, 2]
        ar_vals   = self.mesh.area_faces[mask]
        total_a   = ar_vals.sum()

        # Only count overhang faces whose cumulative area exceeds 2% of total
        # — suppresses tiny marching-cubes stairstepping artefacts
        overhang_mask = nz_signed < -sin_amax
        if not np.any(overhang_mask):
            return 0
        oh_area = ar_vals[overhang_mask].sum()
        if total_a > 0 and oh_area / total_a < 0.08:
            return 0   # artefact — less than 8% of face area
        return int(np.sum(overhang_mask))

    def _adaptive_layer_thickness_cmax(self, z: float,
                                       h_b: float,
                                       t_min: float,
                                       C_max: float) -> float:
        """
        Dolenc-Mäkelä adaptive layer thickness.

        t(z) = clamp( C_max / n_z_max,  t_min,  h_b )

        IMPORTANT: C_max must be < h_b for adaptation to occur.
          If C_max >= h_b: t = C_max/nz >= C_max >= h_b → always clips to h_b.
          If C_max = 0.3*h_b: t varies from 0.3*h_b (nz=1) to h_b (nz<=0.3).

        n_z_max = 0 (vertical wall): t → ∞, clipped to h_b. Thick layers.
        n_z_max = 1 (horizontal top): t = C_max. Thinnest layer.
        """
        nz = self._nz_critical_at(z)
        if nz is None or nz < 1e-6:
            return h_b   # vertical / no data → maximum layer

        t = C_max / nz
        return max(t_min, min(h_b, t))

    # Legacy signature kept for backward compat
    def _adaptive_layer_thickness(self, z, h_b, t_min, sin_amax):
        cfg   = self.cfg
        C_max = (cfg.cusp_height_max if cfg.cusp_height_max > 0
                 else h_b * 0.3)
        return self._adaptive_layer_thickness_cmax(z, h_b, t_min, C_max)

    def _critical_normal_at(self, z: float) -> Optional[tuple]:
        """
        Return the full normal vector (nx, ny, nz) of the facet with
        the largest |nz| among all facets straddling height z.

        This is the "most horizontal" surface — the one that constrains
        both the adaptive layer thickness AND the torch tilt direction.

        Returns None if no facets found at this height.
        """
        verts   = self.mesh.vertices
        faces   = self.mesh.faces
        normals = self.mesh.face_normals

        z_verts = verts[:, 2]
        f_z_min = z_verts[faces].min(axis=1)
        f_z_max = z_verts[faces].max(axis=1)
        mask    = (f_z_min < z) & (f_z_max > z)

        if not np.any(mask):
            return None

        nz_abs   = np.abs(normals[mask, 2])
        best_idx = int(np.argmax(nz_abs))        # facet with largest |nz|
        n        = normals[mask][best_idx]        # full normal vector (nx,ny,nz)
        return (float(n[0]), float(n[1]), float(n[2]))

    def _overhang_direction_at(self, z: float) -> Optional[tuple]:
        """
        Analyse overhang direction at height z using two complementary methods:

        Method A (mesh normals) — works for arch/crown geometries where the
        overhanging surface actually faces downward in the STL mesh.

        Method B (centroid shift) — works for tilted wall geometries where
        the wall leans in a consistent XY direction but the mesh has no
        downward-facing faces. Compares cross-section centroid at z vs z-dz
        to find the lean direction and magnitude.

        Returns None if no single-direction overhang is detected by either method.
        Returns (direction_deg, severity_deg, consistency_frac) otherwise.
        """
        import math as _math

        # ── Method A: downward mesh face normals ─────────────────────
        verts   = self.mesh.vertices
        faces   = self.mesh.faces
        normals = self.mesh.face_normals
        areas   = self.mesh.area_faces

        z_verts = verts[:, 2]
        f_z_min = z_verts[faces].min(axis=1)
        f_z_max = z_verts[faces].max(axis=1)
        mask    = (f_z_min < z) & (f_z_max > z)

        if np.any(mask):
            nx        = normals[mask, 0]
            ny        = normals[mask, 1]
            nz_signed = normals[mask, 2]
            ar_vals   = areas[mask]
            total_a   = ar_vals.sum()

            down_mask = nz_signed < -0.05
            if np.any(down_mask):
                nx_d  = nx[down_mask]
                ny_d  = ny[down_mask]
                nz_d  = np.abs(nz_signed[down_mask])
                ar_d  = ar_vals[down_mask]

                MC_NZ = np.array([
                    _math.sin(_math.radians(35.264)),
                    _math.sin(_math.radians(45.000)),
                    _math.sin(_math.radians(54.736)),
                    1.000,
                ])
                TOL = 0.04
                is_artefact = np.any(
                    np.abs(nz_d[:, None] - MC_NZ[None, :]) < TOL, axis=1)
                real = ~is_artefact

                if np.any(real):
                    nx_r, ny_r = nx_d[real], ny_d[real]
                    nz_r, ar_r = nz_d[real], ar_d[real]
                    real_area   = ar_r.sum()

                    if total_a > 0 and real_area / total_a >= 0.02:
                        sum_x = float(np.sum(nx_r * ar_r))
                        sum_y = float(np.sum(ny_r * ar_r))
                        mag   = _math.hypot(sum_x, sum_y)

                        if mag >= 1e-9:
                            direction_deg = _math.degrees(
                                _math.atan2(sum_y, sum_x)) % 360
                            face_angles   = np.degrees(
                                np.arctan2(ny_r, nx_r)) % 360
                            delta = np.abs(
                                (face_angles - direction_deg + 180) % 360 - 180)
                            consistent_mask = delta < 30.0
                            consistency = float(
                                ar_r[consistent_mask].sum() / real_area)

                            if consistency >= self.cfg.adaptive_tilt_consistency_min:
                                severity_deg = _math.degrees(_math.asin(min(
                                    float(np.average(
                                        nz_r[consistent_mask],
                                        weights=ar_r[consistent_mask])), 1.0)))
                                return (round(direction_deg, 1),
                                        round(severity_deg, 1),
                                        round(consistency, 2))

        # ── Method B: centroid shift between adjacent layers ─────────
        # Detects leaning walls where mesh normals don't show downward faces.
        # Uses 3D section vertices (NOT to_2D() which strips absolute XY
        # position and returns zero shift regardless of wall lean).
        dz = 2.5   # look-back step
        z_lo = z - dz
        if z_lo < 0:
            return None

        try:
            cs_hi = self.mesh.section(
                plane_origin=[0, 0, z],    plane_normal=[0, 0, 1])
            cs_lo = self.mesh.section(
                plane_origin=[0, 0, z_lo], plane_normal=[0, 0, 1])
            if cs_hi is None or cs_lo is None:
                return None

            vhi = cs_hi.vertices
            vlo = cs_lo.vertices

            cx_hi = float(vhi[:, 0].mean())
            cy_hi = float(vhi[:, 1].mean())
            cx_lo = float(vlo[:, 0].mean())
            cy_lo = float(vlo[:, 1].mean())

            shift_x   = cx_hi - cx_lo
            shift_y   = cy_hi - cy_lo
            shift_mag = _math.hypot(shift_x, shift_y)

            if shift_mag < 0.15:
                return None

            # ── Project onto dominant axis only ───────────────────
            # The wall leans in one primary direction (X or Y).
            # Cross-sectional clipping noise can produce a large
            # shift on the minor axis (especially at geometry
            # transitions / cap edges). Zero the minor axis when
            # it is less than 30% of the major axis magnitude.
            # This ensures W and P are never both non-zero from
            # a single-direction lean.
            if abs(shift_x) >= abs(shift_y):
                # Primarily X lean → W only, zero P contribution
                if abs(shift_y) < abs(shift_x) * 0.30:
                    shift_y = 0.0
            else:
                # Primarily Y lean → P only, zero W contribution
                if abs(shift_x) < abs(shift_y) * 0.30:
                    shift_x = 0.0
            shift_mag = _math.hypot(shift_x, shift_y)
            if shift_mag < 0.15:
                return None

            # ── Direction consistency with previous layer ──────────
            # If the direction reverses sharply (>120°) relative to
            # the prior layer, this is likely a geometry transition
            # (step end, top cap) — clamp to the previous direction
            # rather than following the reversal. The rate-limiter
            # in _compute_wpr_at will then gracefully reduce tilt.
            direction_deg = _math.degrees(_math.atan2(shift_y, shift_x)) % 360

            # ── Validate against mesh face normals at this Z ───────
            # The centroid shift must agree with the dominant face
            # normal XY direction at this height — if they differ by
            # more than 90°, the shift is a geometry transition
            # artefact (cross-section clipping at a step edge, top
            # cap, or geometry termination) rather than a real lean.
            try:
                f_zmins = self.mesh.vertices[:,2][self.mesh.faces].min(axis=1)
                f_zmaxs = self.mesh.vertices[:,2][self.mesh.faces].max(axis=1)
                near    = (f_zmins < z) & (f_zmaxs > z)
                if np.any(near):
                    fn = self.mesh.face_normals[near]
                    ar = self.mesh.area_faces[near]
                    # Area-weighted XY face normal direction
                    wx = float(np.sum(fn[:,0] * ar))
                    wy = float(np.sum(fn[:,1] * ar))
                    if _math.hypot(wx, wy) > 1e-6:
                        face_dir = _math.degrees(
                            _math.atan2(wy, wx)) % 360
                        delta = abs(((direction_deg - face_dir + 180)
                                     % 360) - 180)
                        if delta > 60:
                            return None   # shift disagrees with faces
            except Exception:
                pass

            lean_angle = _math.degrees(_math.atan2(shift_mag, dz))

            return (round(direction_deg, 1),
                    round(lean_angle, 1),
                    1.00)

        except Exception:
            return None

    def _compute_wpr_at(self, z: float) -> tuple:
        """
        Return torch orientation (W, P, R) for any layer at height z.

        DESIGN RATIONALE (revised):

        Surface-normal-driven tilt is NOT appropriate for standard WAAM
        because:

        1. For a rotationally symmetric part (bowl, capsule) built with
           the part axis vertical, the overhang is equally distributed
           around the circumference. Tilting toward one side helps that
           face but worsens the opposite. Net benefit = zero.
           The torch should remain vertical for symmetric builds.

        2. Torch tilt is genuinely needed only when:
           (a) The ENTIRE PART is intentionally set up at an angle on
               the weld table (part axis not vertical) — the user knows
               this and sets W, P accordingly as fixed offsets.
           (b) A branch grows at a non-vertical angle (non-planar
               slicing case) — handled separately per-branch, not here.

        3. Deriving W from mesh normals produces arbitrary, wildly
           varying values for symmetric parts (the argmax picks different
           facets at different Z heights) causing dangerous inter-layer
           base rotation.

        IMPLEMENTATION:
        When torch_tilt_enabled = True, return the user-specified
        (W_base, P_base, R_base) for ALL layers uniformly.
        These values represent the fixed torch offset for the build —
        set by the user based on part setup on the weld table.

        Example use cases:
          Vertical part, vertical torch:   W=0, P=0,  R=-70
          Part tilted 30° on table:        W=0, P=30, R=-70
          Part facing 45° in XY:           W=45, P=0, R=-70

        z is kept as a parameter for API compatibility with future
        extensions (e.g. per-branch orientation in non-planar slicing).
        """
        cfg = self.cfg
        if not cfg.torch_tilt_enabled:
            return (cfg.torch_W_base, cfg.torch_P_base, cfg.torch_R_base)

        if not cfg.adaptive_tilt_enabled:
            # Fixed tilt mode — same orientation for all layers
            return (round(cfg.torch_W_base, 3),
                    round(cfg.torch_P_base, 3),
                    round(cfg.torch_R_base, 3))

        # ── Adaptive directional tilt mode ────────────────────────
        result = self._overhang_direction_at(z)

        if result is None:
            # No single-direction overhang detected — vertical torch.
            W_target, P_target = cfg.torch_W_base, cfg.torch_P_base
        else:
            direction_deg, severity_deg, consistency = result

            # ── Decompose lean into W and P ───────────────────────
            # Robot axis convention (confirmed on Fanuc R-30iB,
            # observed from in front of robot, +Y toward robot):
            #   W = rotation about Y axis → tilts torch in XZ plane
            #   P = rotation about X axis → tilts torch in YZ plane
            #   R = rotation about Z axis → torch spin (fixed)
            #
            # Sign convention verified experimentally:
            #   Wall leans toward +X (away from robot, rightward)
            #   → correct tilt is W = −(lean_angle)
            #   Wall leans toward +Y (toward robot)
            #   → correct tilt is P = −(lean_angle)
            #
            # Physical meaning: to lean the torch TOP toward the
            # overhanging side, W and P must be negated relative to
            # the raw lean direction vector.
            #
            # direction_deg = XY angle of wall lean (0°=+X, 90°=+Y)
            import math as _math
            rad = _math.radians(direction_deg)
            lean_x = severity_deg * _math.cos(rad)   # X component
            lean_y = severity_deg * _math.sin(rad)   # Y component

            # Negate: torch must tilt TOWARD the lean — sign is opposite
            # to the lean direction in the robot's W/P convention.
            W_target = -lean_x   # +X lean → W negative
            P_target = -lean_y   # +Y lean → P negative

        # ── Cap magnitudes ────────────────────────────────────────
        p_max = cfg.adaptive_tilt_p_max
        W_target = max(-p_max, min(p_max, W_target))
        P_target = max(-p_max, min(p_max, P_target))

        # ── Rate-limit W change between layers ────────────────────
        # Works in signed degrees (−180 to +180) to handle negative W
        # correctly without wrapping through 360.
        W_prev = getattr(self, '_last_adaptive_W', 0.0)
        dW = W_target - W_prev
        max_step = cfg.adaptive_tilt_w_max_step
        if abs(dW) > max_step:
            dW = math.copysign(max_step, dW)
        W_final = W_prev + dW
        self._last_adaptive_W = W_final

        P_final = round(max(-p_max, min(p_max, P_target)), 2)

        return (round(W_final, 2), P_final, round(cfg.torch_R_base, 3))

    # ── Main slice loop ───────────────────────

    def slice(self):
        if self.mesh is None:
            raise RuntimeError("No mesh loaded.")

        z_levels    = self._compute_z_levels()
        self.layers = []
        cfg         = self.cfg
        self._last_adaptive_W         = 0.0
        self._last_method_b_direction = None

        print(f"\nSlicing {len(z_levels)} layers  "
              f"(height={cfg.layer_height}mm, "
              f"bead={cfg.bead_width}mm, "
              f"infill={cfg.infill_type.value})\n")

        for i, z_entry in enumerate(z_levels):
            # z_levels is List[float] in constant mode,
            # List[(z, weld_speed_mms|None)] in adaptive+bead-model mode.
            if isinstance(z_entry, tuple):
                z, layer_weld_speed = z_entry
            else:
                z, layer_weld_speed = z_entry, None

            cross = self.mesh.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1])
            if cross is None:
                continue
            try:
                slice_2d, transform = cross.to_planar()
            except Exception:
                continue

            polygons_raw = [p for p in slice_2d.polygons_full
                            if p.is_valid and not p.is_empty]
            if not polygons_raw:
                continue

            # ── Re-align polygons to world XY frame ───────────────
            # to_planar() uses an arbitrary 2D origin that shifts
            # between layers — causing apparent displacement in the
            # visualiser and 3D stack. We recover the true XY
            # position by applying the inverse of the planar transform.
            #
            # The transform is a 4x4 matrix mapping 2D plane coords
            # to 3D world coords. Columns 0 and 1 are the X and Y
            # basis vectors of the plane; column 3 is the plane origin.
            #
            # For a horizontal (Z=const) slice, the plane basis vectors
            # are just X and Y of world space, and the origin offset is
            # what shifts between layers. We extract the XY offset from
            # column 3 of the transform and apply it to all polygons.
            try:
                ox = float(transform[0, 3])  # X offset of plane origin
                oy = float(transform[1, 3])  # Y offset of plane origin
            except Exception:
                ox, oy = 0.0, 0.0

            from shapely.affinity import translate as shapely_translate
            polygons = []
            for p in polygons_raw:
                shifted = shapely_translate(p, xoff=ox, yoff=oy)
                if shifted.is_valid and not shifted.is_empty:
                    polygons.append(shifted)

            if not polygons:
                continue

            layer = LayerData(index=i, z=z)

            # ── Actual bead width at this layer's speed ───────────
            # All infill spacing uses actual deposited width from the
            # DoE model, not nominal cfg.bead_width. This ensures ring
            # spacing and raster step reflect what the process actually
            # deposits at the assigned travel speed.
            if _BEAD_MODEL_AVAILABLE and layer_weld_speed is not None:
                _bm_inst = _BeadModel()
                actual_bead_width = _bm_inst.width_ml_at_speed(layer_weld_speed)
            else:
                actual_bead_width = cfg.bead_width

            for poly in polygons:
                # ── Walls ─────────────────────────────────────────
                walls = self._compute_walls(poly, layer_index=i,
                                            actual_bead_width=actual_bead_width)
                layer.walls.extend(walls)

                # ── Infill region = whatever remains after walls ───
                infill_region = self._infill_region(
                    poly, layer_index=i, precomputed_walls=walls)
                if infill_region is None or infill_region.is_empty:
                    continue

                # ── Infill ─────────────────────────────────────────
                if cfg.infill_type == InfillType.RASTER:
                    angle = (cfg.raster_angle
                             + i * cfg.angle_increment) % 180
                    paths, edge_passes, mismatch = \
                        self._raster_infill(infill_region, angle, i,
                                            actual_bead_width=actual_bead_width)
                    layer.edge_passes.extend(edge_passes)
                    layer.mismatch_info.extend(mismatch)

                elif cfg.infill_type == InfillType.CONTOUR:
                    paths, edge_passes, mismatch = \
                        self._contour_infill(infill_region,
                                             actual_bead_width=actual_bead_width,
                                             layer_index=i)
                    layer.edge_passes.extend(edge_passes)
                    layer.mismatch_info.extend(mismatch)

                else:
                    paths = []

                layer.infill_paths.extend(paths)

            # ── Compute torch orientation for this layer ───────────
            wpr        = self._compute_wpr_at(z)
            layer.wpr  = wpr
            layer.weld_speed_mms = layer_weld_speed

            self.layers.append(layer)
            wpr_str = (f"  W={wpr[0]:.1f} P={wpr[1]:.1f} R={wpr[2]:.1f}"
                       if cfg.torch_tilt_enabled else "")
            spd_str = (f"  spd={layer_weld_speed:.1f}mm/s"
                       if layer_weld_speed is not None else "")
            print(f"  Layer {i+1:>4}/{len(z_levels)}  "
                  f"z={z:.2f}mm  "
                  f"walls={len(layer.walls)}  "
                  f"paths={len(layer.infill_paths)}  "
                  f"edge={len(layer.edge_passes)}"
                  f"{spd_str}{wpr_str}")

        print(f"\nSlicing complete — {len(self.layers)} layers generated.")
        self._compute_global_bounds()
        self._validate_parameters()
        return self

    # ── Bounds (from actual slice data) ───────

    def _compute_global_bounds(self):
        """
        Compute XY bounds from actual sliced coordinates.
        Must use slice data — not mesh bounds — because to_planar()
        recentres cross sections in their own local 2D frame.
        """
        all_x, all_y = [], []
        for layer in self.layers:
            for path in layer.infill_paths:
                try:
                    xs, ys = zip(*path.coords)
                    all_x.extend(xs); all_y.extend(ys)
                except Exception:
                    pass
            for wall in layer.walls:
                from shapely.geometry import LineString as _LS
                if isinstance(wall, _LS):
                    try:
                        xs, ys = zip(*wall.coords)
                        all_x.extend(xs); all_y.extend(ys)
                    except Exception:
                        pass
                    continue
                geoms = wall.geoms if hasattr(wall, "geoms") else [wall]
                for g in geoms:
                    try:
                        x, y = g.exterior.xy
                        all_x.extend(x); all_y.extend(y)
                    except Exception:
                        pass
            for ep, _ in layer.edge_passes:
                try:
                    xs, ys = zip(*ep.coords)
                    all_x.extend(xs); all_y.extend(ys)
                except Exception:
                    pass

        if all_x and all_y:
            pad = self.cfg.bead_width
            self.global_bounds = (
                min(all_x) - pad, max(all_x) + pad,
                min(all_y) - pad, max(all_y) + pad,
            )

    # ── Geometry analysis and suggestions ────

    def analyse_geometry(self) -> dict:
        """
        Analyses the mesh geometry to classify it and generate
        parameter suggestions. Called automatically after load_stl()
        or can be called manually before slice().

        Samples a few Z levels and examines the cross-section polygons
        to determine dominant geometry type:
          "solid"      — all polygons are solid (no holes)
          "annular"    — dominant polygons are ring-shaped (hollow walls)
          "mixed"      — mix of solid and annular polygons per layer
          "multi_body" — multiple disconnected polygons per layer
                         (bracket-type parts)

        Returns a dict with geometry info and parameter suggestions.
        Stores result in self.geometry_profile for UI access.
        """
        if self.mesh is None:
            return {}

        b       = self.mesh.bounds
        z_min   = b[0, 2]
        z_max   = b[1, 2]
        z_range = z_max - z_min

        # Sample at 5 evenly spaced Z levels across the middle 80% of height
        sample_zs = [z_min + z_range * f
                     for f in [0.1, 0.25, 0.5, 0.75, 0.9]]

        n_solid    = 0
        n_annular  = 0
        n_multi    = 0
        min_widths = []
        max_widths = []

        for z in sample_zs:
            cross = self.mesh.section(
                plane_origin=[0, 0, z],
                plane_normal=[0, 0, 1])
            if cross is None:
                continue
            try:
                slice_2d, transform = cross.to_planar()
                from shapely.affinity import translate as _tr
                try:
                    ox = float(transform[0, 3])
                    oy = float(transform[1, 3])
                except Exception:
                    ox, oy = 0.0, 0.0

                polys = [_tr(p, xoff=ox, yoff=oy)
                         for p in slice_2d.polygons_full
                         if p.is_valid and not p.is_empty]
            except Exception:
                continue

            if not polys:
                continue

            if len(polys) > 1:
                n_multi += 1

            for poly in polys:
                has_holes = (hasattr(poly, 'interiors') and
                             len(list(poly.interiors)) > 0)
                if has_holes:
                    n_annular += 1
                else:
                    n_solid += 1

                w = self._poly_min_width(poly)
                min_widths.append(w)
                max_widths.append(w)

        total = n_solid + n_annular
        if total == 0:
            return {}

        ann_frac     = n_annular / total
        global_min_w = min(min_widths) if min_widths else 0
        global_max_w = max(max_widths) if max_widths else 0
        bw           = self.cfg.bead_width

        # ── Classification ────────────────────────────────────────
        # thin_wall: solid polygons (no holes) whose minimum width
        #   is comparable to bead_width. These are single-bead walls
        #   — a bowl fragment, a thin curved panel, a partial shell.
        #   The inscribed-circle width is <= 1.5 * bead_width so
        #   even one wall pass covers the full cross-section.
        #   Note: annular polygons with similar thickness are already
        #   caught by the annulus single-pass fallback, so we only
        #   need this for solid thin polygons.
        # The 1.5 factor gives a 50% margin over bead_width —
        # accommodating manufacturing tolerances in wall thickness.
        is_thin_wall = (ann_frac <= 0.2 and           # mostly solid (no holes)
                        global_min_w > 0 and           # valid measurement
                        global_min_w <= bw * 1.5)      # thin relative to bead

        if n_multi >= 3:
            dom_type = "multi_body"
        elif ann_frac >= 0.7:
            dom_type = "annular"
        elif is_thin_wall:
            dom_type = "thin_wall"
        elif ann_frac <= 0.2:
            dom_type = "solid"
        else:
            dom_type = "mixed"

        suggestions = self._generate_suggestions(
            dom_type, global_min_w, global_max_w, ann_frac, n_multi, bw)

        result = {
            "dominant_type":      dom_type,
            "solid_count":        n_solid,
            "annular_count":      n_annular,
            "multi_body_layers":  n_multi,
            "annular_fraction":   round(ann_frac, 2),
            "min_wall_thickness": round(global_min_w, 2),
            "max_wall_thickness": round(global_max_w, 2),
            "is_thin_wall":       is_thin_wall,
            "suggestions":        suggestions,
        }
        self.geometry_profile = result
        return result

    def _generate_suggestions(self, dom_type: str,
                               min_w: float,
                               max_w: float,
                               ann_frac: float,
                               n_multi: int,
                               bw: float = None) -> List[dict]:
        """
        Returns a list of parameter suggestion dicts, each with:
          param    : parameter name as used in SlicerConfig
          value    : recommended value (string)
          reason   : plain-English explanation
          severity : "ok" | "warn" | "critical"
        """
        cfg         = self.cfg
        if bw is None:
            bw = cfg.bead_width
        step        = bw * (1 - cfg.overlap_pct / 100)
        suggestions = []

        if dom_type == "thin_wall":
            # ── Thin wall (solid polygon, width ≈ bead_width) ────────
            # e.g. bowl fragment, curved panel, partial shell.
            # The wall thickness is 1-1.5x bead width — one pass covers all.
            suggestions.append({
                "param":    "num_walls",
                "value":    "1",
                "reason":   f"Thin wall geometry detected (wall thickness "
                            f"~{min_w:.1f}mm ≈ bead width {bw:.1f}mm). "
                            f"A single wall pass covers the full cross-section. "
                            f"Additional wall passes or infill will OVERFILL the wall.",
                "severity": "ok"
            })
            suggestions.append({
                "param":    "infill_type",
                "value":    "none (walls only)",
                "reason":   "No infill needed — the wall pass IS the deposit. "
                            "Contour or raster infill on a thin wall will attempt "
                            "to fill an already-full cross-section, producing "
                            "excess deposition and dimensional error.",
                "severity": "warn"
            })
            bw_match = abs(min_w - bw) <= bw * 0.3
            suggestions.append({
                "param":    "bead_width",
                "value":    f"{min_w:.1f} mm" if not bw_match else f"{bw:.1f} mm (current)",
                "reason":   (f"Wall thickness ({min_w:.1f}mm) is well matched to "
                             f"bead_width ({bw:.1f}mm). Good — one bead covers the wall."
                             if bw_match else
                             f"Wall thickness ({min_w:.1f}mm) differs from bead_width "
                             f"({bw:.1f}mm). Set bead_width ≈ wall thickness so a single "
                             f"bead covers the full wall width without gaps or overfill."),
                "severity": "ok" if bw_match else "warn"
            })
            suggestions.append({
                "param":    "seam_mode",
                "value":    "rotate",
                "reason":   "Rotating seam distributes the wall start/stop defect "
                            "along the curved surface. Especially important for "
                            "curved thin-wall parts like bowl fragments.",
                "severity": "ok"
            })

        elif dom_type == "solid":
            # Solid bulk part — standard wall + infill works well
            suggestions.append({
                "param":    "num_walls",
                "value":    "1–2",
                "reason":   "Solid cross-section. One or two wall passes followed "
                            "by raster or contour infill is the standard approach. "
                            "Walls define the outer surface; infill fills the bulk.",
                "severity": "ok"
            })
            suggestions.append({
                "param":    "infill_type",
                "value":    "raster or contour",
                "reason":   "Both strategies work for solid geometry. Raster with "
                            "90° angle increment balances heat distribution. "
                            "Contour produces better surface on curved profiles.",
                "severity": "ok"
            })
            if min_w < bw:
                suggestions.append({
                    "param":    "bead_width",
                    "value":    f"{min_w * 0.9:.1f} mm",
                    "reason":   f"Part has features as thin as {min_w:.1f} mm — "
                                f"narrower than current bead width ({bw:.1f} mm). "
                                f"Reduce bead_width to match the thinnest feature, "
                                f"or those regions will be over-deposited.",
                    "severity": "warn"
                })
            else:
                suggestions.append({
                    "param":    "bead_width",
                    "value":    f"{bw:.1f} mm (current)",
                    "reason":   f"Minimum feature width ({min_w:.1f} mm) is larger "
                                f"than bead width ({bw:.1f} mm). Current setting is fine.",
                    "severity": "ok"
                })

        elif dom_type == "annular":
            # Hollow wall — bowl, pipe, pressure vessel
            suggestions.append({
                "param":    "num_walls",
                "value":    "0 or 1",
                "reason":   "Hollow wall cross-section detected. The wall IS the feature. "
                            "With num_walls=0 and contour infill, the infill traces the "
                            "ring boundary directly — clean and correct. With num_walls=1, "
                            "one perimeter pass is placed first, then any remaining "
                            "interior is filled by contour.",
                "severity": "ok"
            })
            # Determine if raster+walls would be problematic
            raster_walls_ok = (min_w > bw * 2.2)
            suggestions.append({
                "param":    "infill_type",
                "value":    "contour (recommended)",
                "reason":   "Contour infill follows the ring shape naturally and is "
                            "correct for hollow wall parts. "
                            + ("Raster with walls is also possible here since the wall "
                               "is thick enough to buffer safely."
                               if raster_walls_ok else
                               "AVOID raster + walls: when wall thickness ≈ bead width, "
                               "buffer operations on ring polygons cause boundary collision "
                               "producing spurious paths. Raster with num_walls=0 is "
                               "acceptable if contour is not preferred."),
                "severity": "ok" if raster_walls_ok else "warn"
            })
            bw_match = abs(min_w - bw) <= bw * 0.3
            suggestions.append({
                "param":    "bead_width",
                "value":    f"{min_w:.1f} mm" if not bw_match else f"{bw:.1f} mm (current)",
                "reason":   (f"Wall thickness ({min_w:.1f} mm) is well matched to "
                             f"bead_width ({bw:.1f} mm). Good — one bead covers the wall."
                             if bw_match else
                             f"Wall thickness ({min_w:.1f} mm) differs from bead_width "
                             f"({bw:.1f} mm). For single-bead hollow wall deposition, "
                             f"set bead_width ≈ wall thickness so one bead covers the "
                             f"full wall width cleanly."),
                "severity": "ok" if bw_match else "warn"
            })
            suggestions.append({
                "param":    "seam_mode",
                "value":    "rotate",
                "reason":   "Rotating seam (67°/layer) distributes the wall start/stop "
                            "defect around the circumference. Fixed seam creates a "
                            "visible vertical line on circular/elliptic hollow parts.",
                "severity": "ok"
            })

        elif dom_type == "multi_body":
            # Bracket-type with multiple disconnected polygons per layer
            suggestions.append({
                "param":    "infill_type",
                "value":    "raster with num_contours=0",
                "reason":   "Multiple disconnected polygons detected (bracket/frame "
                            "geometry). Raster with 0 walls handles each arm polygon "
                            "independently and fills them correctly. Wall passes on "
                            "thin arm polygons collapse to spurious geometry.",
                "severity": "warn"
            })
            suggestions.append({
                "param":    "USE BRACKET SLICER",
                "value":    "Run Bracket Slicer instead",
                "reason":   "Frame/bracket geometry with multiple arms is best handled "
                            "by the dedicated Bracket Slicer (bottom button row). It "
                            "computes centreline paths per arm and plans traversal to "
                            "minimise junction overlap — better than the general slicer "
                            "for this geometry class.",
                "severity": "warn"
            })
            if min_w < bw:
                suggestions.append({
                    "param":    "bead_width",
                    "value":    f"{min_w:.1f} mm",
                    "reason":   f"Arm thickness ({min_w:.1f} mm) is narrower than "
                                f"current bead width ({bw:.1f} mm). Set bead_width "
                                f"to match arm thickness for single-bead arm deposition.",
                    "severity": "critical"
                })

        elif dom_type == "mixed":
            # Solid base transitioning to hollow walls (e.g. tapered vessel)
            suggestions.append({
                "param":    "num_walls",
                "value":    "1",
                "reason":   "Mixed geometry (solid base + hollow wall upper section). "
                            "One wall pass works for both: on solid layers it defines "
                            "the perimeter; on annular layers it traces the ring. "
                            "Two wall passes may over-fill thin annular layers.",
                "severity": "ok"
            })
            suggestions.append({
                "param":    "infill_type",
                "value":    "contour",
                "reason":   "Contour infill handles both solid and annular layers "
                            "correctly. On solid layers it spirals inward; on annular "
                            "layers the contour loop guard prevents spurious paths.",
                "severity": "ok"
            })
            suggestions.append({
                "param":    "adaptive_layers",
                "value":    "True",
                "reason":   "Mixed geometry with curved transitions benefits from "
                            "adaptive layer thickness. Thin layers at curved surfaces "
                            "reduce staircase error; thick layers on straight walls "
                            "maximise build speed.",
                "severity": "ok"
            })

        return suggestions

    # ── Parameter validation ──────────────────

    def _validate_parameters(self):
        """
        After slicing, check whether configured parameters suit the geometry.
        Prints warnings and suggestions if mismatches are found.
        """
        cfg  = self.cfg
        step = cfg.bead_width * (1 - cfg.overlap_pct / 100)

        min_widths = []
        for layer in self.layers:
            cross = self.mesh.section(
                plane_origin=[0, 0, layer.z],
                plane_normal=[0, 0, 1])
            if cross is None:
                continue
            try:
                slice_2d, _ = cross.to_planar()
                for poly in slice_2d.polygons_full:
                    if poly.is_valid and not poly.is_empty:
                        min_widths.append(self._poly_min_width(poly))
            except Exception:
                pass

        if not min_widths:
            return

        global_min_w = min(min_widths)
        global_max_w = max(min_widths)

        print("\n" + "-"*50)
        print("  Parameter Validation Report")
        print("-"*50)
        print(f"  Bead width         : {cfg.bead_width:.1f} mm")
        print(f"  Overlap            : {cfg.overlap_pct:.1f} %")
        print(f"  Wall step          : {step:.2f} mm")
        print(f"  Num wall passes    : {cfg.num_contours}")
        print(f"  Min wall thickness : {global_min_w:.2f} mm (across all layers)")
        print(f"  Max wall thickness : {global_max_w:.2f} mm")

        min_w_for_one_wall = step * 2
        min_w_for_n_walls  = step * 2 * max(cfg.num_contours, 1)

        if global_min_w < cfg.bead_width:
            print(f"\n  WARNING: Some cross sections ({global_min_w:.2f}mm) "
                  f"are narrower than one bead width ({cfg.bead_width:.1f}mm).")
            print(f"  Centred reduced-step pass will be used for those layers.")
            print(f"  Suggestion: set bead_width to "
                  f"{global_min_w * 0.9:.1f}mm or use faster travel speed.")
        elif global_min_w < min_w_for_one_wall:
            print(f"\n  WARNING: Some cross sections ({global_min_w:.2f}mm) "
                  f"are too narrow for standard wall step ({step:.2f}mm each side).")
            sugg_bw = global_min_w / 2.1 / (1 - cfg.overlap_pct / 100)
            print(f"  Suggestion: reduce bead_width to ~{sugg_bw:.1f}mm "
                  f"or reduce num_contours to 1.")
        elif global_min_w < min_w_for_n_walls:
            n_fit = max(1, int(global_min_w / (step * 2)))
            print(f"\n  INFO: Some cross sections ({global_min_w:.2f}mm) "
                  f"fit only {n_fit} of {cfg.num_contours} requested wall passes.")
            print(f"  Suggestion: reduce num_contours to {n_fit}.")
        else:
            print(f"\n  OK: Wall parameters compatible with all cross sections.")

        if cfg.infill_type == InfillType.RASTER and cfg.num_contours > 0:
            typical_infill_w = global_max_w - 2 * step * cfg.num_contours
            if typical_infill_w < cfg.bead_width:
                print(f"\n  INFO: After {cfg.num_contours} wall pass(es), "
                      f"most layers will have walls only (no raster infill).")
                print(f"  This is correct for thin-walled parts.")

        if len(self.layers) < 3:
            print(f"\n  WARNING: Only {len(self.layers)} layers generated. "
                  f"Check layer_height vs part height.")

        print("-"*50 + "\n")

    # ── Seam angle ────────────────────────────

    def _seam_angle(self, layer_index: int) -> float:
        cfg = self.cfg
        if cfg.seam_mode == SeamMode.FIXED:
            return cfg.seam_fixed_deg
        elif cfg.seam_mode == SeamMode.ROTATE:
            return (layer_index * cfg.seam_rotate_deg) % 360
        elif cfg.seam_mode == SeamMode.RANDOM:
            rng = random.Random(layer_index * 7919)
            return rng.uniform(0, 360)
        return 0.0

    def _point_on_polygon_at_angle(self, polygon: Polygon,
                                    angle_deg: float) -> Point:
        cx, cy    = polygon.centroid.x, polygon.centroid.y
        angle_rad = np.radians(angle_deg)
        far       = 1e6
        ray       = LineString([
            (cx, cy),
            (cx + far * np.cos(angle_rad),
             cy + far * np.sin(angle_rad))
        ])
        intersection = ray.intersection(polygon.exterior)
        if intersection.is_empty:
            return Point(polygon.exterior.coords[0])
        if intersection.geom_type == "MultiPoint":
            return intersection.geoms[0]
        if intersection.geom_type == "Point":
            return intersection
        return Point(polygon.exterior.coords[0])

    def _reorder_ring_to_seam(self, polygon: Polygon,
                               seam_deg: float) -> List[Tuple]:
        coords = list(polygon.exterior.coords[:-1])
        n      = len(coords)
        if n == 0:
            return coords
        seam_pt   = self._point_on_polygon_at_angle(polygon, seam_deg)
        sx, sy    = seam_pt.x, seam_pt.y
        nearest_i = min(range(n),
                        key=lambda i: (coords[i][0]-sx)**2
                                    + (coords[i][1]-sy)**2)
        reordered = coords[nearest_i:] + coords[:nearest_i]
        reordered.append(reordered[0])
        return reordered

    # ── Walls ─────────────────────────────────

    def _thin_wall_centreline(self, polygon: Polygon,
                               layer_index: int) -> Optional['LineString']:
        """
        Extract the centreline of a thin solid polygon for single-bead
        wall deposition.

        For straight/gently-curved thin walls (including tilted wall
        cross-sections which produce parallelograms/hexagons), we use
        the polygon's minimum bounding rectangle (OBB) to find the long
        axis and return a clean 2-point centreline along it.

        This avoids the extra stub vertices that appeared when the
        erosion-based algorithm traced through the diagonal corners of
        a tilted wall's hexagonal cross-section.

        For complex curved shapes (bowls, crescents) where the OBB
        long axis doesn't adequately represent the path, we fall back
        to the erosion-based method.
        """
        cfg = self.cfg
        w   = self._poly_min_width(polygon)
        if w < 0.1:
            return None

        # ── OBB-based centreline (fast, robust for straight walls) ──
        # The minimum bounding rectangle (mrr) gives us the long axis
        # directly. Centroid of each short side = the two endpoints.
        # For curved shapes (arc sections, C-profiles), the OBB long
        # axis is a straight line — not the arc — so we check vertex
        # deviation from the centreline and fall through to erosion
        # if the shape is significantly curved.
        try:
            mrr = polygon.minimum_rotated_rectangle
            mrr_coords = list(mrr.exterior.coords)[:-1]  # 4 corners

            if len(mrr_coords) == 4:
                def slen(i, j):
                    return math.hypot(mrr_coords[j][0]-mrr_coords[i][0],
                                      mrr_coords[j][1]-mrr_coords[i][1])
                s01 = slen(0,1); s12 = slen(1,2)

                if s01 >= s12:
                    mid_a = ((mrr_coords[1][0]+mrr_coords[2][0])/2,
                             (mrr_coords[1][1]+mrr_coords[2][1])/2)
                    mid_b = ((mrr_coords[3][0]+mrr_coords[0][0])/2,
                             (mrr_coords[3][1]+mrr_coords[0][1])/2)
                    short_side = s12
                else:
                    mid_a = ((mrr_coords[0][0]+mrr_coords[1][0])/2,
                             (mrr_coords[0][1]+mrr_coords[1][1])/2)
                    mid_b = ((mrr_coords[2][0]+mrr_coords[3][0])/2,
                             (mrr_coords[2][1]+mrr_coords[3][1])/2)
                    short_side = s01

                # ── Curvature check ───────────────────────────────
                # Straight walls (flat-faced solids cut horizontally)
                # produce cross-sections with very few polygon vertices:
                # typically 4–16 (a parallelogram or hexagon).
                # Curved shapes (arc sections, nozzles, vases) produce
                # many vertices: 40–400 depending on mesh resolution.
                # Vertex count is the most reliable discriminator.
                n_poly_verts = len(polygon.exterior.coords) - 1
                if n_poly_verts > 20:
                    raise ValueError("curved shape — use erosion")

                obb_area  = mrr.area
                poly_area = polygon.area
                if obb_area > 0 and poly_area / obb_area > 0.55:
                    pts = [mid_a, mid_b]
                    if cfg.thin_wall_alternating and layer_index % 2 == 1:
                        pts = pts[::-1]
                    return LineString(pts)
        except Exception:
            pass

        # ── Erosion-based fallback (curved shapes, bowls, crescents) ─
        epsilon = 0.15
        inset_r = max(0.01, w / 2.0 - epsilon)

        core = None
        try:
            core = polygon.buffer(-inset_r)
            if core is None or core.is_empty:
                core = None
        except Exception:
            core = None

        if core is None:
            return self._centreline_midpoint_fallback(polygon, layer_index)

        try:
            if hasattr(core, 'geoms') and core.geom_type == 'MultiPolygon':
                core = max(core.geoms, key=lambda g: g.exterior.length)
        except Exception:
            pass

        try:
            coords = list(core.exterior.coords)
        except Exception:
            return self._centreline_midpoint_fallback(polygon, layer_index)

        n = len(coords) - 1
        if n < 2:
            return self._centreline_midpoint_fallback(polygon, layer_index)

        x0, y0 = coords[0]
        dists   = [math.hypot(coords[i][0]-x0, coords[i][1]-y0)
                   for i in range(n)]
        split_i = dists.index(max(dists))

        if split_i < 1 or split_i >= n - 1:
            half1 = coords[:n]
        else:
            half1 = coords[:split_i + 1]
            half2 = coords[split_i:]

            def arc_length(pts):
                return sum(math.hypot(pts[i+1][0]-pts[i][0],
                                      pts[i+1][1]-pts[i][1])
                           for i in range(len(pts)-1))

            half1 = half1 if arc_length(half1) >= arc_length(half2) else half2

        if len(half1) < 2:
            return self._centreline_midpoint_fallback(polygon, layer_index)

        if cfg.thin_wall_alternating and layer_index % 2 == 1:
            half1 = half1[::-1]

        return LineString(half1)

    def _centreline_midpoint_fallback(self, polygon: Polygon,
                                       layer_index: int
                                       ) -> Optional['LineString']:
        """
        Fallback centreline extraction via boundary midpoint sampling.

        Used when erosion-based extraction fails (very concave shapes,
        degenerate geometry near the pole of a bowl fragment).

        Samples N points around the exterior boundary. For each sampled
        point, finds the nearest point on the boundary that is roughly
        on the "opposite side" of the strip (distance ≈ wall thickness
        in the perpendicular direction). Takes the midpoint of each pair.
        Connects midpoints → approximate centreline.

        This is less geometrically precise than erosion but always
        produces a usable path for any thin polygon shape.
        """
        cfg = self.cfg
        w   = self._poly_min_width(polygon)
        if w < 0.1:
            return None

        try:
            coords = list(polygon.exterior.coords[:-1])  # no duplicate
        except Exception:
            return None

        n = len(coords)
        if n < 4:
            return None

        # Sample at most 60 points for performance
        step  = max(1, n // 60)
        sampled = coords[::step]

        midpoints = []
        all_coords = np.array(coords)

        for sx, sy in sampled:
            # Find the point on the boundary furthest from the current point
            # in the direction perpendicular to the boundary tangent.
            # Approximation: find the point closest to the centroid-side
            # of the strip — i.e. the point approximately w away from (sx,sy).
            dists_to_s = np.hypot(all_coords[:,0]-sx, all_coords[:,1]-sy)
            # Exclude points too close (same side of strip)
            # and too far (different feature entirely)
            target_dist = w
            dist_diff   = np.abs(dists_to_s - target_dist)
            opp_i       = int(np.argmin(dist_diff))
            ox, oy = coords[opp_i]
            midpoints.append(((sx+ox)/2.0, (sy+oy)/2.0))

        if len(midpoints) < 2:
            return None

        # Remove duplicate or near-duplicate midpoints
        filtered = [midpoints[0]]
        for pt in midpoints[1:]:
            if math.hypot(pt[0]-filtered[-1][0],
                          pt[1]-filtered[-1][1]) > 0.5:
                filtered.append(pt)

        if len(filtered) < 2:
            return None

        if cfg.thin_wall_alternating and layer_index % 2 == 1:
            filtered = filtered[::-1]

        return LineString(filtered)

    def _poly_min_width(self, polygon: Polygon) -> float:
        """
        Estimate the minimum width of a polygon using binary search
        on inward buffer. Returns the largest inset radius r such that
        polygon.buffer(-r) is still non-empty, × 2 (diameter).
        """
        lo, hi = 0.0, min(
            polygon.bounds[2] - polygon.bounds[0],
            polygon.bounds[3] - polygon.bounds[1]) / 2.0
        tol = 0.05   # mm precision
        while hi - lo > tol:
            mid = (lo + hi) / 2.0
            if polygon.buffer(-mid).is_empty:
                hi = mid
            else:
                lo = mid
        return lo * 2.0   # diameter

    def _compute_walls(self, polygon: Polygon,
                        layer_index: int = 0,
                        actual_bead_width: float = None) -> List:
        """
        Compute wall passes for a polygon.

        THREE geometry cases are handled:

        1. THIN SOLID (w_min ≈ bead_width, no interior holes):
           Centreline extraction — deposit one open path along the
           midline of the strip. The bead covers the full width.
           e.g. wall.stl, bowl_fragment.stl.
           Returns a list containing one LineString (the centreline).
           Direction alternates between layers if thin_wall_alternating.

        2. ANNULUS (polygon with interior hole, thin wall):
           Single closed ring pass tracing the outer boundary.
           e.g. bowl cross-section, pipe.

        3. SOLID (wide polygon, no hole):
           Standard inset loop: diff polygons per pass.
           e.g. solid block, thick-walled part.
        """
        cfg  = self.cfg
        if cfg.num_contours == 0:
            return []

        bw   = actual_bead_width if actual_bead_width is not None else cfg.bead_width
        step = bw * (1 - cfg.overlap_pct / 100)

        # ── Case 1: Thin solid polygon ────────────────────────────
        has_holes = (hasattr(polygon, 'interiors') and
                     len(list(polygon.interiors)) > 0)

        if not has_holes:
            w_min = self._poly_min_width(polygon)
            if w_min <= bw * 1.5:
                # Thin solid — extract centreline, return as a single pass.
                # The wall tag in LayerData stores the centreline LineString
                # directly (not a ring polygon). The LS generator already
                # handles LineString wall paths via list(path.coords).
                cl = self._thin_wall_centreline(polygon, layer_index)
                if cl is not None and not cl.is_empty:
                    return [cl]
                # Centreline extraction failed — this polygon is genuinely
                # too degenerate for any reliable path.
                # Return [] so _infill_region also returns None and this
                # polygon is silently skipped for this layer.
                # DO NOT fall through to the standard inset loop — it will
                # produce a degenerate ring wall that triggers contour infill
                # on the full thin polygon (the multiple-contours bug).
                print(f"  WARN: centreline failed for thin wall "
                      f"(w_min={w_min:.2f}mm), layer {layer_index} skipped.")
                return []

        # ── Case 2: Annulus (ring with interior hole) ─────────────
        if has_holes:
            thickness = self._poly_min_width(polygon)
            if thickness <= step * 1.1:
                # Single pass — the exterior boundary IS the torch path.
                return [polygon]
            # Thick annulus — fall through to standard inset loop below

        # ── Case 3: Standard inset wall passes ────────────────────
        walls   = []
        current = polygon

        for pass_n in range(cfg.num_contours):
            inset = current.buffer(-step)

            if inset.is_empty:
                remain_w = self._poly_min_width(current)
                if remain_w < bw * 0.3:
                    break
                # Place a final pass at reduced step, then done.
                reduced_step = remain_w / 2.0 - 0.1
                inset = current.buffer(-max(0.01, reduced_step))
                if inset.is_empty:
                    # Cannot inset at all — append current exterior as pass
                    walls.append(current)
                    break

            ring = current.difference(inset)
            if not ring.is_empty:
                walls.append(ring)
            current = inset
            if current.is_empty:
                break

        return walls

    def _infill_region(self, polygon: Polygon,
                        layer_index: int = 0,
                        precomputed_walls: List = None
                        ) -> Optional[Polygon]:
        """
        Returns the region inside all wall passes for infill.

        Accepts precomputed_walls to avoid calling _compute_walls twice
        per layer — once in the slice loop (to store walls) and once here.

        THIN WALL special case:
          When walls contains a LineString (centreline path), the
          centreline covers the full cross-section — return None so no
          infill is attempted.

        Empty walls (centreline failed, [] returned):
          Return None — polygon is skipped entirely this layer.

        Normal case:
          Subtract all ring polygon walls from the original polygon.
        """
        cfg   = self.cfg
        if cfg.num_contours == 0:
            return polygon

        # Use precomputed walls if provided (avoids double computation)
        walls = (precomputed_walls
                 if precomputed_walls is not None
                 else self._compute_walls(polygon, layer_index=layer_index))

        # Empty walls → polygon was skipped (e.g. centreline failed)
        if not walls:
            return None

        # ── Thin wall check ───────────────────────────────────────
        # LineString wall = centreline path covering full width → no infill
        from shapely.geometry import LineString as _LS
        if isinstance(walls[0], _LS):
            return None

        # ── Normal: subtract ring walls from polygon ──────────────
        from shapely.ops import unary_union
        ring_walls = [w for w in walls if not isinstance(w, _LS)]
        if not ring_walls:
            return None

        wall_union = unary_union(ring_walls)
        try:
            remaining = polygon.difference(wall_union)
        except Exception:
            remaining = None

        if remaining is None or remaining.is_empty:
            return None

        min_area = cfg.bead_width ** 2 * 0.1
        if remaining.area < min_area:
            return None

        try:
            minx, miny, maxx, maxy = remaining.bounds
            if min(maxx - minx, maxy - miny) < cfg.bead_width * 0.3:
                return None
        except Exception:
            pass

        return remaining

    # ── Path utilities ────────────────────────

    def _filter_short_paths(self, segments: List[LineString],
                             min_length: float) -> List[LineString]:
        """
        Remove segments shorter than min_length (mm).
        Prevents arc restrikes on tiny edge stubs that appear when
        raster lines clip to a narrow polygon boundary.
        """
        return [s for s in segments if s.length >= min_length]

    # ── Raster infill ─────────────────────────

    def _raster_infill(self, region, angle_deg: float,
                       layer_idx: int,
                       actual_bead_width: float = None):
        """
        Returns (infill_paths, edge_passes, mismatch_info).
        infill_paths : List[LineString]
        edge_passes  : List[(LineString, speed_mm_s)]
        mismatch_info: List[MismatchInfo]
        """
        cfg = self.cfg
        bw  = actual_bead_width if actual_bead_width is not None else cfg.bead_width
        spacing = bw * (1 - cfg.overlap_pct / 100)

        minx, miny, maxx, maxy = region.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        diag   = np.hypot(maxx - minx, maxy - miny)

        # Generate parallel lines across full bounding box
        raw_lines = []
        y = miny - diag
        while y < maxy + diag:
            raw_lines.append(
                LineString([(minx - diag, y), (maxx + diag, y)]))
            y += spacing

        # Rotate and clip to region
        multi    = MultiLineString(raw_lines)
        rotated  = shapely_rotate(multi, angle_deg, origin=(cx, cy))
        clipped  = rotated.intersection(region)
        segments = _to_linestring_list(clipped)

        if not segments:
            return [], [], []

        # ── Fill mismatch ──────────────────────────────────────────
        import math
        rad      = math.radians(angle_deg)
        corners  = [(minx, miny), (maxx, miny),
                    (maxx, maxy), (minx, maxy)]
        proj     = [(-x*math.sin(rad) + y*math.cos(rad))
                    for x, y in corners]
        region_w = max(proj) - min(proj)
        n_lines  = len(segments)
        filled_w  = (n_lines - 1) * spacing + bw
        gap       = region_w - filled_w
        threshold = cfg.mismatch_threshold_frac * bw

        edge_passes   = []
        mismatch_list = []

        # Only report a mismatch if the gap is meaningfully large —
        # at least 5% of bead width. This suppresses floating-point
        # noise when filled_w ≈ region_w (the "filled=region, gap=0" case).
        if gap > threshold * 0.1:
            if abs(gap) <= threshold:
                # Strategy 1 — adjust spacing evenly
                if n_lines > 1:
                    adj = (region_w - bw) / (n_lines - 1)
                    raw2 = []
                    y2   = miny - diag
                    for _ in range(n_lines):
                        raw2.append(LineString(
                            [(minx - diag, y2), (maxx + diag, y2)]))
                        y2 += adj
                    m2  = MultiLineString(raw2)
                    r2  = shapely_rotate(m2, angle_deg, origin=(cx, cy))
                    c2  = r2.intersection(region)
                    s2  = _to_linestring_list(c2)
                    if s2:
                        segments = s2
                strategy   = "adjusted_spacing"
                edge_speed = 0.0
            else:
                # Strategy 2 — one or more edge passes to fill the gap.
                # For large gaps (> bw), deposit multiple passes spaced
                # at actual bead width until the remainder is covered.
                edge_speed = 0.0
                remaining_gap = gap
                pass_offset   = filled_w
                ep_count      = 0
                max_ep        = max(1, int(gap / spacing) + 2)

                while remaining_gap > threshold * 0.1 and ep_count < max_ep:
                    target_w = min(remaining_gap, bw)
                    target_w = max(target_w, bw * 0.3)
                    if _BEAD_MODEL_AVAILABLE:
                        edge_speed = _BeadModel().speed_for_width_sl(target_w)
                    else:
                        edge_speed = (bw / max(target_w, 0.1)
                                      ) * cfg.weld_speed_mms \
                                     * cfg.edge_speed_multiplier
                    edge_speed = round(edge_speed, 2)

                    # Place edge pass centred in remaining gap
                    ep_y = (miny - diag) + pass_offset + target_w / 2
                    el   = LineString([(minx - diag, ep_y),
                                       (maxx + diag, ep_y)])
                    er   = shapely_rotate(el, angle_deg, origin=(cx, cy))
                    ec   = er.intersection(region)
                    for es in _to_linestring_list(ec):
                        edge_passes.append((es, edge_speed))

                    pass_offset   += target_w * (1 - cfg.overlap_pct / 100)
                    remaining_gap -= target_w * (1 - cfg.overlap_pct / 100)
                    ep_count      += 1

                strategy = "extra_edge_pass"

            mismatch_list.append(MismatchInfo(
                region_width_mm = round(region_w, 3),
                n_lines         = n_lines,
                filled_width_mm = round(filled_w, 3),
                gap_mm          = round(gap, 3),
                strategy        = strategy,
                edge_speed      = round(edge_speed, 2),
            ))

        # Reorder lines
        order    = self._compute_line_order(len(segments), layer_idx)
        segments = [segments[i] for i in order if i < len(segments)]

        # Apply direction
        directed = self._apply_direction(segments, layer_idx)

        # Filter segments shorter than minimum path length.
        # Avoids arc restrikes on tiny stubs at polygon edges.
        min_len  = self.cfg.bead_width * self.cfg.min_path_length_frac
        if min_len > 0:
            directed = self._filter_short_paths(directed, min_len)

        return directed, edge_passes, mismatch_list

    def _compute_line_order(self, n: int, layer_idx: int) -> List[int]:
        cfg = self.cfg

        if cfg.line_order == LineOrder.SEQUENTIAL:
            return list(range(n))

        elif cfg.line_order == LineOrder.ALTERNATING:
            evens = list(range(0, n, 2))
            odds  = list(range(1, n, 2))
            return (evens + odds) if layer_idx % 2 == 0 \
                   else (odds + evens)

        elif cfg.line_order == LineOrder.INTERLEAVED:
            indices = list(range(n))
            result  = []

            def bisect(lo, hi):
                if lo > hi:
                    return
                mid = (lo + hi) // 2
                result.append(indices[mid])
                bisect(lo, mid - 1)
                bisect(mid + 1, hi)

            bisect(0, n - 1)
            return result

        elif cfg.line_order == LineOrder.RANDOM:
            rng   = random.Random(layer_idx * 1337)
            order = list(range(n))
            rng.shuffle(order)
            return order

        elif cfg.line_order == LineOrder.CUSTOM:
            valid   = [i for i in cfg.custom_order if i < n]
            missing = [i for i in range(n) if i not in valid]
            return valid + missing

        return list(range(n))

    def _apply_direction(self, segments: List[LineString],
                         layer_idx: int) -> List[LineString]:
        cfg    = self.cfg
        result = []
        rng    = random.Random(layer_idx * 2053)

        for line_idx, line in enumerate(segments):
            coords = list(line.coords)
            if cfg.direction_mode == DirectionMode.BOUSTROPHEDON:
                if line_idx % 2 == 1:
                    coords = coords[::-1]
            elif cfg.direction_mode == DirectionMode.RANDOM:
                if rng.random() > 0.5:
                    coords = coords[::-1]
            result.append(LineString(coords))

        return result

    # ── Contour infill ────────────────────────

    def _contour_region_paths(self, region,
                               layer_index: int = 0) -> List[LineString]:
        """
        Extract all valid exterior paths from a region (Polygon or
        MultiPolygon), applying seam rotation so the start/end point
        is distributed around the circumference rather than stacked.
        """
        cfg     = self.cfg
        min_len = cfg.bead_width * 0.3
        seam_deg = self._seam_angle(layer_index)
        paths   = []

        def _extract(p: Polygon):
            if p.exterior.length < min_len:
                return
            # Reorder ring so start point is at the seam angle
            try:
                coords = self._reorder_ring_to_seam(p, seam_deg)
            except Exception:
                coords = list(p.exterior.coords)
            paths.append(LineString(coords))

        if isinstance(region, Polygon):
            _extract(region)
        elif isinstance(region, MultiPolygon):
            for p in region.geoms:
                _extract(p)
        return paths

    def _contour_infill(self, region, actual_bead_width: float = None,
                        layer_index: int = 0):
        """
        Generate concentric contour paths for the infill region.
        Ring order: inside-to-outside (avoids height accumulation at centre).
        Spacing derived from actual deposited bead width at the layer's
        travel speed, not the nominal cfg.bead_width.
        """
        cfg     = self.cfg
        bw      = actual_bead_width if actual_bead_width is not None else cfg.bead_width
        spacing = bw * (1 - cfg.overlap_pct / 100)

        # ── Pre-check: thin region → single pass, no gap ─────────
        try:
            minx, miny, maxx, maxy = region.bounds
            approx_width = min(maxx - minx, maxy - miny)
        except Exception:
            approx_width = cfg.bead_width * 10

        if approx_width <= spacing * 2:
            # Too thin for more than one pass — just trace boundary
            paths = self._contour_region_paths(region, layer_index=layer_index)
            if not cfg.contour_inset_start:
                paths = paths[::-1]
            return paths, [], []

        # ── Count rings at nominal spacing and record remainder ─────
        # CRITICAL: pass_count is incremented AFTER confirming the ring
        # is valid — i.e. after placing the ring exterior path and
        # successfully computing the next inset. This ensures N reflects
        # rings actually placed, not loop iterations.
        #
        # The loop structure:
        #   1. Trace current exterior  → this IS the Nth ring
        #   2. Buffer inward           → check if valid
        #   3. If valid: N += 1, advance; else: stop, current = remainder
        current     = region
        max_passes  = int(approx_width / spacing) + 2
        n_placed    = 0         # rings actually placed (traced)
        remainder   = region    # initialise to full region

        for _ in range(max_passes):
            # Attempt one more inset
            inset = current.buffer(-spacing)

            # Is the inset valid for a further ring?
            is_valid = False
            if not inset.is_empty and not inset.equals(current):
                try:
                    area_ratio = inset.area / max(current.area, 1e-9)
                    if area_ratio >= 0.01:
                        minx2,miny2,maxx2,maxy2 = inset.bounds
                        if min(maxx2-minx2, maxy2-miny2) >= spacing * 0.5:
                            is_valid = True
                except Exception:
                    pass

            if is_valid:
                # Ring at current level will be placed → count it
                n_placed += 1
                remainder = inset   # remainder shrinks with each ring
                current   = inset
            else:
                # No more rings fit — remainder is current state
                remainder = current
                break

        N = n_placed

        # ── Early exit conditions ─────────────────────────────────
        # No rings placed at all (region too thin for even one ring
        # — the pre-check should have caught this, but belt-and-braces):
        if N == 0:
            paths = self._contour_region_paths(region, layer_index=layer_index)
            if not cfg.contour_inset_start:
                paths = paths[::-1]
            return paths, [], []

        # The 'remainder' after N rings is the centre void.
        # Measure its minimum width — this is the actual centre gap.
        # Using _poly_min_width (binary search) rather than bbox so that
        # non-circular remainders (elliptic, rectangular) are measured
        # accurately along their narrow axis.
        if remainder is None or remainder.is_empty:
            return self._build_contour_paths(
                region, spacing, N, cfg,
                layer_index=layer_index), [], []

        gap_w = self._poly_min_width(remainder)

        threshold = cfg.mismatch_threshold_frac * bw

        if gap_w < threshold * 0.1 or gap_w >= approx_width:
            return self._build_contour_paths(
                region, spacing, N, cfg,
                layer_index=layer_index), [], []

        threshold   = cfg.mismatch_threshold_frac * bw
        edge_passes = []
        mismatch    = []

        if gap_w <= threshold:
            use_s1 = False
            d_adj  = spacing
            if N > 1:
                total_depth = approx_width / 2.0 - bw / 2.0
                d_adj_candidate = total_depth / (N - 1)
                if d_adj_candidate <= bw:
                    d_adj  = max(spacing * 0.5, d_adj_candidate)
                    use_s1 = True
            elif N == 1:
                use_s1 = True

            if use_s1:
                paths = self._build_contour_paths(region, d_adj, N, cfg, layer_index=layer_index)
                strategy   = "adjusted_spacing"
                edge_speed = 0.0
            else:
                gap_w = approx_width - N * spacing * 2
                gap_w = max(gap_w, bw * 0.1)
                paths = self._build_contour_paths(region, spacing, N, cfg, layer_index=layer_index)
                b_e   = min(bw, gap_w + bw * cfg.overlap_pct / 100)
                if _BEAD_MODEL_AVAILABLE:
                    edge_speed = _BeadModel().speed_for_width_sl(b_e)
                else:
                    edge_speed = (bw / max(b_e, 0.1)
                                  ) * cfg.weld_speed_mms * cfg.edge_speed_multiplier
                fill_paths = self._contour_region_paths(remainder, layer_index=layer_index)
                for fp in fill_paths:
                    edge_passes.append((fp, round(edge_speed, 2)))
                strategy = "extra_edge_pass"

        else:
            # Strategy 2: centre fill — loop inward until remainder is filled
            paths = self._build_contour_paths(region, spacing, N, cfg, layer_index=layer_index)
            current_rem = remainder
            ep_count    = 0
            max_ep      = max(1, int(gap_w / spacing) + 2)

            while (current_rem is not None
                   and not current_rem.is_empty
                   and ep_count < max_ep):
                rem_w = self._poly_min_width(current_rem)
                if rem_w < bw * 0.05:
                    break
                b_e = min(bw, rem_w + bw * (cfg.overlap_pct / 100))
                if _BEAD_MODEL_AVAILABLE:
                    edge_speed = _BeadModel().speed_for_width_sl(b_e)
                else:
                    edge_speed = (bw / max(b_e, 0.1)
                                  ) * cfg.weld_speed_mms * cfg.edge_speed_multiplier
                fill_paths = self._contour_region_paths(current_rem, layer_index=layer_index)
                for fp in fill_paths:
                    edge_passes.append((fp, round(edge_speed, 2)))
                # Inset remainder for next iteration
                next_rem = current_rem.buffer(-spacing)
                if next_rem.is_empty or next_rem.equals(current_rem):
                    break
                current_rem = next_rem
                ep_count   += 1

            strategy = "extra_edge_pass"

        # filled_width_mm = diameter covered by N contour rings.
        # Each ring covers its own bead_width; rings are centred at
        # depth spacing, 2*spacing, ... from the outer edge on each side.
        # Total diameter filled = 2*N*spacing + bead_width
        # (N steps of spacing from each side + one bead_width at centre)
        filled_w = min(approx_width, 2.0 * N * spacing + bw)

        mismatch.append(MismatchInfo(
            region_width_mm = round(approx_width, 3),
            n_lines         = N,
            filled_width_mm = round(filled_w, 3),
            gap_mm          = round(gap_w, 3),
            strategy        = strategy,
            edge_speed      = round(edge_speed, 2)
                              if strategy == "extra_edge_pass" else 0.0,
        ))

        if not cfg.contour_inset_start:
            paths = paths[::-1]

        return paths, edge_passes, mismatch

    def _build_contour_paths(self, region, spacing: float,
                              n_rings: int,
                              cfg,
                              layer_index: int = 0) -> List[LineString]:
        """
        Generate exactly n_rings contour paths at the given spacing.
        Each ring's start point is rotated by the layer seam angle
        plus a small per-ring offset so adjacent rings within the
        same layer don't stack their seam at the same location.
        """
        paths   = []
        current = region

        for ring_i in range(n_rings):
            # Per-ring seam: layer base angle + small irrational offset
            # 17.3 deg is near-golden — distributes across ~21 rings
            ring_seam = (self._seam_angle(layer_index)
                         + ring_i * 17.3) % 360
            ring_paths = self._contour_region_paths(
                current, layer_index=layer_index)

            # Override with per-ring seam angle
            adjusted = []
            for path in ring_paths:
                try:
                    poly_approx = Polygon(path.coords).buffer(0)
                    if poly_approx.is_valid and not poly_approx.is_empty:
                        coords = self._reorder_ring_to_seam(
                            poly_approx, ring_seam)
                        adjusted.append(LineString(coords))
                    else:
                        adjusted.append(path)
                except Exception:
                    adjusted.append(path)
            paths.extend(adjusted)

            if not ring_paths:
                break

            inset = current.buffer(-spacing)
            if inset.is_empty:
                break
            try:
                if inset.area / max(current.area, 1e-9) < 0.01:
                    break
                minx2, miny2, maxx2, maxy2 = inset.bounds
                if min(maxx2-minx2, maxy2-miny2) < spacing * 0.5:
                    break
            except Exception:
                break
            if inset.equals(current):
                break
            current = inset

        return paths

    # ─────────────────────────────────────────
    #  Visualisation
    # ─────────────────────────────────────────

    def visualize_layer(self, layer_index: int, ax=None,
                        show_as_beads: bool = True):
        if layer_index >= len(self.layers):
            raise IndexError(f"Layer {layer_index} out of range.")

        layer      = self.layers[layer_index]
        cfg        = self.cfg
        standalone = ax is None

        if standalone:
            fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0e0e0e")

        ax.set_facecolor("#0e0e0e")
        ax.set_aspect("equal")
        mode = "bead view" if show_as_beads else "line view"

        # Extract WPR for this layer
        wpr       = getattr(layer, 'wpr', None)
        tilt_on   = (wpr is not None and
                     cfg.torch_tilt_enabled and
                     abs(wpr[1]) > 0.5)   # P > 0.5deg = meaningful tilt

        wpr_str = ""
        if wpr is not None and cfg.torch_tilt_enabled:
            wpr_str = f"  |  W={wpr[0]:.1f}° P={wpr[1]:.1f}° R={wpr[2]:.1f}°"

        ax.set_title(
            f"Layer {layer.index + 1}  |  z={layer.z:.2f}mm  |  {mode}"
            f"{wpr_str}",
            color="#f9e2af" if tilt_on else "white",
            fontsize=8, pad=8)

        bead_r = cfg.bead_width / 2.0

        # ── Colour palette ─────────────────────────────────────────
        # Wall passes:  distinct cyan shades per pass number
        #   Pass 1 (outermost) = bright cyan  #00e5ff
        #   Pass 2             = mid cyan      #0095a8
        #   Pass 3+            = dark cyan     #005f6b
        # Infill lines: warm yellow-orange colourmap (deposition order)
        # Edge passes:  solid red-pink
        WALL_COLOURS = ["#00e5ff", "#0095a8", "#005f6b",
                        "#003d47", "#002530"]

        # ── Walls ──────────────────────────────────────────────────
        if cfg.show_contours:
            from shapely.geometry import LineString as _LS
            for wall_i, ring in enumerate(layer.walls):
                wall_col = WALL_COLOURS[min(wall_i, len(WALL_COLOURS)-1)]

                # ── Thin-wall centreline (LineString) ───────────────
                if isinstance(ring, _LS):
                    try:
                        coords = list(ring.coords)
                        xs = [c[0] for c in coords]
                        ys = [c[1] for c in coords]
                        if show_as_beads:
                            bead = ring.buffer(bead_r,
                                               cap_style=2, join_style=2)
                            if not bead.is_empty:
                                bx, by = bead.exterior.xy
                                ax.fill(bx, by, alpha=0.65,
                                        color=wall_col, linewidth=0)
                        ax.plot(xs, ys, color=wall_col, linewidth=1.8)
                        # Direction arrow
                        mid = len(coords) // 2
                        if mid > 0:
                            ax.annotate("",
                                xy=coords[mid],
                                xytext=coords[mid-1],
                                arrowprops=dict(arrowstyle="-|>",
                                    color="#ffffff", lw=1.2,
                                    mutation_scale=9),
                                annotation_clip=True)
                        # Start/end markers
                        ax.plot(xs[0],  ys[0],  "o",
                                color="#a6e3a1", markersize=5)
                        ax.plot(xs[-1], ys[-1], "s",
                                color="#f38ba8", markersize=5)
                    except Exception:
                        pass
                    continue

                # ── Normal ring polygon wall ────────────────────────
                if ring.is_empty:
                    continue
                geoms = ring.geoms if hasattr(ring, "geoms") else [ring]
                for g in geoms:
                    if not hasattr(g, "exterior"):
                        continue
                    try:
                        x, y = g.exterior.xy
                        if show_as_beads:
                            ax.fill(x, y, alpha=0.55,
                                    color=wall_col, linewidth=0)
                            ax.plot(x, y, color="#ffffff",
                                    linewidth=0.6, alpha=0.5,
                                    linestyle="--")
                        else:
                            ax.fill(x, y, alpha=0.18, color=wall_col)
                            ax.plot(x, y, color=wall_col, linewidth=1.4)

                        coords = list(zip(x, y))
                        mid = len(coords) // 2
                        if mid > 0:
                            ax.annotate("",
                                xy=coords[mid],
                                xytext=coords[mid-1],
                                arrowprops=dict(arrowstyle="-|>",
                                    color="#ffffff", lw=0.9,
                                    mutation_scale=7),
                                annotation_clip=True)
                        try:
                            from shapely.geometry import Polygon as SPoly
                            ring_poly = SPoly(list(zip(x, y)))
                            cx  = ring_poly.centroid.x
                            cy_ = ring_poly.centroid.y
                            ax.text(cx, cy_, f"W{wall_i+1}",
                                    color="#ffffff", fontsize=5,
                                    ha="center", va="center", alpha=0.7)
                        except Exception:
                            pass
                    except Exception:
                        pass

        # ── Infill ─────────────────────────────────────────────────
        if cfg.show_infill:
            segs = [list(p.coords) for p in layer.infill_paths
                    if len(list(p.coords)) >= 2]
            if segs:
                # Warm colourmap — visually distinct from cyan walls
                cmap   = plt.get_cmap("YlOrRd")
                colors = [cmap(0.25 + 0.7 * k / max(len(segs), 1))
                          for k in range(len(segs))]
                for line_i, (seg, color) in enumerate(
                        zip(segs, colors)):
                    if show_as_beads:
                        try:
                            bead = LineString(seg).buffer(
                                bead_r, cap_style=2, join_style=2)
                            if not bead.is_empty:
                                bx, by = bead.exterior.xy
                                ax.fill(bx, by, alpha=0.70,
                                        color=color, linewidth=0)
                            # Thin centreline
                            xs, ys = zip(*seg)
                            ax.plot(xs, ys, color="#111",
                                    linewidth=0.3, alpha=0.4)
                        except Exception:
                            xs, ys = zip(*seg)
                            ax.plot(xs, ys, color=color,
                                    linewidth=1.0)
                    else:
                        xs, ys = zip(*seg)
                        ax.plot(xs, ys, color=color,
                                linewidth=1.2, alpha=0.9)

                    # Direction arrow at midpoint
                    try:
                        mid = len(seg) // 2
                        if mid > 0:
                            ax.annotate("",
                                xy=seg[mid],
                                xytext=seg[mid - 1],
                                arrowprops=dict(
                                    arrowstyle="-|>",
                                    color="#ffffff", lw=1.0,
                                    mutation_scale=9),
                                annotation_clip=True)
                    except Exception:
                        pass

        # ── Edge passes (red-pink) ──────────────────────────────────
        for ep, ep_spd in layer.edge_passes:
            try:
                if show_as_beads:
                    bead = ep.buffer(bead_r * 0.5,
                                     cap_style=2, join_style=2)
                    if not bead.is_empty:
                        bx, by = bead.exterior.xy
                        ax.fill(bx, by, alpha=0.8,
                                color="#ff4d6d", linewidth=0)
                else:
                    xs, ys = zip(*list(ep.coords))
                    ax.plot(xs, ys, color="#ff4d6d",
                            linewidth=1.8, alpha=0.9)
            except Exception:
                pass

        # ── Torch tilt indicator (XY plan view) ───────────────────
        # Shows the tilt direction (W) and magnitude (P) as an arrow.
        # Arrow length scales with P: max at P=90deg = 2x bead_width.
        # Placed at the plot centre so it is always visible.
        if (wpr is not None and cfg.torch_tilt_enabled and tilt_on):
            W_rad     = math.radians(wpr[0])
            P_deg     = wpr[1]
            arrow_len = cfg.bead_width * 2.0 * (P_deg / 90.0)
            dx = arrow_len * math.cos(W_rad)
            dy = arrow_len * math.sin(W_rad)
            if self.global_bounds is not None:
                cx = (self.global_bounds[0] + self.global_bounds[1]) / 2
                cy = (self.global_bounds[2] + self.global_bounds[3]) / 2
            else:
                try:
                    minx, miny, maxx, maxy = layer.walls[0].bounds                         if layer.walls else (0,0,10,10)
                    cx, cy = (minx+maxx)/2, (miny+maxy)/2
                except Exception:
                    cx, cy = 0.0, 0.0
            ax.annotate("",
                xy=(cx + dx, cy + dy),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>",
                    color="#f9e2af", lw=2.0,
                    mutation_scale=14),
                annotation_clip=False)
            ax.text(cx + dx * 1.2, cy + dy * 1.2,
                    f"P={P_deg:.1f}°",
                    color="#f9e2af", fontsize=7,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="#1e1e2e", ec="#f9e2af",
                              alpha=0.85))

        # ── Fixed scale ────────────────────────────────────────────
        if self.global_bounds is not None:
            xmin, xmax, ymin, ymax = self.global_bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        # ── Legend ─────────────────────────────────────────────────
        legend_items = []
        if cfg.show_contours and layer.walls:
            n_walls = len(layer.walls)
            for wi in range(min(n_walls, len(WALL_COLOURS))):
                legend_items.append(mpatches.Patch(
                    color=WALL_COLOURS[wi],
                    label=f"Wall pass {wi+1}"))
        if cfg.show_infill and layer.infill_paths:
            legend_items.append(mpatches.Patch(
                color="#e07b39",
                label=f"Infill ({cfg.line_order.value} / "
                      f"{cfg.direction_mode.value})"))
        if layer.edge_passes:
            legend_items.append(mpatches.Patch(
                color="#ff4d6d", label="Edge compensation"))
        if tilt_on:
            from matplotlib.lines import Line2D as _L2D
            legend_items.append(_L2D([0],[0],
                color="#f9e2af", lw=2,
                marker=">", markersize=5,
                label=f"Torch tilt W={wpr[0]:.1f}° P={wpr[1]:.1f}°"))
        if legend_items:
            ax.legend(handles=legend_items, facecolor="#111",
                      labelcolor="white", fontsize=6,
                      loc="upper right",
                      framealpha=0.8)

        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        ax.set_xlabel("X (mm)", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Y (mm)", color="#aaaaaa", fontsize=8)
        ax.grid(True, color="#222", linewidth=0.4, linestyle="--")

        if standalone:
            plt.tight_layout()
            plt.show()

    def visualize_multiple_layers(self, layer_indices: List[int] = None,
                                   cols: int = 3,
                                   show_as_beads: bool = True,
                                   show_all: bool = True,
                                   page_size: int = 12):
        """
        Paged layer viewer. Renders page_size layers per page.
        ← → keys or buttons to navigate.
        Each page fully replaces the previous figure.
        """
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        if layer_indices is None:
            layer_indices = list(range(len(self.layers)))

        total   = len(layer_indices)
        n_pages = max(1, math.ceil(total / page_size))
        mode    = "bead view" if show_as_beads else "line view"

        # ── Window ────────────────────────────────────────────────
        win = tk.Toplevel()
        win.title("WAAM Slicer — Layer Viewer")
        win.configure(bg="#0a0a0a")
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{min(1200,int(sw*0.9))}x{min(880,int(sh*0.9))}")

        # ── Nav bar (packed at top before canvas) ─────────────────
        nav = tk.Frame(win, bg="#1e1e2e", pady=6)
        nav.pack(fill="x", side="top")

        page_label = tk.Label(nav, text="", font=("Courier", 9),
                              fg="#cdd6f4", bg="#1e1e2e")
        page_label.pack(side="left", padx=10)

        info_label = tk.Label(nav, text="", font=("Courier", 9),
                              fg="#6c7086", bg="#1e1e2e")
        info_label.pack(side="left", padx=10)

        # ── Toolbar area (packed at bottom) ───────────────────────
        tb_frame = tk.Frame(win, bg="#1e1e2e")
        tb_frame.pack(fill="x", side="bottom")

        # ── Canvas container (fills remaining space) ──────────────
        # We use a fixed container and replace only the mpl widget inside.
        container = tk.Frame(win, bg="#0a0a0a")
        container.pack(fill="both", expand=True)

        state = {"page": 0, "canvas": None, "toolbar": None, "fig": None}

        def render_page(page: int):
            page = max(0, min(n_pages - 1, page))
            state["page"] = page

            start = page * page_size
            end   = min(start + page_size, total)
            idxs  = layer_indices[start:end]
            n_plt = len(idxs)
            rows  = math.ceil(n_plt / cols)

            page_label.config(
                text=f"Page {page+1}/{n_pages}  |  "
                     f"layers {start+1}–{end} of {total}")
            info_label.config(
                text=f"{mode}  |  ←  → to navigate")

            # ── Destroy old canvas and figure completely ───────────
            if state["canvas"] is not None:
                state["canvas"].get_tk_widget().destroy()
                state["canvas"] = None
            if state["toolbar"] is not None:
                # toolbar is inside tb_frame — destroy all children
                for child in tb_frame.winfo_children():
                    child.destroy()
                state["toolbar"] = None
            if state["fig"] is not None:
                plt.close(state["fig"])
                state["fig"] = None

            # ── Build new figure ───────────────────────────────────
            cell  = 4
            fig_w = cols * cell
            fig_h = rows * cell

            fig, axes = plt.subplots(rows, cols,
                                     figsize=(fig_w, fig_h),
                                     facecolor="#0a0a0a")
            if rows * cols == 1:
                axes = np.array([[axes]])
            axes_flat = np.array(axes).flatten()

            for ax_i, layer_idx in enumerate(idxs):
                self.visualize_layer(layer_idx,
                                     ax=axes_flat[ax_i],
                                     show_as_beads=show_as_beads)
            for ax in axes_flat[n_plt:]:
                ax.set_visible(False)

            fig.suptitle(f"WAAM — {mode}  |  page {page+1}/{n_pages}",
                         color="white", fontsize=11)
            fig.tight_layout()
            state["fig"] = fig

            # ── Embed in container ─────────────────────────────────
            mpl_canvas = FigureCanvasTkAgg(fig, master=container)
            mpl_canvas.draw()
            widget = mpl_canvas.get_tk_widget()
            widget.pack(fill="both", expand=True)
            state["canvas"] = mpl_canvas

            # Toolbar in dedicated frame
            tb = NavigationToolbar2Tk(mpl_canvas, tb_frame)
            tb.update()
            state["toolbar"] = tb

        def prev_page():
            render_page(state["page"] - 1)

        def next_page():
            render_page(state["page"] + 1)

        tk.Button(nav, text="◀  Prev", command=prev_page,
                  bg="#45475a", fg="#cdd6f4", font=("Courier", 9),
                  relief="flat", padx=10).pack(side="right", padx=4)
        tk.Button(nav, text="Next  ▶", command=next_page,
                  bg="#45475a", fg="#cdd6f4", font=("Courier", 9),
                  relief="flat", padx=10).pack(side="right", padx=4)

        win.bind("<Left>",  lambda e: prev_page())
        win.bind("<Right>", lambda e: next_page())

        def _on_close():
            if state["fig"] is not None:
                plt.close(state["fig"])
            plt.close("all")
            try:
                win.destroy()
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", _on_close)

        # Do NOT call win.mainloop() — this viewer is a Toplevel window
        # sharing the parent Tk event loop. mainloop() in a non-main thread
        # makes the window unresponsive and uncloseable.
        # render_page() draws the figure; the parent event loop handles events.
        render_page(0)

    def visualize_3d_stack(self, max_layers: int = 60,
                            show_as_beads: bool = True):
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg, NavigationToolbar2Tk)

        fig = plt.figure(figsize=(10, 8), facecolor="#0a0a0a")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0a0a0a")
        fig.patch.set_facecolor("#0a0a0a")

        cmap   = plt.get_cmap("plasma")
        step   = max(1, len(self.layers) // max_layers)
        subset = self.layers[::step]

        for layer in subset:
            z    = layer.z
            frac = layer.index / max(len(self.layers) - 1, 1)
            col  = cmap(frac)

            for wall in layer.walls:
                geoms = wall.geoms if hasattr(wall, "geoms") else [wall]
                for g in geoms:
                    if hasattr(g, "exterior"):
                        x, y = g.exterior.xy
                        ax.plot(list(x), list(y), [z]*len(x),
                                color="#00cfff", linewidth=0.8, alpha=0.6)

            for path in layer.infill_paths:
                coords = list(path.coords)
                if len(coords) >= 2:
                    xs, ys = zip(*coords)
                    ax.plot(list(xs), list(ys), [z]*len(xs),
                            color=col, linewidth=0.6, alpha=0.8)

            for ep, _ in layer.edge_passes:
                try:
                    coords = list(ep.coords)
                    if len(coords) >= 2:
                        xs, ys = zip(*coords)
                        ax.plot(list(xs), list(ys), [z]*len(xs),
                                color="#f38ba8", linewidth=1.2, alpha=0.9)
                except Exception:
                    pass

        ax.set_xlabel("X (mm)", color="#aaaaaa", labelpad=8)
        ax.set_ylabel("Y (mm)", color="#aaaaaa", labelpad=8)
        ax.set_zlabel("Z (mm)", color="#aaaaaa", labelpad=8)
        ax.set_title("WAAM Build — 3D Layer Stack",
                     color="white", fontsize=12, pad=15)
        ax.tick_params(colors="#888888", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#222")
        ax.yaxis.pane.set_edgecolor("#222")
        ax.zaxis.pane.set_edgecolor("#222")
        ax.grid(True, color="#1a1a1a", linewidth=0.5)

        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0], color="#00cfff", lw=1.5, label="Walls"),
            Line2D([0],[0], color="#cba6f7", lw=1.5, label="Infill"),
            Line2D([0],[0], color="#f38ba8", lw=1.5, label="Edge pass"),
        ], facecolor="#1e1e2e", labelcolor="white",
           fontsize=8, loc="upper left")

        win = tk.Toplevel()
        win.title("WAAM Slicer — 3D Build View")
        win.configure(bg="#0a0a0a")

        mpl_canvas = FigureCanvasTkAgg(fig, master=win)
        mpl_canvas.draw()
        mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

        tb = tk.Frame(win, bg="#1e1e2e")
        tb.pack(fill="x", side="bottom")
        NavigationToolbar2Tk(mpl_canvas, tb).update()

        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{min(1100,int(sw*0.85))}x{min(800,int(sh*0.85))}")
        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close("all"), win.destroy()))

    def visualize_seam_progression(self):
        if not self.layers:
            print("No layers sliced yet.")
            return
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0e0e0e")
        ax.set_facecolor("#0e0e0e")
        ax.set_aspect("equal")
        ax.set_title("Seam Progression Across Layers",
                     color="white", fontsize=11)
        cmap   = plt.get_cmap("hsv")
        angles = [self._seam_angle(i) for i in range(len(self.layers))]
        for i, angle in enumerate(angles):
            rad   = np.radians(angle)
            color = cmap(i / max(len(angles), 1))
            ax.plot(np.cos(rad), np.sin(rad), "o",
                    color=color, markersize=5, alpha=0.8)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#444", linewidth=1)
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.tick_params(colors="gray")
        plt.tight_layout()
        plt.show()

    def print_summary(self, export_csv: bool = False,
                      csv_path: str = "waam_slice_report.csv"):
        """
        Print detailed slice summary: layer schedule, gap compensation,
        overhang criticality, per-layer table. Optional CSV export.
        """
        cfg  = self.cfg
        bw   = cfg.bead_width
        step = bw * (1 - cfg.overlap_pct / 100)
        W    = 56

        # ── Per-layer records ──────────────────────────────────────
        records = []
        for i, layer in enumerate(self.layers):
            thickness = (round(self.layers[i+1].z - layer.z, 4)
                         if i + 1 < len(self.layers) else None)
            n_s1    = sum(1 for m in layer.mismatch_info
                          if m.strategy == "adjusted_spacing")
            n_s2    = sum(1 for m in layer.mismatch_info
                          if m.strategy == "extra_edge_pass")
            gaps    = [m.gap_mm for m in layer.mismatch_info]
            max_gap = round(max(gaps), 3) if gaps else 0.0
            ep_spds = sorted(set(round(s,1) for _,s in layer.edge_passes))
            ep_str  = ", ".join(str(s) for s in ep_spds) if ep_spds else "—"
            oh_deg  = self._nz_overhang_at(layer.z)
            records.append({
                "layer": i+1, "z_mm": round(layer.z, 3),
                "thickness_mm": thickness,
                "weld_speed_mms": layer.weld_speed_mms,
                "infill_paths": len(layer.infill_paths),
                "edge_passes": len(layer.edge_passes),
                "n_strategy1": n_s1, "n_strategy2": n_s2,
                "max_gap_mm": max_gap, "ep_speeds_mms": ep_str,
                "oh_deg": oh_deg,
            })

        thicknesses = [r["thickness_mm"] for r in records if r["thickness_mm"]]
        speeds      = [r["weld_speed_mms"] for r in records if r["weld_speed_mms"]]
        real_gaps   = [r["max_gap_mm"] for r in records if r["max_gap_mm"] > 0]
        n_s1_total  = sum(r["n_strategy1"] for r in records)
        n_s2_total  = sum(r["n_strategy2"] for r in records)
        n_mismatch  = sum(1 for r in records
                          if r["n_strategy1"]+r["n_strategy2"] > 0)
        total_paths = sum(r["infill_paths"] for r in records)
        total_ep    = sum(r["edge_passes"]  for r in records)

        # ── Header ────────────────────────────────────────────────
        print("\n" + "="*W)
        print("  WAAM Slicer Summary")
        print("="*W)
        print(f"  Total layers        : {len(self.layers)}")
        if cfg.adaptive_layers:
            t_lo = min(thicknesses) if thicknesses else 0
            t_hi = max(thicknesses) if thicknesses else 0
            print(f"  Layer height        : adaptive "
                  f"({t_lo:.3f}–{t_hi:.3f} mm)")
            if speeds:
                print(f"  Weld speed range    : {min(speeds):.1f}–"
                      f"{max(speeds):.1f} mm/sec (bead model)")
        else:
            print(f"  Layer height        : {cfg.layer_height} mm (constant)")
        print(f"  Bead width (nominal): {bw:.1f} mm")
        print(f"  Overlap             : {cfg.overlap_pct:.1f}% "
              f"(step = {step:.2f} mm)")
        print(f"  Infill type         : {cfg.infill_type.value}")
        print(f"  Contour order       : "
              f"{'inside→out' if not cfg.contour_inset_start else 'outside→in'}")
        print(f"  Num contours        : {cfg.num_contours}")
        print(f"  Total infill paths  : {total_paths}")
        print(f"  Total edge passes   : {total_ep}")

        # ── Diverging geometry warning ─────────────────────────────
        # If cross-section area increases monotonically from layer 1,
        # the geometry is likely upside-down (bowl, cup, funnel etc.).
        # Each layer deposits on thin air outside the previous ring.
        if len(self.layers) >= 4:
            try:
                areas = []
                for l in self.layers[:6]:
                    cs = self.mesh.section(
                        plane_origin=[0,0,l.z], plane_normal=[0,0,1])
                    if cs:
                        p2, _ = cs.to_2D()
                        areas.append(p2.area)
                if len(areas) >= 3:
                    increasing = sum(1 for i in range(1,len(areas))
                                     if areas[i] > areas[i-1]*1.05)
                    if increasing >= len(areas)-1:
                        print(f"\n  ⚠  WARNING: Cross-section area increases in "
                              f"each of the first {len(areas)} layers.")
                        print(f"     This geometry may be upside-down.")
                        print(f"     Diverging walls deposit on air — each ring")
                        print(f"     is wider than the previous, causing collapse.")
                        print(f"     → Flip the STL 180° around X or Y axis before slicing.")
            except Exception:
                pass
        import math as _math

        WIRE_DIA   = cfg.wire_diameter_mm
        DENSITY    = cfg.material_density * 1e-3   # g/cc → g/mm³
        WFR        = cfg.wire_feed_rate_m_min * 1000  # m/min → mm/min
        EFF        = max(0.01, min(1.0, cfg.deposition_efficiency))
        travel_spd = cfg.travel_speed_mms
        lift_clear = cfg.lift_clearance_mm
        ls_dwell   = cfg.dwell_time_sec

        wire_area  = _math.pi / 4 * WIRE_DIA**2
        bw         = cfg.bead_width

        total_weld_len   = 0.0
        total_dep_vol    = 0.0
        total_weld_time  = 0.0
        total_dwell_time = 0.0
        total_travel_len = 0.0

        def _last_pt(layer):
            for src in [layer.edge_passes, layer.infill_paths, layer.walls]:
                for item in reversed(src):
                    try:
                        obj = item[0] if isinstance(item, tuple) else item
                        coords = (list(obj.coords) if hasattr(obj, 'coords')
                                  else list(obj.exterior.coords))
                        if coords:
                            return coords[-1]
                    except Exception:
                        pass
            return None

        def _first_pt(layer):
            for src in [layer.walls, layer.infill_paths, layer.edge_passes]:
                for item in src:
                    try:
                        obj = item[0] if isinstance(item, tuple) else item
                        coords = (list(obj.coords) if hasattr(obj, 'coords')
                                  else list(obj.exterior.coords))
                        if coords:
                            return coords[0]
                    except Exception:
                        pass
            return None

        for i, layer in enumerate(self.layers):
            if i + 1 < len(self.layers):
                bh = self.layers[i+1].z - layer.z
            else:
                bh = cfg.bead_width * 0.4
            bh = max(bh, 0.5)

            bead_area = _math.pi / 4 * bw * bh
            spd       = layer.weld_speed_mms or cfg.weld_speed_mms

            for wall in layer.walls:
                try:
                    seg_len = (wall.length if hasattr(wall, 'length')
                               else wall.exterior.length)
                    total_weld_len  += seg_len
                    total_dep_vol   += bead_area * seg_len
                    total_weld_time += seg_len / max(spd, 0.1)
                except Exception:
                    pass

            for path in layer.infill_paths:
                try:
                    seg_len = path.length
                    total_weld_len  += seg_len
                    total_dep_vol   += bead_area * seg_len
                    total_weld_time += seg_len / max(spd, 0.1)
                except Exception:
                    pass

            for ep, ep_spd in layer.edge_passes:
                try:
                    seg_len = ep.length
                    total_weld_len  += seg_len
                    total_dep_vol   += bead_area * seg_len * 0.5
                    total_weld_time += seg_len / max(ep_spd, 0.1)
                except Exception:
                    pass

            total_dwell_time += ls_dwell

            # ── Inter-layer travel: lift + horizontal XY + plunge ─
            if i + 1 < len(self.layers):
                next_layer = self.layers[i + 1]
                pt_end   = _last_pt(layer)
                pt_start = _first_pt(next_layer)
                if pt_end and pt_start:
                    horiz = _math.hypot(pt_start[0] - pt_end[0],
                                        pt_start[1] - pt_end[1])
                    total_travel_len += lift_clear + horiz + lift_clear
                else:
                    total_travel_len += lift_clear * 2

        # Apply deposition efficiency to wire consumed
        # (more wire fed than actually deposited due to spatter)
        dep_vol_eff      = total_dep_vol          # actual deposited
        wire_vol_fed     = total_dep_vol / EFF    # wire consumed incl. spatter
        wire_consumed_mm = wire_vol_fed / wire_area
        wire_consumed_m  = wire_consumed_mm / 1000.0
        mass_g           = dep_vol_eff * DENSITY
        arc_time_min     = total_weld_time / 60.0
        dwell_time_min   = total_dwell_time / 60.0
        travel_time_sec  = total_travel_len / max(travel_spd, 1.0)
        travel_time_min  = travel_time_sec / 60.0
        total_time_min   = arc_time_min + dwell_time_min + travel_time_min

        print(f"\n  ── Wire & Time Estimate ──")
        print(f"  Wire: ⌀{WIRE_DIA}mm  "
              f"density={cfg.material_density:.2f}g/cc  "
              f"WFR={cfg.wire_feed_rate_m_min:.1f}m/min  "
              f"efficiency={EFF*100:.0f}%")
        print(f"  Travel: {travel_spd:.0f}mm/sec  "
              f"lift={lift_clear:.0f}mm  "
              f"dwell={ls_dwell:.0f}s/layer")
        print(f"  Total weld length   : {total_weld_len/1000:.3f} m")
        print(f"  Deposited volume    : {dep_vol_eff:.1f} mm³  "
              f"= {dep_vol_eff/1000:.2f} cc")
        print(f"  Wire consumed       : {wire_consumed_m:.3f} m  "
              f"({wire_consumed_mm:.0f} mm, incl. {(1/EFF-1)*100:.0f}% spatter)")
        print(f"  Deposit mass        : {mass_g:.2f} g  ({mass_g/1000:.4f} kg)")
        print(f"  Arc-on time         : {arc_time_min:.1f} min  "
              f"({total_weld_time:.0f} sec)")
        print(f"  Interlayer dwell    : {dwell_time_min:.1f} min  "
              f"({ls_dwell:.0f}s × {len(self.layers)} layers)")
        print(f"  Travel distance     : {total_travel_len/1000:.3f} m  "
              f"@ {travel_spd:.0f} mm/sec")
        print(f"  Travel time         : {travel_time_min:.1f} min  "
              f"({travel_time_sec:.0f} sec)")
        print(f"  ── Total est. time  : {total_time_min:.1f} min  "
              f"({total_time_min/60:.2f} hr)")

        print(f"\n  ── Gap Compensation ──")
        print(f"  Layers with gaps    : {n_mismatch}/{len(self.layers)}")
        if real_gaps:
            print(f"  Max gap             : {max(real_gaps):.3f} mm")
            print(f"  Avg gap (affected)  : "
                  f"{sum(real_gaps)/len(real_gaps):.3f} mm")
        else:
            print(f"  No significant gaps detected.")
        print(f"  Strategy 1 (spacing): {n_s1_total} regions")
        print(f"  Strategy 2 (ep pass): {n_s2_total} regions "
              f"({total_ep} passes)")

        if n_s1_total > 0:
            s1_overlaps = []
            for layer in self.layers:
                for m in layer.mismatch_info:
                    if m.strategy == "adjusted_spacing" and m.n_lines > 1:
                        actual_step = ((m.region_width_mm - bw)
                                       / (m.n_lines - 1))
                        s1_overlaps.append((bw - actual_step) / bw * 100)
            if s1_overlaps:
                print(f"  S1 overlap range    : "
                      f"{min(s1_overlaps):.1f}%–{max(s1_overlaps):.1f}% "
                      f"(nominal {cfg.overlap_pct:.0f}%)")

        # ── Overhang criticality ──────────────────────────────────
        oh_lim = cfg.overhang_limit_deg
        print(f"\n  ── Overhang Criticality (limit={oh_lim:.0f}°) ──")
        safe_l     = [r for r in records if r["oh_deg"] < oh_lim * 0.6]
        caution_l  = [r for r in records
                      if oh_lim * 0.6 <= r["oh_deg"] < oh_lim]
        critical_l = [r for r in records if r["oh_deg"] >= oh_lim]

        print(f"  Safe (<{oh_lim*0.6:.0f}°)        : {len(safe_l)} layers")
        print(f"  Caution ({oh_lim*0.6:.0f}°–{oh_lim:.0f}°)  : "
              f"{len(caution_l)} layers")
        print(f"  Critical (≥{oh_lim:.0f}°)     : {len(critical_l)} layers")

        if critical_l:
            zs = [r["z_mm"] for r in critical_l]
            print(f"  Critical Z range    : {min(zs):.2f}–{max(zs):.2f} mm")
            print(f"  ⚠  Increase dwell 2× at caution, 3× at critical layers.")

        # Crown detection
        crown_layers = []
        for i, r in enumerate(records[:-2]):
            z  = r["z_mm"]
            dz = records[i+1]["thickness_mm"] or 2.5
            try:
                c0 = self.mesh.section(
                    plane_origin=[0,0,z], plane_normal=[0,0,1])
                c1 = self.mesh.section(
                    plane_origin=[0,0,z+dz], plane_normal=[0,0,1])
                if c0 and c1:
                    a0 = c0.to_2D()[0].area
                    a1 = c1.to_2D()[0].area
                    if a0 > 0 and a1/a0 < 0.85 and a1 < 200:
                        crown_layers.append(
                            (r["layer"], z, round(a0,1), round(a1,1)))
            except Exception:
                pass

        if crown_layers:
            print(f"\n  Crown/closure layers:")
            print(f"  {'Lyr':>4}  {'Z':>6}  {'Area_now':>9}  {'Area_nxt':>9}")
            for lyr, z, a0, a1 in crown_layers:
                flag = " ⚠ CLOSURE" if a1/a0 < 0.5 else ""
                print(f"  {lyr:>4}  {z:>6.2f}  {a0:>9.1f}  {a1:>9.1f}{flag}")
            print(f"  → Increase dwell 3–5× at crown layers.")
            print(f"    If collapse occurs, increase crown radius in CAD.")
        else:
            print(f"  No crown approach detected.")

        # ── Per-layer table ───────────────────────────────────────
        print(f"\n  ── Per-layer Breakdown ──")
        hdr = (f"  {'Lyr':>4}  {'z':>6}  {'t':>6}  {'spd':>5}  "
               f"{'pths':>5}  {'ep':>4}  {'S1':>3}  {'S2':>3}  "
               f"{'gap':>6}  {'OH°':>5}")
        print(hdr)
        print("  " + "─"*(len(hdr)-2))
        for r in records:
            t_s  = f"{r['thickness_mm']:.3f}" if r['thickness_mm'] else "  — "
            sp_s = f"{r['weld_speed_mms']:.1f}" if r['weld_speed_mms'] else " — "
            g_s  = f"{r['max_gap_mm']:.2f}" if r['max_gap_mm'] > 0 else " — "
            oh_s = f"{r['oh_deg']:.1f}" if r['oh_deg'] > 0 else " — "
            print(f"  {r['layer']:>4}  {r['z_mm']:>6.2f}  {t_s:>6}  "
                  f"{sp_s:>5}  {r['infill_paths']:>5}  {r['edge_passes']:>4}  "
                  f"{r['n_strategy1']:>3}  {r['n_strategy2']:>3}  "
                  f"{g_s:>6}  {oh_s:>5}")
        print("="*W)

        # ── CSV export ────────────────────────────────────────────
        if export_csv:
            import csv as _csv
            try:
                with open(csv_path, "w", newline="") as f:
                    writer = _csv.DictWriter(f,
                                fieldnames=list(records[0].keys()))
                    writer.writeheader()
                    writer.writerows(records)
                print(f"\n  Per-layer report → {csv_path}")
            except Exception as e:
                print(f"\n  WARNING: CSV export failed — {e}")


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _to_linestring_list(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if hasattr(geom, "geoms"):
        result = []
        for g in geom.geoms:
            result.extend(_to_linestring_list(g))
        return result
    return []


# ─────────────────────────────────────────────
#  Quick-start
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_mesh = trimesh.creation.cylinder(radius=20, height=40, sections=64)

    cfg = SlicerConfig(
        layer_height    = 2.0,
        bead_width      = 5.0,
        overlap_pct     = 10.0,
        num_contours    = 2,
        infill_type     = InfillType.RASTER,
        raster_angle    = 0.0,
        angle_increment = 90.0,
        line_order      = LineOrder.INTERLEAVED,
        direction_mode  = DirectionMode.BOUSTROPHEDON,
        seam_mode       = SeamMode.ROTATE,
        seam_rotate_deg = 67.0,
    )

    slicer = WAAMSlicer(cfg).load_mesh_object(test_mesh).slice()
    slicer.print_summary()
    slicer.visualize_multiple_layers()
    slicer.visualize_3d_stack()