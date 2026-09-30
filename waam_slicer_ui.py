"""
WAAM Slicer — Desktop UI
========================
Run this file directly:  python waam_slicer_ui.py
Both waam_slicer.py and waam_ls_generator.py must be in the same folder.

Dependencies:
    pip install trimesh numpy shapely matplotlib scipy
    tkinter is built into Python — no install needed.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
import io
import matplotlib.pyplot as plt
import math


# ── Redirect stdout to the UI console ─────────────────────────────────────────

class ConsoleRedirect(io.StringIO):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def write(self, text):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

    def flush(self):
        pass


# ── Parameter definitions ──────────────────────────────────────────────────────
# (label, key, default, type, description, dropdown_options)
# type: "file" | "folder" | "float" | "int" | "str" | "dropdown"

PARAMETERS = [

    # ── STL File ──────────────────────────────────────────────────────────────
    ("__section__", "STL File", None, None, None, None),
    ("STL File Path", "stl_path", "", "file",
     "Path to your .stl file. Click Browse to select.\n"
     "The slicer auto-translates the mesh base to Z=0.", None),

    # ── Layer Settings ─────────────────────────────────────────────────────────
    ("__section__", "Layer Settings", None, None, None, None),
    ("Layer Height (mm)", "layer_height", "2.5", "float",
     "Layer thickness in constant-height mode.\n"
     "Typical WAAM range: 2.0 – 3.0 mm.\n"
     "Ignored when Adaptive Layer Thickness = on.", None),
    ("Adaptive Layer Thickness", "adaptive_layers", "off", "dropdown",
     "ON: layer thickness varies with surface inclination.\n"
     "Near-vertical walls get thicker layers (faster build).\n"
     "Near-horizontal surfaces get thinner layers (better accuracy).\n"
     "Uses your DoE bead model — enable Use Bead Model below.",
     ["off", "on"]),
    ("Use Bead Model (DoE)", "use_bead_model", "on", "dropdown",
     "ON: uses waam_bead_models.py (your DoE data) to:\n"
     "  1. Invert height vs speed to set R[2] per layer\n"
     "  2. Invert width vs speed for edge compensation passes\n"
     "  3. Override bead height bounds from physical measurements\n"
     "OFF: uses manual weld speed from LS Settings.\n"
     "Requires waam_bead_models.py in the same folder.",
     ["on", "off"]),
    ("Overhang Limit (deg)", "overhang_limit_deg", "20.0", "float",
     "Maximum stable overhang angle measured from horizontal.\n"
     "Layers with overhanging surfaces below this angle are flagged.\n"
     "Typical GMAW-WAAM: 15–25°.", None),
    ("Cusp Height C_max (mm)", "cusp_height_max", "1.0", "float",
     "Maximum allowed surface staircase deviation between layers.\n"
     "Consistent with Dolenc & Mäkelä (1994): 0.5–1.5 mm typical.\n"
     "Smaller = more layers and finer surface reproduction.", None),

    # ── Bead Geometry ──────────────────────────────────────────────────────────
    ("__section__", "Bead Geometry", None, None, None, None),
    ("Bead Width (mm)", "bead_width", "6.0", "float",
     "Nominal single-bead width from your DoE experiments.\n"
     "When Use Bead Model = on, actual deposited width is computed\n"
     "per layer from speed via waam_bead_models.py.\n"
     "This value sets the thin-wall detection threshold.", None),
    ("Overlap (%)", "overlap_pct", "30.0", "float",
     "Bead-to-bead overlap percentage.\n"
     "30% is recommended for flat GMAW beads (Xiong et al. 2019).\n"
     "50% or higher causes height accumulation — avoid for solid fills.", None),

    # ── Wall & Infill ──────────────────────────────────────────────────────────
    ("__section__", "Wall & Infill", None, None, None, None),
    ("Number of Wall Passes", "num_contours", "1", "int",
     "Perimeter contour passes before infill.\n"
     "0 = no wall passes (infill only).\n"
     "1 = one wall pass (recommended for most geometries).", None),
    ("Infill Type", "infill_type", "contour", "dropdown",
     "raster  — parallel lines. Good for solid fills.\n"
     "contour — concentric rings. Good for thin walls and annular parts.\n"
     "For solid fills, raster avoids the pyramid accumulation problem.",
     ["raster", "contour"]),
    ("Raster Angle (deg)", "raster_angle", "0.0", "float",
     "Base angle of raster lines for layer 0.\n"
     "0 = horizontal lines, 90 = vertical, 45 = diagonal.\n"
     "Only used when Infill Type = raster.", None),
    ("Angle Increment per Layer (deg)", "angle_increment", "90.0", "float",
     "Raster angle rotation per layer.\n"
     "90 = alternates 0°/90°. 0 = same angle every layer.\n"
     "Only used when Infill Type = raster.", None),
    ("Line Order", "line_order", "interleaved", "dropdown",
     "Raster line deposition order.\n"
     "  sequential  — top to bottom\n"
     "  alternating — even then odd lines\n"
     "  interleaved — maximally spread (best heat distribution)\n"
     "  random      — shuffled per layer\n"
     "Only used when Infill Type = raster.",
     ["sequential", "alternating", "interleaved", "random"]),
    ("Direction Mode", "direction_mode", "boustrophedon", "dropdown",
     "Torch travel direction per line.\n"
     "  boustrophedon — alternates direction per line (recommended)\n"
     "  same          — always left to right\n"
     "  random        — randomly flipped",
     ["boustrophedon", "same", "random"]),
    ("Thin Wall Alternating", "thin_wall_alternating", "on", "dropdown",
     "For thin-wall paths (single centreline), reverse direction each layer.\n"
     "ON = boustrophedon (recommended).\n"
     "OFF = always same direction.",
     ["on", "off"]),

    # ── Seam Control ───────────────────────────────────────────────────────────
    ("__section__", "Seam Control", None, None, None, None),
    ("Seam Mode", "seam_mode", "rotate", "dropdown",
     "Controls where the contour ring start/end point is placed.\n"
     "  rotate — advances each layer by Seam Rotation angle (recommended)\n"
     "  fixed  — always the same angle (creates visible vertical line)\n"
     "  random — random per layer",
     ["rotate", "fixed", "random"]),
    ("Seam Rotation per Layer (deg)", "seam_rotate_deg", "67.0", "float",
     "Degrees the seam advances per layer.\n"
     "67° is irrational relative to 360° — avoids visible patterns.\n"
     "Only used when Seam Mode = rotate.", None),
    ("Seam Fixed Angle (deg)", "seam_fixed_deg", "0.0", "float",
     "Fixed seam angle (degrees from +X axis).\n"
     "Only used when Seam Mode = fixed.", None),

    # ── Gap Compensation ───────────────────────────────────────────────────────
    ("__section__", "Gap Compensation", None, None, None, None),
    ("Mismatch Threshold (× bead width)", "mismatch_threshold_frac",
     "0.5", "float",
     "Gap fraction below which spacing is redistributed (Strategy 1).\n"
     "Above this, extra edge passes are added (Strategy 2).\n"
     "Default 0.5 = half a bead width.", None),
    ("Min Path Length (× bead width)", "min_path_length_frac",
     "0.5", "float",
     "Raster segments shorter than this fraction of bead width are discarded.\n"
     "Prevents arc restrike on tiny stubs. Range 0–1. Default 0.5.", None),

    # ── Torch Orientation ──────────────────────────────────────────────────────
    ("__section__", "Torch Orientation", None, None, None, None),
    ("Torch Tilt Enabled", "torch_tilt_enabled", "off", "dropdown",
     "Enables torch tilt (W/P) in the LS program.\n"
     "OFF: vertical torch, W=0 P=0 every layer.\n"
     "ON (Fixed): uses W Base and P Base below, same every layer.\n"
     "ON (Adaptive): W and P computed per layer from geometry.\n"
     "Enable Adaptive Tilt below for geometry-driven mode.",
     ["off", "on"]),
    ("R (torch spin, deg)", "torch_R_base", "-70.0", "float",
     "Torch rotation around its own axis.\n"
     "Fixed by your torch geometry — does not change with tilt.\n"
     "Typical Fanuc WAAM torch: -70.0°.", None),

    ("__section__", "Fixed Tilt (manual W, P)", None, None, None, None),
    ("W Base (deg)", "torch_W_base", "0.0", "float",
     "Torch direction in the XY plane.\n"
     "0° = torch tilts toward +X. 90° = toward +Y.\n"
     "Used when Adaptive Tilt = off and Torch Tilt = on.", None),
    ("P Base (deg)", "torch_P_base", "0.0", "float",
     "Torch tilt from vertical. 0 = vertical. 25 = 25° from vertical.\n"
     "Used when Adaptive Tilt = off and Torch Tilt = on.\n"
     "Set equal to your wall lean angle for fixed-tilt wall deposition.", None),

    ("__section__", "Adaptive Tilt (geometry-driven)", None, None, None, None),
    ("Adaptive Tilt Enabled", "adaptive_tilt_enabled", "off", "dropdown",
     "ON: W and P are computed per layer from geometry.\n"
     "Detects the dominant overhang direction using two methods:\n"
     "  Method A (mesh normals) — works for arch/crown geometry\n"
     "  Method B (centroid shift) — works for leaning walls\n"
     "Automatically stays vertical for symmetric/annular overhangs.\n"
     "Requires Torch Tilt Enabled = on.",
     ["off", "on"]),
    ("Max Tilt P_max (deg)", "adaptive_tilt_p_max", "25.0", "float",
     "Maximum torch tilt P that adaptive mode will assign.\n"
     "Set to your robot's safe wrist limit.\n"
     "P is capped at this value regardless of geometry severity.", None),
    ("Max W Step per Layer (deg)", "adaptive_tilt_w_max_step", "15.0",
     "float",
     "Maximum W direction change between consecutive layers.\n"
     "Rate-limits rapid direction swings that could cause wrist\n"
     "limit errors on the robot. 15° is a safe default.", None),
    ("Directional Consistency Min", "adaptive_tilt_consistency_min",
     "0.65", "float",
     "Minimum fraction of overhang area that must point in one direction\n"
     "for tilt to activate. Below this, overhang is too distributed\n"
     "(e.g. annular/dome) and torch stays vertical.\n"
     "Range 0–1. Default 0.65 (65%).", None),

    # ── LS Code Settings ───────────────────────────────────────────────────────
    ("__section__", "LS Code Settings", None, None, None, None),
    ("Program Name", "prog_name", "WAAM_PROG", "str",
     "Fanuc LS program name. No spaces. Max 8 characters.", None),
    ("User Frame (UF)", "uframe", "0", "int",
     "Fanuc User Frame number — must match robot controller.", None),
    ("Tool Number (UT)", "utool", "6", "int",
     "Fanuc Tool number — must match your welding torch.", None),
    ("Weld Schedule S1", "weld_start_s1", "1", "int",
     "First number in Weld Start[S1,S2].", None),
    ("Weld Schedule S2", "weld_start_s2", "3", "int",
     "Second number in Weld Start[S1,S2].", None),
    ("Travel Speed R[1] (mm/sec)", "travel_speed_val", "50", "float",
     "Non-weld travel speed stored in R[1].\n"
     "Used for air moves and inter-layer positioning.", None),
    ("Weld Speed R[2] (mm/sec)", "weld_speed_val", "5.0", "float",
     "Global weld speed stored in R[2] at program start.\n"
     "When Use Bead Model = on and Adaptive Layer = on,\n"
     "R[2] is overwritten per layer by the bead model speed.", None),
    ("Dwell Time R[3] (sec)", "dwell_val", "10", "float",
     "Interlayer cooling dwell time in seconds.\n"
     "Robot waits this long at the start of each layer.\n"
     "Increase at overhang-critical layers (see Overhang Report).", None),
    ("Lift Clearance (mm)", "lift_clearance_mm", "0", "float",
     "Z clearance height for inter-layer travel.\n"
     "0 = no lift (saves travel points but requires reliable retract).", None),
    ("Point Reduction Mode", "reduction_mode", "reduced", "dropdown",
     "  full    — all vertices kept (large file)\n"
     "  reduced — Douglas-Peucker simplification (recommended)\n"
     "  custom  — set tolerance manually below",
     ["full", "reduced", "custom"]),
    ("Curve Tolerance (mm)", "curve_tolerance_mm", "0.5", "float",
     "Max chord deviation for curve simplification.\n"
     "0.2–0.5 mm is imperceptible at WAAM scale.\n"
     "Only used when Point Reduction = reduced or custom.", None),
    ("Max Point Spacing (mm)", "max_point_spacing_mm", "0.0", "float",
     "Maximum distance between weld points. 0 = off.\n"
     "Enable for curved paths needing dense interpolation.", None),

    # ── Process Calculator ─────────────────────────────────────────────────
    ("__section__", "Process Calculator", None, None, None, None),
    ("Wire Diameter (mm)", "wire_diameter_mm", "1.2", "float",
     "Filler wire diameter in mm.\n"
     "Standard WAAM wire: 0.9, 1.2, 1.6 mm.", None),
    ("Material Density (g/cc)", "material_density", "8.0", "float",
     "Density of the deposited material.\n"
     "SS 304/316: 8.0 g/cc\n"
     "Mild steel: 7.85 g/cc\n"
     "Ti-6Al-4V: 4.43 g/cc\n"
     "Inconel 625: 8.44 g/cc", None),
    ("Wire Feed Rate (m/min)", "wire_feed_rate_m_min", "4.0", "float",
     "Wire feed rate from your welding parameter set (WPS).\n"
     "Used to cross-check arc time against wire consumption.\n"
     "Typical GMAW-WAAM: 3–8 m/min.", None),
    ("Deposition Efficiency (%)", "deposition_efficiency_pct", "95.0",
     "float",
     "Fraction of wire that becomes deposit (rest = spatter).\n"
     "GMAW-WAAM typical: 90–97%.\n"
     "Used to compute actual wire consumed = deposit / efficiency.", None),
    ("Travel Speed R[1] (mm/sec)", "travel_speed_mms", "50.0", "float",
     "Air-move (non-weld) speed used for inter-layer repositioning.\n"
     "Must match R[1] value in LS Settings above.\n"
     "Used for accurate travel time calculation.", None),
    ("Lift Clearance (mm)", "lift_clearance_mm_calc", "0.0", "float",
     "Z clearance height for inter-layer travel.\n"
     "Must match Lift Clearance in LS Settings above.\n"
     "Travel distance = lift + horizontal XY move + plunge.", None),
    ("Interlayer Dwell (sec)", "dwell_time_sec", "10.0", "float",
     "Interlayer cooling wait time in seconds.\n"
     "Must match Dwell Time R[3] in LS Settings above.", None),

    # ── Output ─────────────────────────────────────────────────────────────────
    ("__section__", "Output", None, None, None, None),
    ("Output Folder", "output_dir", "", "folder",
     "Folder for LS code and slice data.\n"
     "Leave blank to save alongside the STL file.", None),
]


# ── Main UI class ──────────────────────────────────────────────────────────────

class WAAMSlicerUI:

    def __init__(self, root):
        self.root = root
        self.root.title("WAAM Slicer  —  IIT Bombay")
        self.root.geometry("900x960")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        self.entries = {}
        self.bead_mode = tk.BooleanVar(value=True)
        self.slicer = None
        self.bracket_slicer = None
        self._custom_mesh = None
        self.out_dir_ = "."
        self._last_ls_path = None  # path of most recently generated LS file

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#11111b", pady=10)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="WAAM Slicer",
                 font=("Courier", 18, "bold"),
                 fg="#cba6f7", bg="#11111b").pack()
        tk.Label(title_frame,
                 text="STL → Fanuc LS  |  IIT Bombay  |  DoE Bead Model + Adaptive Tilt",
                 font=("Courier", 9), fg="#6c7086",
                 bg="#11111b").pack()

        # Bead model status pill
        bm_frame = tk.Frame(self.root, bg="#1e1e2e", pady=2)
        bm_frame.pack(fill="x", padx=10)
        try:
            from waam_bead_models import BeadModel as _BM
            bm_status = "✓  waam_bead_models.py loaded"
            bm_color = "#a6e3a1"
        except ImportError:
            bm_status = "⚠  waam_bead_models.py not found — bead model disabled"
            bm_color = "#f38ba8"
        tk.Label(bm_frame, text=bm_status,
                 font=("Courier", 8), fg=bm_color,
                 bg="#1e1e2e").pack(anchor="w")

        # Scrollable parameter area
        canvas_frame = tk.Frame(self.root, bg="#1e1e2e")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(canvas_frame, bg="#1e1e2e",
                           highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                                  command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg="#1e1e2e")
        self.scroll_frame.bind("<Configure>",
                               lambda e: canvas.configure(
                                   scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame,
                             anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(
                            -1 * (e.delta // 120), "units"))

        self._build_params()

        # Button bar — row 1
        btn1 = tk.Frame(self.root, bg="#1e1e2e", pady=4)
        btn1.pack(fill="x", padx=10)

        self._btn(btn1, "Analyse Geometry", self._analyse_geometry,
                  "#89dceb")
        self._btn(btn1, "Run Slicer", self._run_slicer,
                  "#a6e3a1", bold=True)
        self._btn(btn1, "Visualize Layers", self._visualize,
                  "#89b4fa")
        tk.Checkbutton(btn1, text="Bead View",
                       variable=self.bead_mode,
                       bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244",
                       activebackground="#1e1e2e",
                       activeforeground="#cdd6f4",
                       font=("Courier", 9)).pack(side="left", padx=4)
        self._btn(btn1, "3D View", self._view_3d,
                  "#cba6f7", bold=True)

        # Button bar — row 2
        btn2 = tk.Frame(self.root, bg="#1e1e2e", pady=4)
        btn2.pack(fill="x", padx=10)

        self._btn(btn2, "Save Slice Data", self._save_slices,
                  "#f9e2af")
        self._btn(btn2, "Mismatch Report", self._show_mismatch_report,
                  "#cba6f7")
        self._btn(btn2, "Generate LS Code", self._generate_ls,
                  "#f38ba8", bold=True)
        self._btn(btn2, "Visualise LS", self._visualise_ls,
                  "#89dceb")
        self._btn(btn2, "LS Offset Editor", self._ls_offset_editor,
                  "#89b4fa")
        self._btn(btn2, "Reset Defaults", self._reset_defaults,
                  "#45475a")

        # Geometry suggestion panel (hidden until analysis runs)
        self._suggestion_frame = tk.Frame(self.root, bg="#1e1e2e")
        self._suggestion_frame.pack(fill="x", padx=10, pady=(0, 4))
        self._suggestion_widgets = []  # rebuilt on each analysis

        # Console
        tk.Label(self.root, text="Console Output",
                 font=("Courier", 9, "bold"),
                 fg="#6c7086", bg="#1e1e2e").pack(anchor="w", padx=12)
        self.console = scrolledtext.ScrolledText(
            self.root, height=10, bg="#11111b", fg="#a6e3a1",
            font=("Courier", 9), state="disabled", relief="flat",
            insertbackground="#cdd6f4")
        self.console.pack(fill="x", padx=10, pady=(0, 10))

        sys.stdout = ConsoleRedirect(self.console)

    def _btn(self, parent, text, cmd, bg, bold=False):
        font = ("Courier", 10, "bold") if bold else ("Courier", 10)
        tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg="#1e1e2e", font=font,
                  relief="flat", padx=14, pady=5
                  ).pack(side="left", padx=4)

    def _build_params(self):
        for label, key, default, ptype, desc, options in PARAMETERS:

            if label == "__section__":
                sf = tk.Frame(self.scroll_frame,
                              bg="#313244", pady=4)
                sf.pack(fill="x", padx=5, pady=(10, 2))
                tk.Label(sf, text=f"  {key}",
                         font=("Courier", 10, "bold"),
                         fg="#cba6f7", bg="#313244").pack(anchor="w")
                continue

            row = tk.Frame(self.scroll_frame, bg="#1e1e2e", pady=3)
            row.pack(fill="x", padx=10)

            # Label — wide enough for long parameter names
            lf = tk.Frame(row, bg="#1e1e2e", width=290)
            lf.pack(side="left", fill="y")
            lf.pack_propagate(False)
            tk.Label(lf, text=label,
                     font=("Courier", 9, "bold"),
                     fg="#cdd6f4", bg="#1e1e2e",
                     anchor="w", justify="left",
                     wraplength=280).pack(anchor="w", pady=2)

            # Input
            inf = tk.Frame(row, bg="#1e1e2e")
            inf.pack(side="left", fill="x", expand=True)

            if ptype == "file":
                var = tk.StringVar(value=default)
                tk.Entry(inf, textvariable=var,
                         bg="#313244", fg="#cdd6f4",
                         font=("Courier", 9), relief="flat",
                         insertbackground="#cdd6f4",
                         width=38).pack(side="left", padx=(0, 4))
                tk.Button(inf, text="Browse",
                          command=lambda v=var: self._browse_file(v),
                          bg="#45475a", fg="#cdd6f4",
                          font=("Courier", 8),
                          relief="flat").pack(side="left")
                self.entries[key] = var

            elif ptype == "folder":
                var = tk.StringVar(value=default)
                tk.Entry(inf, textvariable=var,
                         bg="#313244", fg="#cdd6f4",
                         font=("Courier", 9), relief="flat",
                         insertbackground="#cdd6f4",
                         width=38).pack(side="left", padx=(0, 4))
                tk.Button(inf, text="Browse",
                          command=lambda v=var: self._browse_folder(v),
                          bg="#45475a", fg="#cdd6f4",
                          font=("Courier", 8),
                          relief="flat").pack(side="left")
                self.entries[key] = var

            elif ptype == "dropdown":
                var = tk.StringVar(value=default)
                om = tk.OptionMenu(inf, var, *options)
                om.config(bg="#313244", fg="#cdd6f4",
                          activebackground="#cba6f7",
                          activeforeground="#1e1e2e",
                          font=("Courier", 9),
                          relief="flat", bd=0,
                          highlightthickness=1,
                          highlightbackground="#555",
                          width=16, anchor="w")
                om["menu"].config(bg="#313244", fg="#cdd6f4",
                                  activebackground="#cba6f7",
                                  activeforeground="#1e1e2e",
                                  font=("Courier", 9))
                om.pack(side="left")
                self.entries[key] = var

            else:  # float / int / str
                var = tk.StringVar(value=default)
                tk.Entry(inf, textvariable=var,
                         bg="#313244", fg="#cdd6f4",
                         font=("Courier", 9), relief="flat",
                         insertbackground="#cdd6f4",
                         width=14).pack(side="left")
                self.entries[key] = var

            # Tooltip
            if desc:
                lbl = tk.Label(row, text="ℹ",
                               font=("Courier", 10),
                               fg="#6c7086", bg="#1e1e2e",
                               cursor="hand2")
                lbl.pack(side="left", padx=4)
                self._tooltip(lbl, desc)

            ttk.Separator(self.scroll_frame,
                          orient="horizontal").pack(
                fill="x", padx=10, pady=1)

    def _tooltip(self, widget, text):
        tip = None

        def show(e):
            nonlocal tip
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{e.x_root + 15}+{e.y_root + 10}")
            tk.Label(tip, text=text, bg="#313244", fg="#cdd6f4",
                     font=("Courier", 8), justify="left",
                     padx=8, pady=6, relief="flat",
                     wraplength=360).pack()

        def hide(e):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ── File browsing ──────────────────────────────────────────────────────

    def _browse_file(self, var):
        p = filedialog.askopenfilename(
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")])
        if p:
            var.set(p)

    def _browse_folder(self, var):
        p = filedialog.askdirectory()
        if p:
            var.set(p)

    # ── Reset ──────────────────────────────────────────────────────────────

    def _reset_defaults(self):
        for label, key, default, ptype, desc, options in PARAMETERS:
            if label == "__section__":
                continue
            if key in self.entries and default is not None:
                self.entries[key].set(default)

    # ── Collect config ─────────────────────────────────────────────────────

    def _collect_config(self):
        from waam_slicer import (SlicerConfig, InfillType, LineOrder,
                                 DirectionMode, SeamMode)

        def flt(k): return float(self.entries[k].get())

        def i_(k):  return int(self.entries[k].get())

        def s_(k):  return self.entries[k].get().strip()

        def on_(k): return s_(k) == "on"

        cfg = SlicerConfig(
            # Layer
            layer_height=flt("layer_height"),
            adaptive_layers=on_("adaptive_layers"),
            use_bead_model=on_("use_bead_model"),
            overhang_limit_deg=flt("overhang_limit_deg"),
            cusp_height_max=flt("cusp_height_max"),

            # Bead geometry
            bead_width=flt("bead_width"),
            overlap_pct=flt("overlap_pct"),

            # Wall & infill
            num_contours=i_("num_contours"),
            thin_wall_alternating=on_("thin_wall_alternating"),
            infill_type=InfillType(s_("infill_type")),
            raster_angle=flt("raster_angle"),
            angle_increment=flt("angle_increment"),
            line_order=LineOrder(s_("line_order")),
            direction_mode=DirectionMode(s_("direction_mode")),
            contour_inset_start=False,  # always inside-out (fixed)

            # Seam
            seam_mode=SeamMode(s_("seam_mode")),
            seam_rotate_deg=flt("seam_rotate_deg"),
            seam_fixed_deg=flt("seam_fixed_deg"),

            # Gap compensation
            mismatch_threshold_frac=flt("mismatch_threshold_frac"),
            min_path_length_frac=flt("min_path_length_frac"),

            # Torch
            torch_tilt_enabled=on_("torch_tilt_enabled"),
            torch_W_base=flt("torch_W_base"),
            torch_P_base=flt("torch_P_base"),
            torch_R_base=flt("torch_R_base"),
            adaptive_tilt_enabled=on_("adaptive_tilt_enabled"),
            adaptive_tilt_p_max=flt("adaptive_tilt_p_max"),
            adaptive_tilt_w_max_step=flt("adaptive_tilt_w_max_step"),
            adaptive_tilt_consistency_min=flt(
                "adaptive_tilt_consistency_min"),

            # Process calculator
            wire_diameter_mm=flt("wire_diameter_mm"),
            material_density=flt("material_density"),
            wire_feed_rate_m_min=flt("wire_feed_rate_m_min"),
            deposition_efficiency=flt("deposition_efficiency_pct") / 100.0,
            travel_speed_mms=flt("travel_speed_mms"),
            lift_clearance_mm=flt("lift_clearance_mm_calc"),
            dwell_time_sec=flt("dwell_time_sec"),
        )

        stl = s_("stl_path")
        odir = s_("output_dir")
        return cfg, stl, odir

    def _collect_ls_config(self):
        from waam_ls_generator import LSConfig

        def flt(k): return float(self.entries[k].get())

        def i_(k):  return int(self.entries[k].get())

        def s_(k):  return self.entries[k].get().strip()

        return LSConfig(
            prog_name=s_("prog_name"),
            uframe=i_("uframe"),
            utool=i_("utool"),
            weld_start_s1=i_("weld_start_s1"),
            weld_start_s2=i_("weld_start_s2"),
            weld_end_s1=i_("weld_start_s1"),
            weld_end_s2=i_("weld_start_s2"),
            travel_speed_val=flt("travel_speed_val"),
            weld_speed_val=flt("weld_speed_val"),
            dwell_val=flt("dwell_val"),
            lift_clearance_mm=flt("lift_clearance_mm"),
            max_point_spacing_mm=flt("max_point_spacing_mm"),
            reduction_mode=s_("reduction_mode"),
            curve_tolerance_mm=flt("curve_tolerance_mm"),
        )

    # ── CAD orientation viewer ────────────────────────────────────────────

    def _orient_part(self):
        """
        Opens a 3D viewer with rotation sliders.
        User can rotate the part and confirm the orientation
        before slicing. The rotated mesh is stored and used by
        Run Slicer automatically.
        """
        stl_path = self.entries.get("stl_path")
        path = stl_path.get().strip() if stl_path else ""
        if not path or not os.path.exists(path):
            messagebox.showwarning("No File",
                                   "Select an STL file first.")
            return

        def run():
            try:
                import trimesh
                import tkinter as tk
                from matplotlib.backends.backend_tkagg import (
                    FigureCanvasTkAgg, NavigationToolbar2Tk)
                from matplotlib.widgets import Slider
                import numpy as np

                loaded = trimesh.load(path)
                if isinstance(loaded, trimesh.Scene):
                    meshes = [g for g in loaded.geometry.values()
                              if isinstance(g, trimesh.Trimesh)]
                    mesh = trimesh.util.concatenate(meshes)
                else:
                    mesh = loaded.copy()

                win = tk.Toplevel()
                win.title("Preview & Orient Part")
                win.configure(bg="#0a0a0a")

                fig = plt.figure(figsize=(9, 7), facecolor="#0a0a0a")
                ax = fig.add_subplot(111, projection="3d")
                ax.set_facecolor("#0a0a0a")
                fig.patch.set_facecolor("#0a0a0a")

                self._orient_mesh = mesh.copy()
                self._orient_angles = [0.0, 0.0, 0.0]

                def draw_mesh(m):
                    ax.cla()
                    ax.set_facecolor("#0a0a0a")
                    verts = m.vertices
                    faces = m.faces
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                    poly3d = Poly3DCollection(
                        verts[faces], alpha=0.3,
                        facecolor="#00cfff", edgecolor="#006080",
                        linewidth=0.2)
                    ax.add_collection3d(poly3d)
                    b = m.bounds
                    ax.set_xlim(b[0, 0], b[1, 0])
                    ax.set_ylim(b[0, 1], b[1, 1])
                    ax.set_zlim(b[0, 2], b[1, 2])
                    ax.set_xlabel("X", color="#aaa")
                    ax.set_ylabel("Y", color="#aaa")
                    ax.set_zlabel("Z", color="#aaa")
                    ax.tick_params(colors="#888", labelsize=6)
                    ax.set_title(
                        f"Rx={self._orient_angles[0]:.0f}deg  "
                        f"Ry={self._orient_angles[1]:.0f}deg  "
                        f"Rz={self._orient_angles[2]:.0f}deg  |  "
                        f"Z: {b[0, 2]:.1f} to {b[1, 2]:.1f} mm",
                        color="white", fontsize=9)
                    mpl_canvas.draw()

                draw_mesh(mesh)

                mpl_canvas = FigureCanvasTkAgg(fig, master=win)
                mpl_canvas.draw()
                mpl_canvas.get_tk_widget().pack(
                    fill="both", expand=True)

                tb = tk.Frame(win, bg="#1e1e2e")
                tb.pack(fill="x", side="bottom")
                NavigationToolbar2Tk(mpl_canvas, tb).update()

                # Slider panel
                ctrl = tk.Frame(win, bg="#1e1e2e", pady=6)
                ctrl.pack(fill="x", side="bottom")

                def make_slider(parent, label, row):
                    tk.Label(parent, text=label,
                             fg="#cdd6f4", bg="#1e1e2e",
                             font=("Courier", 9),
                             width=4).grid(
                        row=row, column=0, padx=8)
                    var = tk.DoubleVar(value=0.0)
                    sl = tk.Scale(parent, from_=-180, to=180,
                                  resolution=1,
                                  orient="horizontal",
                                  variable=var, length=300,
                                  bg="#313244", fg="#cdd6f4",
                                  troughcolor="#1e1e2e",
                                  highlightthickness=0,
                                  font=("Courier", 8))
                    sl.grid(row=row, column=1, padx=4)
                    val_lbl = tk.Label(parent, textvariable=var,
                                       fg="#a6e3a1", bg="#1e1e2e",
                                       font=("Courier", 9), width=5)
                    val_lbl.grid(row=row, column=2)
                    return var

                rx_var = make_slider(ctrl, "Rx°", 0)
                ry_var = make_slider(ctrl, "Ry°", 1)
                rz_var = make_slider(ctrl, "Rz°", 2)

                align_var = tk.BooleanVar(value=True)
                tk.Checkbutton(ctrl, text="Auto-align base to Z=0",
                               variable=align_var,
                               bg="#1e1e2e", fg="#cdd6f4",
                               selectcolor="#313244",
                               activebackground="#1e1e2e",
                               font=("Courier", 9)
                               ).grid(row=3, column=0,
                                      columnspan=3, pady=4)

                def apply_rotation(*_):
                    rx = math.radians(rx_var.get())
                    ry = math.radians(ry_var.get())
                    rz = math.radians(rz_var.get())
                    self._orient_angles = [
                        rx_var.get(), ry_var.get(), rz_var.get()]
                    import trimesh.transformations as tf
                    Rx = tf.rotation_matrix(rx, [1, 0, 0])
                    Ry = tf.rotation_matrix(ry, [0, 1, 0])
                    Rz = tf.rotation_matrix(rz, [0, 0, 1])
                    T = Rz @ Ry @ Rx
                    m = mesh.copy()
                    m.apply_transform(T)
                    if align_var.get():
                        m.apply_translation(
                            [0, 0, -m.bounds[0, 2]])
                    self._orient_mesh = m
                    draw_mesh(m)

                for v in (rx_var, ry_var, rz_var):
                    v.trace_add("write",
                                lambda *_: apply_rotation())

                def confirm():
                    self._custom_mesh = self._orient_mesh
                    rx = rx_var.get();
                    ry = ry_var.get();
                    rz = rz_var.get()
                    print(f"\nOrientation confirmed: Rx={rx:.0f}deg  Ry={ry:.0f}deg  Rz={rz:.0f}deg")
                    b = self._custom_mesh.bounds
                    print(f"  Final Z: {b[0, 2]:.2f} to {b[1, 2]:.2f} mm")
                    win.destroy()

                def reset():
                    rx_var.set(0);
                    ry_var.set(0);
                    rz_var.set(0)

                btn_f = tk.Frame(ctrl, bg="#1e1e2e")
                btn_f.grid(row=4, column=0, columnspan=3, pady=6)
                tk.Button(btn_f, text="Confirm Orientation",
                          command=confirm,
                          bg="#a6e3a1", fg="#1e1e2e",
                          font=("Courier", 10, "bold"),
                          relief="flat", padx=14
                          ).pack(side="left", padx=6)
                tk.Button(btn_f, text="Reset",
                          command=reset,
                          bg="#45475a", fg="#cdd6f4",
                          font=("Courier", 10),
                          relief="flat", padx=10
                          ).pack(side="left", padx=6)

                sw = win.winfo_screenwidth()
                sh = win.winfo_screenheight()
                win.geometry(
                    f"{min(1000, int(sw * 0.8))}x"
                    f"{min(800, int(sh * 0.85))}")

            except Exception as e:
                print(f"\nOrientation viewer error: {e}")
                import traceback;
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()

    # ── Bracket slicer ─────────────────────────────────────────────────────

    def _run_bracket(self):
        stl_path = self.entries.get("stl_path")
        path = stl_path.get().strip() if stl_path else ""
        if not path or not os.path.exists(path):
            messagebox.showwarning("No File",
                                   "Select an STL file first.")
            return

        def run():
            try:
                from waam_bracket_slicer import BracketSlicer, BracketConfig
                import math

                def flt(k):
                    try:
                        return float(self.entries[k].get())
                    except:
                        return None

                bw = flt("bead_width") or 5.0
                lh = flt("layer_height") or 2.0

                cfg = BracketConfig(
                    bead_width=bw,
                    layer_height=lh,
                    weld_speed=flt("weld_speed_mms") or 4.0,
                    junction_overlap_mm=2.0,
                    merge_angle_threshold=45.0,
                )

                print("\n" + "=" * 50)
                print("  Bracket Slicer Starting")
                print("=" * 50)

                slicer = BracketSlicer(cfg)

                # Use oriented mesh if available
                if hasattr(self, "_custom_mesh") and self._custom_mesh is not None:
                    slicer.load_mesh_object(self._custom_mesh)
                else:
                    slicer.load_stl(path)

                slicer.slice()
                slicer.print_summary()
                self.bracket_slicer = slicer
                self.out_dir_ = (
                        self.entries["output_dir"].get().strip()
                        or os.path.dirname(path))
                print("\nDone. Use 'View Skeleton' to inspect paths.")

            except Exception as e:
                print(f"\nBracket slicer error: {e}")
                import traceback;
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()

    def _view_skeleton(self):
        if not hasattr(self, "bracket_slicer") or self.bracket_slicer is None:
            messagebox.showwarning("Not Ready",
                                   "Run Bracket Slicer first.")
            return

        def run():
            try:
                self.bracket_slicer.visualize_skeleton()
            except Exception as e:
                print(f"\nSkeleton view error: {e}")
                import traceback;
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()

    # ── LS Visualiser and Offset Editor ───────────────────────────────────────

    def _visualise_ls(self):
        """Open the LS code visualiser."""
        ls_path = self._last_ls_path
        if not ls_path or not os.path.exists(ls_path):
            # Ask user to browse for an LS file
            p = filedialog.askopenfilename(
                filetypes=[("LS files", "*.LS *.ls"),
                           ("All files", "*.*")],
                title="Select LS file to visualise")
            if not p:
                return
            ls_path = p
        try:
            from waam_ls_tools import LSVisualiser
            vis = LSVisualiser()
            vis.show(ls_path)
        except Exception as e:
            print(f"\nLS Visualiser error: {e}")
            import traceback;
            traceback.print_exc()

    def _ls_offset_editor(self):
        """Open the LS offset editor."""
        ls_path = self._last_ls_path
        if ls_path and not os.path.exists(ls_path):
            ls_path = None
        try:
            from waam_ls_tools import LSOffsetEditor
            ed = LSOffsetEditor()
            ed.show(ls_path)
        except Exception as e:
            print(f"\nLS Offset Editor error: {e}")
            import traceback;
            traceback.print_exc()

    # ── Geometry analysis ─────────────────────────────────────────────────────

    def _analyse_geometry(self):
        """
        Load the STL, run geometry analysis, and display suggestions
        in the panel above the console. Does NOT slice.
        """
        stl_path = self.entries.get("stl_path")
        path = stl_path.get().strip() if stl_path else ""
        if not path or not os.path.exists(path):
            messagebox.showwarning("No File",
                                   "Select an STL file first.")
            return

        def run():
            try:
                from waam_slicer import WAAMSlicer, SlicerConfig
                print("\nAnalysing geometry...")
                cfg, _, _ = self._collect_config()
                s = WAAMSlicer(cfg)
                if hasattr(self, "_custom_mesh") and self._custom_mesh:
                    s.load_mesh_object(self._custom_mesh)
                    s.analyse_geometry()
                else:
                    s.load_stl(path)  # analyse_geometry called inside

                profile = s.geometry_profile
                if not profile:
                    print("  No geometry profile — check STL file.")
                    return

                print(f"\n  Geometry type     : {profile['dominant_type']}")
                print(f"  Min wall thickness: {profile['min_wall_thickness']:.2f} mm")
                print(f"  Max wall thickness: {profile['max_wall_thickness']:.2f} mm")
                print(f"  Annular fraction  : {profile['annular_fraction'] * 100:.0f}%")
                if profile['multi_body_layers']:
                    print(f"  Multi-body layers : {profile['multi_body_layers']}")

                # Update suggestion panel on main thread
                self.root.after(0, lambda p=profile: self._show_suggestions(p))

            except Exception as e:
                print(f"\nAnalysis error: {e}")
                import traceback;
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()

    def _show_suggestions(self, profile: dict):
        """Rebuild the suggestion panel with analysis results."""
        import tkinter as tk

        # Clear old widgets
        for w in self._suggestion_widgets:
            w.destroy()
        self._suggestion_widgets.clear()

        dom = profile.get("dominant_type", "unknown")

        # Header row
        type_colours = {
            "solid": ("#a6e3a1", "#1e1e2e"),
            "thin_wall": ("#94e2d5", "#1e1e2e"),  # teal — thin wall
            "annular": ("#89b4fa", "#1e1e2e"),
            "mixed": ("#f9e2af", "#1e1e2e"),
            "multi_body": ("#cba6f7", "#1e1e2e"),
        }
        hdr_bg, hdr_fg = type_colours.get(dom, ("#45475a", "#cdd6f4"))

        hdr = tk.Frame(self._suggestion_frame, bg=hdr_bg, pady=4)
        hdr.pack(fill="x")
        self._suggestion_widgets.append(hdr)

        type_labels = {
            "solid": "Solid part",
            "thin_wall": "Thin-walled solid (single-bead wall)",
            "annular": "Hollow wall / annular part",
            "mixed": "Mixed — solid base + hollow walls",
            "multi_body": "Multi-body / bracket / frame",
        }
        tk.Label(hdr,
                 text=f"  Geometry: {type_labels.get(dom, dom)}  |  "
                      f"Wall: {profile['min_wall_thickness']:.1f}–"
                      f"{profile['max_wall_thickness']:.1f} mm  |  "
                      f"Annular: {profile['annular_fraction'] * 100:.0f}%",
                 font=("Courier", 9, "bold"),
                 fg=hdr_fg, bg=hdr_bg, anchor="w").pack(
            side="left", padx=6)

        # Suggestions
        sev_colours = {
            "ok": ("#313244", "#a6e3a1"),
            "warn": ("#313244", "#f9e2af"),
            "critical": ("#45475a", "#f38ba8"),
        }
        for sug in profile.get("suggestions", []):
            sev = sug.get("severity", "ok")
            bg, accent = sev_colours.get(sev, sev_colours["ok"])

            row = tk.Frame(self._suggestion_frame, bg=bg, pady=2)
            row.pack(fill="x", pady=1)
            self._suggestion_widgets.append(row)

            # Parameter name badge
            tk.Label(row,
                     text=f" {sug['param']} ",
                     font=("Courier", 8, "bold"),
                     fg=accent, bg=bg).pack(side="left", padx=(6, 0))

            # Recommended value
            tk.Label(row,
                     text=f"→ {sug['value']}",
                     font=("Courier", 8),
                     fg="#cdd6f4", bg=bg).pack(side="left", padx=(4, 8))

            # Reason (truncated, full text on hover)
            reason_lbl = tk.Label(row,
                                  text=sug["reason"].split("\n")[0][:80],
                                  font=("Courier", 8),
                                  fg="#6c7086", bg=bg,
                                  anchor="w")
            reason_lbl.pack(side="left", fill="x", expand=True)

            # Tooltip with full reason
            self._tooltip_on(reason_lbl, sug["reason"])

    def _tooltip_on(self, widget, text):
        tip = None

        def show(e):
            nonlocal tip
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{e.x_root + 15}+{e.y_root + 10}")
            tk.Label(tip, text=text, bg="#313244", fg="#cdd6f4",
                     font=("Courier", 8), justify="left",
                     padx=8, pady=6, relief="flat",
                     wraplength=400).pack()

        def hide(e):
            nonlocal tip
            if tip: tip.destroy(); tip = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ── Run slicer ─────────────────────────────────────────────────────────

    def _run_slicer(self):
        self.slicer = None
        try:
            cfg, stl_path, out_dir = self._collect_config()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return
        if not stl_path:
            messagebox.showerror("Missing Input",
                                 "Please select an STL file.")
            return

        def run():
            try:
                from waam_slicer import WAAMSlicer
                print("\n" + "=" * 50)
                print("  WAAM Slicer Starting")
                print("=" * 50)
                # Use oriented mesh if user confirmed orientation
                if hasattr(self, "_custom_mesh") and self._custom_mesh is not None:
                    s = WAAMSlicer(cfg).load_mesh_object(
                        self._custom_mesh).slice()
                else:
                    s = WAAMSlicer(cfg).load_stl(stl_path).slice()
                s.print_summary()
                self.slicer = s
                self.out_dir_ = (out_dir
                                 if out_dir
                                 else os.path.dirname(stl_path))
                print("\nDone. Use buttons above to visualize or export.")
            except Exception as e:
                print(f"\nERROR: {e}")
                import traceback;
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()

    # ── Visualize ──────────────────────────────────────────────────────────

    def _visualize(self):
        if not self.slicer:
            messagebox.showwarning("Not Ready", "Run the slicer first.")
            return
        beads = self.bead_mode.get()
        # Tk windows MUST be created on the main thread.
        # Schedule directly — no worker thread needed for window creation.
        try:
            self.slicer.visualize_multiple_layers(
                show_as_beads=beads,
                page_size=12)
        except Exception as e:
            print(f"\nVisualization error: {e}")
            import traceback;
            traceback.print_exc()

    # ── 3D view ────────────────────────────────────────────────────────────

    def _view_3d(self):
        if not self.slicer:
            messagebox.showwarning("Not Ready", "Run the slicer first.")
            return
        try:
            self.slicer.visualize_3d_stack()
        except Exception as e:
            print(f"\n3D view error: {e}")
            import traceback;
            traceback.print_exc()

    # ── Mismatch report ────────────────────────────────────────────────────

    def _show_mismatch_report(self):
        """
        Opens a dedicated window showing layer-by-layer fill compensation stats.
        Shows: layer index, z height, gap (mm), strategy, edge pass speed.
        Provides Export CSV button.
        """
        if not self.slicer:
            messagebox.showwarning("Not Ready", "Run the slicer first.")
            return

        import tkinter as tk
        from tkinter import ttk

        # Collect all mismatch records
        records = []
        for layer in self.slicer.layers:
            for m in layer.mismatch_info:
                records.append({
                    "layer": layer.index + 1,
                    "z_mm": round(layer.z, 3),
                    "region_w": round(m.region_width_mm, 3),
                    "n_lines": m.n_lines,
                    "filled_w": round(m.filled_width_mm, 3),
                    "gap_mm": round(m.gap_mm, 3),
                    "strategy": m.strategy,
                    "edge_speed": round(m.edge_speed, 2),
                })

        win = tk.Toplevel()
        win.title("Fill Compensation Report")
        win.configure(bg="#1e1e2e")
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{min(980, int(sw * 0.85))}x{min(600, int(sh * 0.75))}")

        # ── Summary bar ───────────────────────────────────────────
        total_layers = len(self.slicer.layers)
        n_s1 = sum(1 for r in records if r["strategy"] == "adjusted_spacing")
        n_s2 = sum(1 for r in records if r["strategy"] == "extra_edge_pass")
        gaps = [r["gap_mm"] for r in records]

        summary = tk.Frame(win, bg="#313244", pady=8)
        summary.pack(fill="x", padx=0)

        def stat(parent, label, val, colour="#cdd6f4"):
            f = tk.Frame(parent, bg="#313244")
            f.pack(side="left", padx=18)
            tk.Label(f, text=label, font=("Courier", 8),
                     fg="#6c7086", bg="#313244").pack()
            tk.Label(f, text=val, font=("Courier", 11, "bold"),
                     fg=colour, bg="#313244").pack()

        stat(summary, "Total layers", str(total_layers))
        stat(summary, "Layers affected", str(len(records)), "#f9e2af")
        stat(summary, "Strategy 1 (spacing adj)", str(n_s1), "#a6e3a1")
        stat(summary, "Strategy 2 (edge pass)", str(n_s2), "#cba6f7")
        if gaps:
            stat(summary, "Max gap", f"{max(gaps):.3f} mm", "#f38ba8")
            stat(summary, "Mean gap", f"{sum(gaps) / len(gaps):.3f} mm", "#f9e2af")
        else:
            stat(summary, "Max gap", "none", "#a6e3a1")

        # ── Table ─────────────────────────────────────────────────
        # Column order matches MismatchInfo fields.
        # "Region W" = total infill region width (approx_width)
        # "Filled W" = width actually covered by N rings
        # "Gap"      = centre void remaining
        cols_def = [
            ("Layer", "layer", 60),
            ("Z (mm)", "z_mm", 72),
            ("Infill W (mm)", "region_w", 90),
            ("N rings", "n_lines", 64),
            ("Filled W (mm)", "filled_w", 90),
            ("Gap (mm)", "gap_mm", 72),
            ("Strategy", "strategy", 150),
            ("Edge spd (mm/s)", "edge_speed", 90),
        ]

        table_frame = tk.Frame(win, bg="#1e1e2e")
        table_frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Header
        hdr = tk.Frame(table_frame, bg="#11111b")
        hdr.pack(fill="x")
        for h, _, w in cols_def:
            tk.Label(hdr, text=h, font=("Courier", 8, "bold"),
                     fg="#89b4fa", bg="#11111b", width=w // 7,
                     anchor="center", pady=4).pack(side="left", padx=1)

        # Scrollable rows
        scroll_canvas = tk.Canvas(table_frame, bg="#1e1e2e",
                                  highlightthickness=0)
        scrollbar = tk.Scrollbar(table_frame, orient="vertical",
                                 command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.pack(side="left", fill="both", expand=True)

        rows_frame = tk.Frame(scroll_canvas, bg="#1e1e2e")
        scroll_canvas.create_window((0, 0), window=rows_frame, anchor="nw")
        rows_frame.bind("<Configure>",
                        lambda e: scroll_canvas.configure(
                            scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.bind_all("<MouseWheel>",
                               lambda e: scroll_canvas.yview_scroll(
                                   -1 * (e.delta // 120), "units"))

        sev_colours = {
            "adjusted_spacing": "#a6e3a1",
            "extra_edge_pass": "#cba6f7",
        }

        if not records:
            tk.Label(rows_frame,
                     text="No fill compensation was applied in this slice.",
                     font=("Courier", 10), fg="#6c7086", bg="#1e1e2e",
                     pady=20).pack()
        else:
            for i, r in enumerate(records):
                bg = "#1e1e2e" if i % 2 == 0 else "#252535"
                row = tk.Frame(rows_frame, bg=bg)
                row.pack(fill="x")
                vals = [r["layer"], r["z_mm"], r["region_w"],
                        r["n_lines"], r["filled_w"], r["gap_mm"],
                        r["strategy"],
                        r["edge_speed"] if r["edge_speed"] > 0 else "—"]
                strat_col = sev_colours.get(r["strategy"], "#cdd6f4")
                for j, (_, _, w) in enumerate(cols_def):
                    col = strat_col if j == 6 else "#cdd6f4"
                    tk.Label(row, text=str(vals[j]),
                             font=("Courier", 8),
                             fg=col, bg=bg,
                             width=w // 7, anchor="center",
                             pady=3).pack(side="left", padx=1)

        # ── Export button ─────────────────────────────────────────
        def export_csv():
            import csv
            from tkinter.filedialog import asksaveasfilename
            path = asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All", "*.*")],
                title="Save Mismatch Report",
                initialfile="waam_fill_compensation.csv")
            if not path:
                return
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys()
                if records else [])
                writer.writeheader()
                writer.writerows(records)
            messagebox.showinfo("Saved", "Mismatch report saved to:\n" + path)

        btn_frame = tk.Frame(win, bg="#1e1e2e", pady=8)
        btn_frame.pack(fill="x")
        tk.Button(btn_frame, text="Export as CSV",
                  command=export_csv,
                  bg="#cba6f7", fg="#1e1e2e",
                  font=("Courier", 10, "bold"),
                  relief="flat", padx=14, pady=5
                  ).pack(side="left", padx=12)
        tk.Button(btn_frame, text="Close",
                  command=win.destroy,
                  bg="#45475a", fg="#cdd6f4",
                  font=("Courier", 10),
                  relief="flat", padx=10, pady=5
                  ).pack(side="left", padx=4)

    # ── Save slices ────────────────────────────────────────────────────────

    def _save_slices(self):
        if not self.slicer:
            messagebox.showwarning("Not Ready", "Run the slicer first.")
            return
        import json

        os.makedirs(self.out_dir_, exist_ok=True)
        out_path = os.path.join(self.out_dir_, "slice_data.json")

        if os.path.exists(out_path):
            if not messagebox.askyesno(
                    "File Exists",
                    f"Already exists:\n{out_path}\n\nOverwrite?"):
                print("  Save cancelled.")
                return

        data = []
        for layer in self.slicer.layers:
            data.append({
                "layer_index": layer.index,
                "z_mm": round(layer.z, 4),
                "infill_paths": [
                    [list(pt) for pt in p.coords]
                    for p in layer.infill_paths],
            })

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSlice data saved: {out_path}")
        print(f"  {len(data)} layers")
        messagebox.showinfo("Saved",
                            f"Slice data saved to:\n{out_path}")

    # ── Generate LS ────────────────────────────────────────────────────────

    def _generate_ls(self):
        if not self.slicer:
            messagebox.showwarning("Not Ready",
                                   "Run the slicer first.")
            return
        try:
            ls_cfg = self._collect_ls_config()
        except Exception as e:
            messagebox.showerror("Input Error",
                                 f"LS config error:\n{e}")
            return

        # Check for existing file BEFORE spawning thread — avoids race condition.
        # File check and user confirmation happen on the main thread synchronously.
        prog = ls_cfg.prog_name or "WAAM_PROG"
        out_path = os.path.join(self.out_dir_, f"{prog}.LS")
        os.makedirs(self.out_dir_, exist_ok=True)

        if os.path.exists(out_path):
            if not messagebox.askyesno(
                    "File Exists",
                    f"Already exists:\n{out_path}\n\nOverwrite?"):
                print("  LS generation cancelled.")
                return

        def run():
            try:
                from waam_ls_generator import LSGenerator
                print("\n" + "=" * 50)
                print("  Generating LS Code")
                print("=" * 50)
                gen = LSGenerator(ls_cfg)
                gen.save(out_path, self.slicer.layers)
                self._last_ls_path = out_path
                self.root.after(0, lambda: messagebox.showinfo(
                    "LS Generated", "LS file saved to:\n" + out_path))
            except Exception as e:
                print(f"\nLS generation error: {e}")
                import traceback;
                traceback.print_exc()

        threading.Thread(target=run, daemon=True).start()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TScrollbar",
                    background="#45475a",
                    troughcolor="#1e1e2e",
                    arrowcolor="#cdd6f4")

    root.option_add("*Menu.background", "#313244")
    root.option_add("*Menu.foreground", "#cdd6f4")
    root.option_add("*Menu.activeBackground", "#cba6f7")
    root.option_add("*Menu.activeForeground", "#1e1e2e")
    root.option_add("*Menu.font", "Courier 9")

    app = WAAMSlicerUI(root)
    root.mainloop()
