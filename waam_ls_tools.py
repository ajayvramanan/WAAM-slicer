"""
WAAM LS Tools
=============
Two utilities for working with generated Fanuc R-30iB LS programs:

  LSVisualiser  — Parses and visualises the 3D toolpath from an LS file.
                  Colour-coded by motion type (travel / weld / edge pass).
                  Layer-by-layer breakdown, statistics panel.
                  Run from UI: "Visualise LS" button.

  LSOffsetEditor — Applies real-world coordinate offsets (X, Y, Z, W, P, R)
                   to all position blocks in an LS file.
                   Integrates your existing delta-editor script.
                   Run from UI: "LS Offset Editor" button.
"""

import re
import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.lines import Line2D
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
#  Data model for parsed LS file
# ─────────────────────────────────────────────

@dataclass
class LSPoint:
    idx:   int
    x:     float
    y:     float
    z:     float
    w:     float = 0.0
    p:     float = 0.0
    r:     float = 0.0


@dataclass
class LSMove:
    line_num:  int
    point_idx: int
    speed_str: str        # "R[1]mm/sec", "R[2]mm/sec", "4.0mm/sec", etc.
    in_weld:   bool       # True if between Weld Start and Weld End
    layer:     int        # which layer this belongs to


@dataclass
class LSLayer:
    index:     int
    moves:     List[LSMove] = field(default_factory=list)


@dataclass
class ParsedLS:
    prog_name: str
    points:    Dict[int, LSPoint]     # P[idx] → LSPoint
    moves:     List[LSMove]
    layers:    List[LSLayer]
    errors:    List[str]

    @property
    def all_xyz(self):
        return [(p.x, p.y, p.z) for p in self.points.values()]

    @property
    def weld_moves(self):
        return [m for m in self.moves if m.in_weld]

    @property
    def travel_moves(self):
        return [m for m in self.moves if not m.in_weld]


# ─────────────────────────────────────────────
#  LS Parser
# ─────────────────────────────────────────────

class LSParser:
    """Parse a Fanuc R-30iB LS file into a structured data model."""

    POS_BLOCK = re.compile(
        r'P\[(\d+)\]\{.*?'
        r'X\s*=\s*(-?\d+\.?\d*)\s*mm.*?'
        r'Y\s*=\s*(-?\d+\.?\d*)\s*mm.*?'
        r'Z\s*=\s*(-?\d+\.?\d*)\s*mm.*?'
        r'W\s*=\s*(-?\d+\.?\d*)\s*deg.*?'
        r'P\s*=\s*(-?\d+\.?\d*)\s*deg.*?'
        r'R\s*=\s*(-?\d+\.?\d*)\s*deg',
        re.DOTALL | re.IGNORECASE
    )
    MOTION    = re.compile(r'^\s*\d+\s*:\s*L\s+P\[(\d+)\]\s+(.+?)\s*;',
                            re.MULTILINE)
    WELD_S    = re.compile(r'^\s*\d+\s*:\s*Weld\s+Start', re.MULTILINE)
    WELD_E    = re.compile(r'^\s*\d+\s*:\s*Weld\s+End',   re.MULTILINE)
    LAYER_CMT = re.compile(r'!\s*-+\s*LAYER\s+(\d+)',      re.IGNORECASE)
    PROG_NAME = re.compile(r'/PROG\s+(\S+)')

    def parse(self, path: str) -> ParsedLS:
        if not os.path.exists(path):
            return ParsedLS("unknown", {}, [], [],
                            [f"File not found: {path}"])
        with open(path, 'r', errors='replace') as f:
            content = f.read()

        prog_name = "unknown"
        m = self.PROG_NAME.search(content)
        if m:
            prog_name = m.group(1)

        # ── Parse positions ──────────────────────────────────────
        points = {}
        for m in self.POS_BLOCK.finditer(content):
            idx = int(m.group(1))
            points[idx] = LSPoint(
                idx=idx,
                x=float(m.group(2)), y=float(m.group(3)),
                z=float(m.group(4)), w=float(m.group(5)),
                p=float(m.group(6)), r=float(m.group(7)),
            )

        # ── Parse motion sequence ────────────────────────────────
        # Walk line by line tracking weld state and layer
        mn_content = content
        if '/POS' in content:
            mn_content = content[:content.index('/POS')]

        lines      = mn_content.split('\n')
        in_weld    = False
        cur_layer  = 0
        moves      = []
        layers     = []
        layer_map  = {0: LSLayer(index=0)}

        line_num_re = re.compile(r'^\s*(\d+)\s*:(.*)')

        for raw_line in lines:
            m = line_num_re.match(raw_line)
            if not m:
                continue
            lnum  = int(m.group(1))
            instr = m.group(2).strip()

            # Layer detection
            lm = self.LAYER_CMT.search(instr)
            if lm:
                cur_layer = int(lm.group(1))
                if cur_layer not in layer_map:
                    layer_map[cur_layer] = LSLayer(index=cur_layer)

            # Weld state tracking
            if re.match(r'Weld\s+Start', instr, re.I):
                in_weld = True
                continue
            if re.match(r'Weld\s+End', instr, re.I):
                in_weld = False
                continue

            # Motion lines
            mm = re.match(r'L\s+P\[(\d+)\]\s+(.+)', instr)
            if mm:
                pidx  = int(mm.group(1))
                speed = mm.group(2).strip()
                move  = LSMove(
                    line_num  = lnum,
                    point_idx = pidx,
                    speed_str = speed,
                    in_weld   = in_weld,
                    layer     = cur_layer,
                )
                moves.append(move)
                if cur_layer not in layer_map:
                    layer_map[cur_layer] = LSLayer(index=cur_layer)
                layer_map[cur_layer].moves.append(move)

        layers = [layer_map[k] for k in sorted(layer_map.keys())]

        errors = []
        missing = [m.point_idx for m in moves
                   if m.point_idx not in points]
        if missing:
            errors.append(f"Missing position blocks for: "
                          f"P{sorted(set(missing))[:10]}...")

        return ParsedLS(prog_name, points, moves, layers, errors)

    def classify_speed(self, speed_str: str) -> str:
        """Returns "travel", "weld", or "edge" based on speed string."""
        s = speed_str.upper()
        if 'R[1]' in s:
            return "travel"
        if 'R[2]' in s:
            return "weld"
        return "edge"   # explicit mm/sec = edge compensation pass


# ─────────────────────────────────────────────
#  LS Visualiser
# ─────────────────────────────────────────────

class LSVisualiser:
    """
    3D toolpath visualiser for Fanuc LS programs.
    Opens a Tk window with:
      - 3D plot of the full toolpath (travel=grey, weld=coloured, edge=red)
      - Statistics panel (points, moves, layers, bounding box)
      - Layer selector to highlight individual layers
      - Toggle for travel moves (often clutters the view)
    """

    def __init__(self, ls_path: str = None):
        self.ls_path   = ls_path
        self.parsed    = None
        self.parser    = LSParser()

    def show(self, ls_path: str = None):
        if ls_path:
            self.ls_path = ls_path
        if not self.ls_path:
            messagebox.showerror("No File",
                                 "No LS file specified.")
            return

        self.parsed = self.parser.parse(self.ls_path)
        if self.parsed.errors:
            print("LS parse warnings:")
            for e in self.parsed.errors:
                print(f"  {e}")

        self._build_window()

    def _build_window(self):
        parsed = self.parsed

        # ── Pre-compute stats (needed before any widget creation) ──
        n_pts    = len(parsed.points)
        n_weld   = sum(1 for m in parsed.moves if m.in_weld)
        n_travel = sum(1 for m in parsed.moves if not m.in_weld)
        n_edge   = sum(1 for m in parsed.moves
                       if self.parser.classify_speed(m.speed_str)=="edge")
        n_layers = len(parsed.layers)

        xyz = parsed.all_xyz
        if xyz:
            xs = [p[0] for p in xyz]
            ys = [p[1] for p in xyz]
            zs = [p[2] for p in xyz]
            bbox = (f"X: {min(xs):.1f}–{max(xs):.1f} mm\n"
                    f"Y: {min(ys):.1f}–{max(ys):.1f} mm\n"
                    f"Z: {min(zs):.1f}–{max(zs):.1f} mm")
        else:
            bbox = "No position data"

        # ── State variables ────────────────────────────────────────
        # Defined here so redraw() can close over them.
        # Widget creation (chk, Radiobutton) happens AFTER redraw is
        # defined — this is the fix for the "free variable" error.
        show_travel = tk.BooleanVar(value=False)
        show_edge   = tk.BooleanVar(value=True)
        layer_var   = tk.IntVar(value=-1)   # -1 = all layers

        # ── Window ────────────────────────────────────────────────
        win = tk.Toplevel()
        win.title(f"LS Visualiser — {parsed.prog_name}")
        win.configure(bg="#0a0a0a")
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{min(1300,int(sw*0.92))}x{min(820,int(sh*0.88))}")

        # ── Left panel ────────────────────────────────────────────
        left = tk.Frame(win, bg="#1e1e2e", width=240)
        left.pack(side="left", fill="y", padx=0)
        left.pack_propagate(False)

        def lbl(parent, text, bold=False, colour="#cdd6f4",
                size=9, pady=2):
            tk.Label(parent, text=text,
                     font=("Courier", size, "bold" if bold else "normal"),
                     fg=colour, bg="#1e1e2e",
                     anchor="w", justify="left",
                     wraplength=220).pack(
                fill="x", padx=10, pady=pady)

        lbl(left, "LS CODE VISUALISER", bold=True,
            colour="#cba6f7", size=10, pady=8)
        lbl(left, os.path.basename(self.ls_path),
            colour="#6c7086", size=8)

        # Stats
        tk.Frame(left, bg="#313244", height=1).pack(
            fill="x", padx=8, pady=6)
        lbl(left, "Statistics", bold=True, colour="#89b4fa")

        # Compute WPR ranges from position blocks
        all_w = [p.w for p in parsed.points.values()]
        all_p = [p.p for p in parsed.points.values()]
        all_r = [p.r for p in parsed.points.values()]
        tilt_active = any(abs(p) > 0.5 for p in all_p)
        w_range = f"{min(all_w):.1f} to {max(all_w):.1f}" if all_w else "—"
        p_range = f"{min(all_p):.1f} to {max(all_p):.1f}" if all_p else "—"
        r_val   = f"{all_r[0]:.1f}" if all_r else "—"

        for label, val, col in [
            ("Position blocks", str(n_pts),    "#cdd6f4"),
            ("Layers",          str(n_layers), "#cdd6f4"),
            ("Weld moves",      str(n_weld),   "#cdd6f4"),
            ("Travel moves",    str(n_travel), "#cdd6f4"),
            ("Edge pass moves", str(n_edge),   "#cdd6f4"),
            ("W range (deg)",   w_range,       "#f9e2af"),
            ("P range (deg)",   p_range,
             "#f9e2af" if tilt_active else "#a6e3a1"),
            ("R (fixed, deg)",  r_val,         "#6c7086"),
        ]:
            row = tk.Frame(left, bg="#1e1e2e")
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=label, font=("Courier", 8),
                     fg="#6c7086", bg="#1e1e2e",
                     anchor="w").pack(side="left")
            tk.Label(row, text=val, font=("Courier", 8, "bold"),
                     fg=col, bg="#1e1e2e",
                     anchor="e").pack(side="right")

        if tilt_active:
            lbl(left, "Torch tilt ACTIVE",
                colour="#f9e2af", bold=True, size=8, pady=2)
        else:
            lbl(left, "Torch vertical (P=0)",
                colour="#a6e3a1", size=8, pady=2)

        lbl(left, bbox, colour="#a6e3a1", size=8, pady=4)

        if parsed.errors:
            tk.Frame(left, bg="#f38ba8", height=1).pack(
                fill="x", padx=8, pady=4)
            lbl(left, "Warnings:", bold=True, colour="#f38ba8")
            for e in parsed.errors[:3]:
                lbl(left, e, colour="#f38ba8", size=7)

        tk.Frame(left, bg="#313244", height=1).pack(
            fill="x", padx=8, pady=8)
        lbl(left, "Display", bold=True, colour="#89b4fa")

        # ── 3D plot — built before controls so redraw can use it ──
        plot_frame = tk.Frame(win, bg="#0a0a0a")
        plot_frame.pack(side="right", fill="both", expand=True)

        fig = plt.figure(figsize=(10, 8), facecolor="#f0f0f0")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#f8f8f8")
        fig.patch.set_facecolor("#f0f0f0")

        mpl_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        mpl_canvas.draw()
        mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(plot_frame, bg="#1e1e2e")
        tb_frame.pack(fill="x", side="bottom")
        NavigationToolbar2Tk(mpl_canvas, tb_frame).update()

        cmap = plt.get_cmap("plasma")

        # ── redraw defined before any widget uses it as command ───
        def redraw():
            ax.cla()

            # ── Light background, no grid ──────────────────────────
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")

            pts      = parsed.points
            moves    = parsed.moves
            hi_layer = layer_var.get()
            n_lay    = max(len(parsed.layers), 1)
            draw_arrows = show_torch_arrows.get()

            for i, move in enumerate(moves):
                if move.point_idx not in pts:
                    continue
                if i == 0:
                    continue
                prev_move = moves[i-1]
                if prev_move.point_idx not in pts:
                    continue

                p0 = pts[prev_move.point_idx]
                p1 = pts[move.point_idx]

                if hi_layer >= 0 and move.layer != hi_layer:
                    continue

                spd   = self.parser.classify_speed(move.speed_str)
                layer = move.layer

                if spd == "travel":
                    if not show_travel.get():
                        continue
                    ax.plot([p0.x, p1.x], [p0.y, p1.y],
                            [p0.z, p1.z],
                            color="#aaaaaa", linewidth=0.6,
                            alpha=0.5, linestyle="--")
                elif spd == "weld":
                    frac  = layer / n_lay
                    # color = cmap(frac)
                    color = 'green'
                    ax.plot([p0.x, p1.x], [p0.y, p1.y],
                            [p0.z, p1.z],
                            color=color, linewidth=0.5,
                            alpha=1.0, antialiased=True)
                elif spd == "edge":
                    if not show_edge.get():
                        continue
                    ax.plot([p0.x, p1.x], [p0.y, p1.y],
                            [p0.z, p1.z],
                            color="#cc0000", linewidth=1.25,
                            alpha=1.0, antialiased=True)

            # ── Axes styling — light theme, no grid ────────────────
            ax.set_xlabel("X (mm)", color="black", labelpad=10, fontsize=14)
            ax.set_ylabel("Y (mm)", color="black", labelpad=10, fontsize=14)
            ax.set_zlabel("Z (mm)", color="black", labelpad=10, fontsize=14)
            ax.tick_params(colors="black", labelsize=14)

            # Pane faces: white/off-white
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor("#ebebeb")
            ax.yaxis.pane.set_facecolor("#f2f2f2")
            ax.zaxis.pane.set_facecolor("#f5f5f5")
            ax.xaxis.pane.set_edgecolor("#cccccc")
            ax.yaxis.pane.set_edgecolor("#cccccc")
            ax.zaxis.pane.set_edgecolor("#cccccc")

            # No grid lines
            ax.grid(False)

            layer_lbl = (f"Layer {hi_layer}" if hi_layer >= 0
                         else "All layers")
            # ax.set_title(
            #     f"LS Toolpath — {parsed.prog_name} — {layer_lbl}",
            #     color="#111111", fontsize=10, pad=12)

            # ── Torch direction arrows ────────────────────────────
            # Draw a short arrow at the start of each weld segment
            # showing the torch axis direction in 3D.
            # Torch Z-axis from Fanuc ZYZ (W, P, R):
            #   direction = (sin(P)cos(W), sin(P)sin(W), cos(P))
            # Arrow length = 5mm (visual indicator only).
            if draw_arrows and tilt_active:
                arrow_len  = 8.0   # mm — visual only
                drawn_layers = set()
                for move in moves:
                    if move.point_idx not in pts: continue
                    if not move.in_weld: continue
                    if hi_layer >= 0 and move.layer != hi_layer: continue
                    if move.layer in drawn_layers: continue   # once per layer
                    pt = pts[move.point_idx]
                    W_r = math.radians(pt.w)
                    P_r = math.radians(pt.p)
                    # Torch Z-axis direction vector
                    tx = math.sin(P_r) * math.cos(W_r)
                    ty = math.sin(P_r) * math.sin(W_r)
                    tz = math.cos(P_r)
                    # Only draw if torch is tilted
                    if abs(pt.p) > 0.5:
                        ax.quiver(pt.x, pt.y, pt.z,
                                  tx * arrow_len,
                                  ty * arrow_len,
                                  tz * arrow_len,
                                  color="#black",
                                  linewidth=1.5,
                                  arrow_length_ratio=0.4)
                        drawn_layers.add(move.layer)

            legend = [
                Line2D([0],[0], color='green', lw=1,
                       label="Tool path"),
                # Line2D([0],[0], color=cmap(0.85), lw=2,
                #        label="Weld (late layers)"),
                Line2D([0],[0], color="#cc0022",  lw=2,
                       label="Edge compensation"),
                # Line2D([0],[0], color="#aaaaaa",  lw=1.2,
                #        linestyle="--", label="Travel"),
            ]
            if draw_arrows and tilt_active:
                legend.append(
                    Line2D([0],[0], color="#black", lw=1.5,
                           label="Torch direction"))
            ax.legend(handles=legend,
                      facecolor="white", labelcolor="#111111",
                      fontsize=12, loc="upper left",
                      framealpha=0.9)

            mpl_canvas.draw()

        # ── Controls — created AFTER redraw is defined ────────────
        # This is the key fix: widgets that use command=redraw must
        # be created after redraw() exists in the local scope.

        tk.Checkbutton(left, text="Show travel moves",
                       variable=show_travel,
                       bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244",
                       activebackground="#1e1e2e",
                       activeforeground="#cdd6f4",
                       font=("Courier", 8),
                       command=redraw).pack(anchor="w", padx=10, pady=2)

        tk.Checkbutton(left, text="Show edge passes",
                       variable=show_edge,
                       bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244",
                       activebackground="#1e1e2e",
                       activeforeground="#cdd6f4",
                       font=("Courier", 8),
                       command=redraw).pack(anchor="w", padx=10, pady=2)

        lbl(left, "Highlight layer:", colour="#6c7086", pady=4)
        layer_frame = tk.Frame(left, bg="#1e1e2e")
        layer_frame.pack(fill="x", padx=8)

        tk.Radiobutton(layer_frame, text="All", value=-1,
                       variable=layer_var,
                       bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244",
                       activebackground="#1e1e2e",
                       font=("Courier", 8),
                       command=redraw).pack(side="left")

        layer_spin = tk.Spinbox(
            layer_frame, from_=0, to=max(0, n_layers - 1),
            textvariable=layer_var, width=5,
            bg="#313244", fg="#cdd6f4",
            font=("Courier", 8), relief="flat",
            command=redraw)
        layer_spin.pack(side="left", padx=6)
        layer_spin.bind("<Return>", lambda e: redraw())

        lbl(left, "(-1 = all layers)", colour="#6c7086", size=7)

        tk.Frame(left, bg="#313244", height=1).pack(
            fill="x", padx=8, pady=8)
        lbl(left, "Torch Tilt", bold=True, colour="#89b4fa")

        def show_wpr_chart():
            _show_wpr_chart(parsed, win)

        tk.Button(left, text="W/P/R vs Z chart",
                  command=show_wpr_chart,
                  bg="#f9e2af", fg="#1e1e2e",
                  font=("Courier", 8, "bold"),
                  relief="flat", padx=8, pady=4
                  ).pack(anchor="w", padx=10, pady=4)

        tk.Frame(left, bg="#313244", height=1).pack(
            fill="x", padx=8, pady=4)
        lbl(left, "Export", bold=True, colour="#89b4fa")

        def save_view_pdf():
            """Save the current 3D toolpath view as PDF/PNG/SVG."""
            path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF",  "*.pdf"),
                           ("PNG",  "*.png"),
                           ("SVG",  "*.svg")],
                initialfile=f"{parsed.prog_name}_toolpath.pdf",
                title="Save LS Toolpath View")
            if not path:
                return
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.25,
                        facecolor=fig.get_facecolor())
            print(f"  LS toolpath saved → {path}")

        def save_per_layer_pdf():
            """Save each layer as a separate page in a multi-page PDF."""
            from matplotlib.backends.backend_pdf import PdfPages
            path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf")],
                initialfile=f"{parsed.prog_name}_per_layer.pdf",
                title="Save Per-Layer PDF")
            if not path:
                return
            pts   = parsed.points
            moves = parsed.moves
            cmap  = plt.get_cmap("plasma")
            layer_ids = sorted(set(m.layer for m in moves if m.in_weld))
            n_lay     = max(layer_ids) if layer_ids else 1

            with PdfPages(path) as pdf:
                d = pdf.infodict()
                d["Title"] = f"WAAM LS Toolpath — {parsed.prog_name}"
                for layer_id in layer_ids:
                    fig_l = plt.figure(figsize=(8, 6),
                                       facecolor="#f8f8f8")
                    ax_l  = fig_l.add_subplot(111, projection="3d")
                    ax_l.set_facecolor("#f8f8f8")

                    layer_moves = [m for m in moves if m.layer == layer_id]
                    for i, move in enumerate(layer_moves):
                        if i == 0 or move.point_idx not in pts:
                            continue
                        prev = layer_moves[i-1]
                        if prev.point_idx not in pts:
                            continue
                        p0  = pts[prev.point_idx]
                        p1  = pts[move.point_idx]
                        spd = parser.classify_speed(move.speed_str)
                        if spd == "travel":
                            ax_l.plot([p0.x,p1.x],[p0.y,p1.y],
                                      [p0.z,p1.z],
                                      color="#aaaaaa", lw=0.6,
                                      linestyle="--", alpha=0.5)
                        elif spd == "weld":
                            ax_l.plot([p0.x,p1.x],[p0.y,p1.y],
                                      [p0.z,p1.z],
                                      color=cmap(layer_id/n_lay),
                                      lw=2.0)
                        elif spd == "edge":
                            ax_l.plot([p0.x,p1.x],[p0.y,p1.y],
                                      [p0.z,p1.z],
                                      color="#cc0022", lw=2.2)

                    ax_l.set_xlabel("X (mm)", fontsize=10)
                    ax_l.set_ylabel("Y (mm)", fontsize=10)
                    ax_l.set_zlabel("Z (mm)", fontsize=10)
                    ax_l.set_title(
                        f"{parsed.prog_name} — Layer {layer_id}",
                        fontsize=10)
                    fig_l.tight_layout()
                    pdf.savefig(fig_l, dpi=300, bbox_inches="tight",
                                facecolor=fig_l.get_facecolor())
                    plt.close(fig_l)
                    print(f"  Layer {layer_id} → PDF page done")
            print(f"  Per-layer PDF saved → {path}")

        tk.Button(left, text="💾 Save View (PDF/PNG/SVG)",
                  command=save_view_pdf,
                  bg="#313244", fg="#a6e3a1",
                  font=("Courier", 8, "bold"),
                  relief="flat", padx=8, pady=4
                  ).pack(anchor="w", padx=10, pady=2)

        tk.Button(left, text="📄 Save Per-Layer PDF",
                  command=save_per_layer_pdf,
                  bg="#313244", fg="#89b4fa",
                  font=("Courier", 8, "bold"),
                  relief="flat", padx=8, pady=4
                  ).pack(anchor="w", padx=10, pady=2)

        tk.Frame(left, bg="#313244", height=1).pack(
            fill="x", padx=8, pady=4)

        show_torch_arrows = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Show torch arrows",
                       variable=show_torch_arrows,
                       bg="#1e1e2e", fg="#cdd6f4",
                       selectcolor="#313244",
                       activebackground="#1e1e2e",
                       activeforeground="#cdd6f4",
                       font=("Courier", 8),
                       command=redraw).pack(
            anchor="w", padx=10, pady=2)

        win.protocol("WM_DELETE_WINDOW",
                     lambda: (plt.close(fig), win.destroy()))
        redraw()


# ─────────────────────────────────────────────
#  LS Offset Editor
# ─────────────────────────────────────────────

class LSOffsetEditor:
    """
    Apply real-world coordinate offsets (X, Y, Z, W, P, R) to all
    position blocks in a Fanuc LS file.

    Integrates the existing delta-editor workflow into the slicer UI.
    Supports:
      - Load LS file (browse or use last generated)
      - Enter delta values for each axis
      - Preview: shows current XYZ range and transformed range
      - Apply: writes the modified LS file (in-place or as new file)
      - Undo: re-load original to start over
    """

    # Matches:  X =   10.000  mm,   or  W =   0.000 deg,
    COORD_RE = re.compile(
        r'(\b[XYZWPR]\b)(\s*=\s*)(-?\d+\.?\d*)')

    def __init__(self):
        self.ls_path    = None
        self.original   = None   # original file content
        self.modified   = None   # after applying delta

    def show(self, ls_path: str = None):
        if ls_path:
            self.ls_path = ls_path
            self._load_file()

        self._build_window()

    def _load_file(self):
        if self.ls_path and os.path.exists(self.ls_path):
            with open(self.ls_path, 'r', errors='replace') as f:
                self.original = f.read()
            self.modified = None

    def _apply_delta(self, content: str, delta: dict) -> str:
        """Apply coordinate offsets to all position blocks."""
        lines   = content.split('\n')
        result  = []
        in_pos  = False

        for line in lines:
            if '/POS' in line:
                in_pos = True
            if in_pos:
                matches = self.COORD_RE.findall(line)
                if matches:
                    for axis, spacing, value in matches:
                        if axis in delta and delta[axis] != 0:
                            new_val = float(value) + delta[axis]
                            line = re.sub(
                                rf'\b{axis}\b\s*=\s*-?\d+\.?\d*',
                                f'{axis}{spacing}{new_val:.3f}',
                                line)
            result.append(line)
        return '\n'.join(result)

    def _get_xyz_range(self, content: str):
        """Extract X, Y, Z ranges from position blocks."""
        pos_re = re.compile(
            r'X\s*=\s*(-?\d+\.?\d*).*?'
            r'Y\s*=\s*(-?\d+\.?\d*).*?'
            r'Z\s*=\s*(-?\d+\.?\d*)',
            re.DOTALL)
        xs, ys, zs = [], [], []
        in_pos = False
        # Find /POS section
        if '/POS' in content:
            pos_section = content[content.index('/POS'):]
        else:
            pos_section = content
        for m in pos_re.finditer(pos_section):
            xs.append(float(m.group(1)))
            ys.append(float(m.group(2)))
            zs.append(float(m.group(3)))
        if not xs:
            return None
        return {
            "X": (min(xs), max(xs)),
            "Y": (min(ys), max(ys)),
            "Z": (min(zs), max(zs)),
        }

    def _build_window(self):
        win = tk.Toplevel()
        win.title("LS Offset Editor")
        win.configure(bg="#1e1e2e")
        win.geometry("700x580")

        # ── File section ──────────────────────────────────────────
        file_frame = tk.Frame(win, bg="#11111b", pady=8)
        file_frame.pack(fill="x")

        tk.Label(file_frame, text="LS FILE OFFSET EDITOR",
                 font=("Courier", 11, "bold"),
                 fg="#cba6f7", bg="#11111b").pack(side="left", padx=12)

        path_var = tk.StringVar(value=self.ls_path or "")
        path_entry = tk.Entry(file_frame, textvariable=path_var,
                              bg="#313244", fg="#cdd6f4",
                              font=("Courier", 8), relief="flat",
                              width=38)
        path_entry.pack(side="left", padx=8)

        def browse():
            p = filedialog.askopenfilename(
                filetypes=[("LS files", "*.LS *.ls"),
                           ("All files", "*.*")],
                title="Open LS File")
            if p:
                path_var.set(p)
                self.ls_path = p
                self._load_file()
                update_preview()

        tk.Button(file_frame, text="Browse",
                  command=browse,
                  bg="#45475a", fg="#cdd6f4",
                  font=("Courier", 8), relief="flat",
                  padx=8).pack(side="left")

        # ── Delta inputs ──────────────────────────────────────────
        delta_frame = tk.LabelFrame(win,
                                     text=" Coordinate Offsets (delta) ",
                                     font=("Courier", 9, "bold"),
                                     fg="#89b4fa", bg="#1e1e2e",
                                     pady=8)
        delta_frame.pack(fill="x", padx=12, pady=8)

        tk.Label(delta_frame,
                 text="Enter the offset to ADD to every position block.\n"
                      "Measured from physical weld table in robot world frame.",
                 font=("Courier", 8), fg="#6c7086",
                 bg="#1e1e2e", justify="left").pack(
            anchor="w", padx=10, pady=(0,6))

        axes   = ["X", "Y", "Z", "W", "P", "R"]
        units  = ["mm", "mm", "mm", "deg", "deg", "deg"]
        d_vars = {}

        grid = tk.Frame(delta_frame, bg="#1e1e2e")
        grid.pack(fill="x", padx=10)

        for col, (axis, unit) in enumerate(zip(axes, units)):
            tf = tk.Frame(grid, bg="#1e1e2e")
            tf.grid(row=0, column=col, padx=6)
            tk.Label(tf, text=f"{axis} ({unit})",
                     font=("Courier", 8, "bold"),
                     fg="#a6e3a1" if axis in "XYZ" else "#f9e2af",
                     bg="#1e1e2e").pack()
            var = tk.StringVar(value="0.0")
            tk.Entry(tf, textvariable=var, width=10,
                     bg="#313244", fg="#cdd6f4",
                     font=("Courier", 9), relief="flat",
                     justify="center").pack()
            d_vars[axis] = var

        # Quick-fill from your existing workflow
        tk.Label(delta_frame,
                 text="Tip: paste values from robot controller "
                      "User Frame display.",
                 font=("Courier", 7), fg="#6c7086",
                 bg="#1e1e2e").pack(anchor="w", padx=10, pady=(4,0))

        # ── Preview ───────────────────────────────────────────────
        prev_frame = tk.LabelFrame(win,
                                    text=" Coordinate Preview ",
                                    font=("Courier", 9, "bold"),
                                    fg="#89b4fa", bg="#1e1e2e",
                                    pady=6)
        prev_frame.pack(fill="x", padx=12, pady=4)

        prev_text = tk.Text(prev_frame, height=6,
                            bg="#11111b", fg="#a6e3a1",
                            font=("Courier", 8), relief="flat",
                            state="disabled")
        prev_text.pack(fill="x", padx=8, pady=4)

        def update_preview(*_):
            if not self.original:
                return
            try:
                delta = {ax: float(d_vars[ax].get())
                         for ax in axes}
            except ValueError:
                return

            orig_range = self._get_xyz_range(self.original)
            if not orig_range:
                return

            lines = ["  Axis    Current range           After offset"]
            lines.append("  " + "─"*50)
            for ax in ["X", "Y", "Z"]:
                lo, hi = orig_range[ax]
                nlo    = lo + delta[ax]
                nhi    = hi + delta[ax]
                lines.append(
                    f"  {ax}:  [{lo:>10.3f}, {hi:>10.3f}]  →  "
                    f"[{nlo:>10.3f}, {nhi:>10.3f}] mm")
            lines.append("")
            lines.append(
                f"  W offset: {delta['W']:+.3f}deg   "
                f"P offset: {delta['P']:+.3f}deg   "
                f"R offset: {delta['R']:+.3f}deg")

            prev_text.configure(state="normal")
            prev_text.delete("1.0", tk.END)
            prev_text.insert("1.0", "\n".join(lines))
            prev_text.configure(state="disabled")

        for var in d_vars.values():
            var.trace_add("write", lambda *_: update_preview())

        # ── Buttons ───────────────────────────────────────────────
        btn_frame = tk.Frame(win, bg="#1e1e2e", pady=10)
        btn_frame.pack(fill="x", padx=12)

        status_var = tk.StringVar(value="")
        tk.Label(btn_frame, textvariable=status_var,
                 font=("Courier", 8), fg="#a6e3a1",
                 bg="#1e1e2e").pack(side="bottom", anchor="w")

        def apply_offset(in_place=True):
            if not self.original:
                messagebox.showwarning("No File",
                                       "Load an LS file first.")
                return
            try:
                delta = {ax: float(d_vars[ax].get())
                         for ax in axes}
            except ValueError as e:
                messagebox.showerror("Invalid Input",
                                     f"Enter numeric values only.\n{e}")
                return

            # Check if any delta is non-zero
            if all(v == 0 for v in delta.values()):
                messagebox.showinfo("No Change",
                                    "All delta values are 0. "
                                    "Nothing to apply.")
                return

            self.modified = self._apply_delta(self.original, delta)

            if in_place:
                out_path = self.ls_path
            else:
                base, ext = os.path.splitext(self.ls_path)
                out_path  = base + "_offset" + ext
                out_path  = filedialog.asksaveasfilename(
                    defaultextension=ext,
                    initialfile=os.path.basename(out_path),
                    filetypes=[("LS files", "*.LS *.ls"),
                               ("All files", "*.*")],
                    title="Save Offset LS File")
                if not out_path:
                    return

            with open(out_path, 'w') as f:
                f.write(self.modified)

            status_var.set(
                f"Saved: {os.path.basename(out_path)}  "
                f"({len(self.modified)} chars)")
            messagebox.showinfo(
                "Saved",
                f"Offset LS file saved to:\n{out_path}")

        def reset():
            for ax in axes:
                d_vars[ax].set("0.0")
            self.modified = None
            status_var.set("Reset — showing original values.")
            update_preview()

        for text, cmd, bg in [
            ("Preview",       update_preview,              "#89b4fa"),
            ("Apply (overwrite)", lambda: apply_offset(True),  "#f9e2af"),
            ("Save as new file",  lambda: apply_offset(False), "#a6e3a1"),
            ("Reset",         reset,                       "#45475a"),
        ]:
            tk.Button(btn_frame, text=text, command=cmd,
                      bg=bg, fg="#1e1e2e",
                      font=("Courier", 9, "bold"),
                      relief="flat", padx=12, pady=5
                      ).pack(side="left", padx=4)

        update_preview()


# ─────────────────────────────────────────────
#  WPR Chart — separate popup window
# ─────────────────────────────────────────────

def _show_wpr_chart(parsed: ParsedLS, parent_win=None):
    """
    Opens a separate window showing W, P, R values versus Z height.

    For each layer (identified by the layer comment in the LS file),
    finds the first weld point and reads its W, P, R values.
    Plots three lines: W (direction), P (tilt magnitude), R (fixed).

    This is the primary verification tool for the torch tilt feature:
      - A flat horizontal line for P = constant means no tilt variation
      - A varying P line means the tilt is responding to geometry
      - W should change direction as overhangs rotate around the part
      - R should be constant throughout (fixed by torch geometry)
    """
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)

    # Build per-layer WPR data:
    # For each layer, find the first in_weld move and use its point's WPR.
    # Group moves by layer, take the first weld point of each.
    layer_wpr = {}
    for move in parsed.moves:
        if not move.in_weld:
            continue
        if move.point_idx not in parsed.points:
            continue
        if move.layer not in layer_wpr:
            pt = parsed.points[move.point_idx]
            layer_wpr[move.layer] = (pt.z, pt.w, pt.p, pt.r)

    if not layer_wpr:
        if parent_win:
            messagebox.showwarning("No Data",
                "No weld points found in LS file.",
                parent=parent_win)
        return

    layers_sorted = sorted(layer_wpr.keys())
    zs = [layer_wpr[L][0] for L in layers_sorted]
    ws = [layer_wpr[L][1] for L in layers_sorted]
    ps = [layer_wpr[L][2] for L in layers_sorted]
    rs = [layer_wpr[L][3] for L in layers_sorted]

    tilt_active = any(abs(p) > 0.5 for p in ps)

    win = tk.Toplevel(parent_win)
    win.title(f"W / P / R vs Z — {parsed.prog_name}")
    win.configure(bg="#f0f0f0")
    sw = win.winfo_screenwidth()
    win.geometry(f"{min(900,int(sw*0.7))}x550")

    fig, axes = plt.subplots(3, 1, figsize=(9, 7),
                              facecolor="#f8f8f8", sharex=True)
    fig.suptitle(
        f"Torch Orientation vs Build Height — {parsed.prog_name}\n"
        f"{'Torch tilt ACTIVE' if tilt_active else 'Torch vertical throughout (P = 0)'}",
        fontsize=11, color="#111")

    labels = ["W — tilt direction (deg)", "P — tilt magnitude (deg)",
              "R — torch spin (deg, fixed)"]
    colours = ["#1565C0", "#e65c00", "#555555"]
    data    = [ws, ps, rs]

    for ax, d, label, col in zip(axes, data, labels, colours):
        ax.plot(zs, d, color=col, linewidth=2.0, marker="o",
                markersize=3, alpha=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, color="#dddddd", linewidth=0.5)
        ax.set_facecolor("#ffffff")
        ax.tick_params(labelsize=8)
        # Shade zero line
        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")

    axes[-1].set_xlabel("Z height (mm)", fontsize=9)
    fig.tight_layout()

    mpl_canvas = FigureCanvasTkAgg(fig, master=win)
    mpl_canvas.draw()
    mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

    tb = tk.Frame(win, bg="#eeeeee")
    tb.pack(fill="x", side="bottom")
    NavigationToolbar2Tk(mpl_canvas, tb).update()

    win.protocol("WM_DELETE_WINDOW",
                 lambda: (plt.close(fig), win.destroy()))


# ─────────────────────────────────────────────
#  Standalone entry points
# ─────────────────────────────────────────────

def show_visualiser(ls_path: str = None):
    vis = LSVisualiser()
    vis.show(ls_path)


def show_offset_editor(ls_path: str = None):
    ed = LSOffsetEditor()
    ed.show(ls_path)


if __name__ == "__main__":
    import sys
    root = tk.Tk()
    root.withdraw()
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if "--offset" in sys.argv:
        show_offset_editor(path)
    else:
        show_visualiser(path)
    root.mainloop()
