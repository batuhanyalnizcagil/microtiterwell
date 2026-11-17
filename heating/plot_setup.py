#!/usr/bin/env python3
"""
Figures documenting the set-up of the heating case (no simulation results).

    python3 plot_setup.py            # reads case/, writes results/<sub-folder>/

Output:

    results/geometry/geometry.png, geometry.txt
        dimensions of the well, of the modelled fluid region and of the wedge
    results/mesh/mesh.png, mesh_summary.txt
        the computational mesh (whole wedge, zooms, top view) and its statistics
    results/boundary_conditions/boundary_conditions.png
        which condition is applied on which boundary, heater temperature ramp
    results/boundary_conditions/boundary_conditions_table.png, .csv
        boundary-condition type of every field on every patch, read from 0.orig/

Everything is read from the case itself (caseInfo.txt, 0.orig/, log.checkMesh),
so the figures always describe the case that is actually on disk.
"""

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

from case_info import read_case_info, FILE_NAME as CASE_INFO  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Wedge  # noqa: E402

HERE = Path(__file__).resolve().parent
MM = 1e3
PATCH_COLOURS = {"bottom": "#d62728", "sideWall": "#ff7f0e", "top": "#1f77b4",
                 "front": "#7f7f7f", "back": "#7f7f7f", "axis": "#2ca02c"}
PS_GREY = "#c8c8c8"
WATER = "#9ec5e8"
GAS = "#f4f4f4"


# --------------------------------------------------------------------------- #
# Reading the case
# --------------------------------------------------------------------------- #
def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def parse_block(tokens, i):
    """Parse '{ ... }' starting at tokens[i] == '{' -> (dict, next index)."""
    out, i = {}, i + 1
    while tokens[i] != "}":
        key = tokens[i]
        i += 1
        if tokens[i] == "{":
            out[key], i = parse_block(tokens, i)
            continue
        words, depth = [], 0
        while not (tokens[i] == ";" and depth == 0):
            if tokens[i] == "{":                     # inline dictionary value
                sub, i = parse_block(tokens, i)
                words.append(sub)
                continue
            depth += tokens[i] == "("
            depth -= tokens[i] == ")"
            words.append(tokens[i])
            i += 1
        i += 1
        out[key] = words[0] if len(words) == 1 else words
    return out, i + 1


def boundary_field(path):
    text = strip_comments(path.read_text())
    text = text[text.index("boundaryField") + len("boundaryField"):]
    tokens = re.findall(r"[{}();]|[^\s{}();]+", text)
    return parse_block(tokens, tokens.index("{"))[0]


def flat(value):
    """Words of a parsed entry as one string (inline dictionaries skipped)."""
    if isinstance(value, list):
        return " ".join(w for w in value if isinstance(w, str))
    return str(value)


def read_case(case):
    info = read_case_info(case)
    zero = case / "0.orig"
    fields = {p.name: boundary_field(p) for p in sorted(zero.iterdir()) if p.is_file()}
    check = (case / "log.checkMesh").read_text() if (case / "log.checkMesh").is_file() else ""
    return info, fields, check


def layer(bc):
    """Wall layer of an externalWallHeatFluxTemperature patch -> (thickness, kappa)."""
    if "thicknessLayers" not in bc:
        return None
    nums = lambda v: [float(x) for x in re.findall(r"[-+0-9.eE]+", flat(v))]
    return nums(bc["thicknessLayers"])[0], nums(bc["kappaLayers"])[0]


def number(value):
    return float(re.findall(r"[-+]?[0-9.]+(?:[eE][-+]?\d+)?", flat(value))[-1])


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
class Geo:
    def __init__(self, info):
        self.rb, self.rt, self.h = info["bottom_radius"], info["top_radius"], info["well_height"]
        self.fill = info["fill_height"]
        self.nr, self.nz = info["n_radial"], info["n_axial"]
        self.angle = info["wedge_angle"]

    def r(self, z):
        return self.rb + (self.rt - self.rb) * z / self.h

    def volume(self, z0, z1):
        r0, r1 = self.r(z0), self.r(z1)
        return math.pi * (z1 - z0) / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)


def dim_arrow(ax, p0, p1, text, offset=(0, 0), **kw):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
    mid = (0.5 * (p0[0] + p1[0]) + offset[0], 0.5 * (p0[1] + p1[1]) + offset[1])
    ax.text(mid[0], mid[1], text, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"),
            fontsize=9, bbox=dict(fc="w", ec="none", pad=1.0), **kw)


def draw_fluid(ax, g, mirror=True):
    """Water and gas regions of the cross-section, in mm."""
    signs = (-1, 1) if mirror else (1,)
    zf, h = g.fill * MM, g.h * MM
    for s in signs:
        ax.add_patch(Polygon([(0, 0), (s * g.rb * MM, 0), (s * g.r(g.fill) * MM, zf), (0, zf)],
                             fc=WATER, ec="none"))
        ax.add_patch(Polygon([(0, zf), (s * g.r(g.fill) * MM, zf), (s * g.rt * MM, h), (0, h)],
                             fc=GAS, ec="none"))


# --------------------------------------------------------------------------- #
# 1. Geometry
# --------------------------------------------------------------------------- #
def plot_geometry(info, fields, out):
    g = Geo(info)
    T = fields["T"]
    tb = layer(T["bottom"])
    tsw = layer(T["sideWall"])
    t_bot = tb[0] * MM if tb else 0.0
    t_side = tsw[0] * MM if tsw else 0.0
    rb, rt, h, zf = g.rb * MM, g.rt * MM, g.h * MM, g.fill * MM

    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_axes([0.05, 0.07, 0.50, 0.84])
    draw_fluid(ax, g)
    for s in (-1, 1):                                   # polystyrene walls (not meshed)
        ax.add_patch(Polygon([(s * rb, 0), (s * rt, h), (s * (rt + t_side), h),
                              (s * (rb + t_side), -t_bot), (s * rb, -t_bot)],
                             fc=PS_GREY, ec="k", lw=0.6, hatch="///"))
    ax.add_patch(Polygon([(-rb, 0), (rb, 0), (rb, -t_bot), (-rb, -t_bot)],
                         fc=PS_GREY, ec="k", lw=0.6, hatch="///"))
    ax.plot([-rb, -rt], [0, h], "k", lw=1.4)
    ax.plot([rb, rt], [0, h], "k", lw=1.4)
    ax.plot([-rb, rb], [0, 0], "k", lw=1.4)
    ax.plot([-rt - t_side, rt + t_side], [h, h], color=PATCH_COLOURS["top"], lw=3)
    ax.plot([-g.r(g.fill) * MM, g.r(g.fill) * MM], [zf, zf], color="#1f4e9c", lw=1.5)
    ax.plot([0, 0], [-t_bot - 0.6, h + 0.6], "k-.", lw=0.7)

    dim_arrow(ax, (-rb, -t_bot - 0.9), (rb, -t_bot - 0.9), "bottom diameter %.2f mm" % (2 * rb))
    dim_arrow(ax, (-rt, h + 0.9), (rt, h + 0.9), "top diameter %.2f mm" % (2 * rt))
    dim_arrow(ax, (rt + t_side + 1.0, 0), (rt + t_side + 1.0, h), "well height\n%.1f mm" % h,
              offset=(0.9, 0), ha="left")
    dim_arrow(ax, (-rt - t_side - 1.0, 0), (-rt - t_side - 1.0, zf), "water\n%.2f mm" % zf,
              offset=(-0.9, 0), ha="right")
    if t_side:
        ax.annotate("polystyrene side wall %.1f mm" % t_side, xy=(rb + 0.5 * t_side + 0.05, 2.0),
                    xytext=(rt + t_side + 0.9, 2.0), fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    if t_bot:
        ax.annotate("polystyrene bottom %.1f mm" % t_bot, xy=(1.5, -0.5 * t_bot),
                    xytext=(rt + t_side + 0.9, -t_bot), fontsize=9, va="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0, 0.5 * zf, "water", ha="center", va="center", fontsize=12)
    ax.text(0, 0.5 * (zf + h), "air + water vapour", ha="center", va="center", fontsize=12)
    ax.text(rt + t_side + 0.3, h, "sealed top", ha="left", va="center", fontsize=9,
            color=PATCH_COLOURS["top"])
    ax.text(0.15, -t_bot - 0.1, "axis", fontsize=8, rotation=90, va="top")
    ax.set_xlim(-rt - t_side - 4.2, rt + t_side + 7.5)
    ax.set_ylim(-t_bot - 1.8, h + 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Vertical cross-section of one well (to scale)", fontsize=11)

    # top view: the simulated wedge
    ax2 = fig.add_axes([0.60, 0.50, 0.36, 0.40])
    ax2.add_patch(plt.Circle((0, 0), rt, fc="#eeeeee", ec="k", lw=1.0))
    ax2.add_patch(plt.Circle((0, 0), rt + t_side, fc="none", ec="k", lw=0.6, ls=":"))
    ax2.add_patch(Wedge((0, 0), rt, -g.angle / 2, g.angle / 2, fc="#d62728", ec="#d62728"))
    ax2.annotate("simulated wedge\n(opening angle %g deg)" % g.angle, xy=(0.75 * rt, 0),
                 xytext=(0.2 * rt, -0.75 * rt), fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", lw=0.8))
    ax2.set_xlim(-rt - 2, rt + 2)
    ax2.set_ylim(-rt - 2, rt + 2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Top view: the solution is axisymmetric,\nonly a thin wedge is meshed",
                  fontsize=11)

    v_well, v_water = g.volume(0, g.h), g.volume(0, g.fill)
    lines = [
        "Modelled region: the fluid inside the well (water + gas).",
        "The polystyrene walls are not meshed; they enter as a",
        "thermal resistance in the wall boundary conditions.",
        "",
        "bottom radius      %.4f mm" % rb,
        "top radius         %.4f mm" % rt,
        "well height        %.2f mm" % h,
        "wall taper         %.2f deg" % math.degrees(math.atan((g.rt - g.rb) / g.h)),
        "water fill height  %.3f mm (%.0f %% of the height)" % (zf, 100 * g.fill / g.h),
        "",
        "well volume        %.1f uL" % (v_well * 1e9),
        "water volume       %.1f uL" % (v_water * 1e9),
        "gas volume         %.1f uL" % ((v_well - v_water) * 1e9),
        "wedge volume       %.3f uL (%.2f %% of the well)"
        % (v_well * 1e9 * g.angle / 360, 100 * g.angle / 360),
    ]
    fig.text(0.60, 0.44, "\n".join(lines), family="monospace", fontsize=9, va="top")
    fig.suptitle("Heating case - geometry", fontsize=13)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "geometry.png", dpi=150)
    plt.close(fig)
    (out / "geometry.txt").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 2. Mesh
# --------------------------------------------------------------------------- #
def mesh_nodes(g):
    xi = np.linspace(0, 1, g.nr + 1)
    zf = np.linspace(0, g.h, g.nz + 1)
    return np.outer(xi, g.r(zf)) * MM, np.tile(zf, (g.nr + 1, 1)) * MM


def draw_mesh(ax, g, lw=0.4):
    r, z = mesh_nodes(g)
    zf = g.fill * MM
    colour = np.where(0.5 * (z[:-1, :-1] + z[:-1, 1:]) < zf, 0.0, 1.0)
    ax.pcolormesh(r, z, colour, cmap=matplotlib.colors.ListedColormap([WATER, GAS]),
                  edgecolors="k", linewidth=lw, vmin=0, vmax=1)
    ax.plot(r[-1, :], z[-1, :], color=PATCH_COLOURS["sideWall"], lw=2.5)
    ax.plot(r[:, 0], z[:, 0], color=PATCH_COLOURS["bottom"], lw=2.5)
    ax.plot(r[:, -1], z[:, -1], color=PATCH_COLOURS["top"], lw=2.5)
    ax.plot(r[0, :], z[0, :], color=PATCH_COLOURS["axis"], lw=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("r [mm]")


def plot_mesh(info, check, out):
    g = Geo(info)
    rt, h, zf = g.rt * MM, g.h * MM, g.fill * MM
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_axes([0.04, 0.07, 0.24, 0.84])
    draw_mesh(ax, g)
    ax.set_ylabel("height above the bottom [mm]")
    ax.set_title("whole wedge, side view\n%d x %d cells (radial x axial)" % (g.nr, g.nz), fontsize=10)
    zooms = {"A": (rt - 1.1, rt + 0.05, zf - 0.6, zf + 0.6), "B": (rt - 1.1, rt + 0.05, -0.05, 1.1),
             "C": (-0.05, 1.1, zf - 0.6, zf + 0.6)}
    for name, (x0, x1, y0, y1) in zooms.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="m", lw=1.5))
        ax.text(x0 + 0.08, y1 - 0.08, name, color="m", fontsize=11, va="top", weight="bold")

    titles = {"A": "A: water surface at the side wall", "B": "B: bottom corner",
              "C": "C: water surface at the axis"}
    for k, (name, (x0, x1, y0, y1)) in enumerate(zooms.items()):
        axz = fig.add_axes([0.33 + 0.215 * k, 0.56, 0.19, 0.33])
        draw_mesh(axz, g, lw=0.7)
        axz.set_xlim(x0, x1)
        axz.set_ylim(y0, y1)
        axz.set_title(titles[name], fontsize=10)
        if k == 0:
            axz.set_ylabel("z [mm]")

    # top view of the wedge (true angle)
    axt = fig.add_axes([0.33, 0.30, 0.62, 0.16])
    half = math.radians(g.angle / 2)
    xi = np.linspace(0, 1, g.nr + 1) * rt
    for x in xi:
        axt.plot([x * math.cos(half)] * 2, [-x * math.sin(half), x * math.sin(half)], "k", lw=0.5)
    for s in (-1, 1):
        axt.plot([0, rt * math.cos(half)], [0, s * rt * math.sin(half)],
                 color=PATCH_COLOURS["front"], lw=2)
    axt.text(0.45 * rt, 0.27, "front (wedge)", fontsize=8, color="#555555", ha="center")
    axt.text(0.45 * rt, -0.27, "back (wedge)", fontsize=8, color="#555555", ha="center", va="top")
    axt.set_aspect("equal")
    axt.set_xlim(-0.1, rt + 0.1)
    axt.set_ylim(-0.4, 0.4)
    axt.set_xlabel("r [mm]")
    axt.set_yticks([])
    axt.set_title("top view at the rim (true scale): one cell thick, opening angle %g deg" % g.angle,
                  fontsize=10)

    def grab(pattern, default="n/a"):
        m = re.search(pattern, check)
        return m.group(1) if m else default

    dr, dz = rt / g.nr, h / g.nz
    lines = [
        "mesh level            %s (blockMesh, structured)" % info.get("mesh_level", "?"),
        "cells                 %s  (hexahedra %s, prisms at the axis %s)"
        % (grab(r"cells:\s+(\d+)"), grab(r"hexahedra:\s+(\d+)"), grab(r"prisms:\s+(\d+)")),
        "cell size             dr = %.3f mm (at the rim), dz = %.3f mm" % (dr, dz),
        "cells across water    %d axial" % round(g.fill / g.h * g.nz),
        "max aspect ratio      %s" % grab(r"Max aspect ratio = ([\d.eE+-]+)"),
        "max non-orthogonality %s deg" % grab(r"non-orthogonality Max: ([\d.eE+-]+)"),
        "max skewness          %s" % grab(r"Max skewness = ([\d.eE+-]+)"),
        "checkMesh             %s" % ("Mesh OK" if "Mesh OK" in check else "see log.checkMesh"),
        "solved directions     2 (r, z)",
    ]
    fig.text(0.33, 0.21, "\n".join(lines), family="monospace", fontsize=9, va="top")
    handles = [plt.Line2D([], [], color=PATCH_COLOURS[p], lw=3, label=lab) for p, lab in
               [("bottom", "patch bottom"), ("sideWall", "patch sideWall"), ("top", "patch top"),
                ("axis", "axis of symmetry"), ("front", "patches front/back (wedge)")]]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=WATER, ec="k", lw=0.4, label="initially water"),
                plt.Rectangle((0, 0), 1, 1, fc=GAS, ec="k", lw=0.4, label="initially gas")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.97, 0.03), fontsize=9, ncol=2)
    fig.suptitle("Heating case - computational mesh", fontsize=13)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "mesh.png", dpi=150)
    plt.close(fig)
    (out / "mesh_summary.txt").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 3. Boundary conditions
# --------------------------------------------------------------------------- #
def thermal_text(name, bc, info):
    c = 273.15
    if bc.get("type") != "externalWallHeatFluxTemperature":
        return "%s: %s" % (name, bc.get("type"))
    h = number(bc["h"])
    lay = layer(bc)
    ta = bc["Ta"]
    if isinstance(ta, dict) or (isinstance(ta, list) and any(isinstance(w, dict) for w in ta)):
        outside = ("outside temperature: %.0f -> %.0f degC in %.0f s, then constant"
                   % (info["T_ambient"] - c, info["T_heater_end"] - c, info["heat_ramp_time"]))
    else:
        outside = "outside temperature: %.0f degC (ambient)" % (number(ta) - c)
    parts = [outside, "outside heat-transfer coefficient h = %g W/m2K" % h]
    if lay:
        parts.append("wall layer: %.1f mm polystyrene, k = %.2f W/mK" % (lay[0] * MM, lay[1]))
        u = 1.0 / (1.0 / h + lay[0] / lay[1])
    else:
        parts.append("no wall layer in the model")
        u = h
    parts.append("=> overall U = %.1f W/m2K" % u)
    return "\n".join(parts)


def plot_boundary_conditions(info, fields, out):
    g = Geo(info)
    c = 273.15
    rb, rt, h, zf = g.rb * MM, g.rt * MM, g.h * MM, g.fill * MM
    T, U, A = fields["T"], fields["U"], fields["alpha.liquid"]
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_axes([0.25, 0.08, 0.26, 0.80])
    draw_fluid(ax, g, mirror=False)
    ax.plot([0, rb], [0, 0], color=PATCH_COLOURS["bottom"], lw=5, solid_capstyle="butt")
    ax.plot([rb, rt], [0, h], color=PATCH_COLOURS["sideWall"], lw=5, solid_capstyle="butt")
    ax.plot([0, rt], [h, h], color=PATCH_COLOURS["top"], lw=5, solid_capstyle="butt")
    ax.plot([0, 0], [0, h], color=PATCH_COLOURS["axis"], lw=3, ls="-.")
    ax.plot([0, g.r(g.fill) * MM], [zf, zf], color="#1f4e9c", lw=1.5)
    ax.text(0.5 * rb, 0.5 * zf, "water\n%.0f degC" % (info["T_ambient"] - c), ha="center",
            va="center", fontsize=10)
    ax.text(0.5 * rb, 0.5 * (zf + h), "air + vapour\n%.0f degC\nRH %.0f %%\n%.3f kPa"
            % (info["T_ambient"] - c, 100 * info["rh_initial"], info["p_initial"] / 1e3),
            ha="center", va="center", fontsize=10)
    ax.set_xlim(-0.4, rt + 0.4)
    ax.set_ylim(-0.4, h + 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("height above the bottom [mm]")
    ax.set_title("simulated half cross-section\n(initial state written inside)", fontsize=10)

    wall_common = ("velocity: %s (no slip)\npressure: %s\nwater/gas, vapour: %s"
                   % (U["bottom"]["type"], fields["p_rgh"]["bottom"]["type"], A["bottom"]["type"]))
    boxes = [
        ("top", "top  (sealed lid)", (0.5 * rt, h), (0.55, 0.80)),
        ("sideWall", "sideWall", (g.r(0.55 * g.h) * MM, 0.55 * h), (0.55, 0.50)),
        ("bottom", "bottom  (heated)", (0.5 * rb, 0), (0.55, 0.20)),
    ]
    for patch, title, xy, pos in boxes:
        text = "%s\n%s\n%s" % (title, thermal_text(patch, T[patch], info), wall_common)
        ax.annotate(text, xy=xy, xycoords="data", xytext=pos, textcoords="figure fraction",
                    fontsize=8.5, va="center", ha="left",
                    bbox=dict(fc="w", ec=PATCH_COLOURS[patch], lw=2, boxstyle="round,pad=0.5"),
                    arrowprops=dict(arrowstyle="->", color=PATCH_COLOURS[patch], lw=1.5))
    ax.annotate("axis of symmetry\nfront / back: wedge\n(axisymmetric solution,\nno flow around the axis)",
                xy=(0, 0.75 * h), xycoords="data", xytext=(0.03, 0.72),
                textcoords="figure fraction", fontsize=8.5, va="center",
                bbox=dict(fc="w", ec=PATCH_COLOURS["axis"], lw=2, boxstyle="round,pad=0.5"),
                arrowprops=dict(arrowstyle="->", color=PATCH_COLOURS["axis"], lw=1.5))
    fig.text(0.03, 0.50, "Closed domain: no inlet or outlet.\nThe pressure level is fixed at one\n"
             "reference point (%.3f kPa);\nthe pressure rise of the sealed gas\nis evaluated in "
             "post-processing.\n\nGravity acts downwards (-z)." % (info["p_initial"] / 1e3),
             fontsize=8.5, va="top")

    axr = fig.add_axes([0.05, 0.10, 0.15, 0.22])
    t = [0, info["heat_ramp_time"], info["end_time"]]
    axr.plot(t, [info["T_ambient"] - c, info["T_heater_end"] - c, info["T_heater_end"] - c],
             color=PATCH_COLOURS["bottom"], lw=2)
    axr.axhline(info["T_ambient"] - c, color=PATCH_COLOURS["top"], lw=1.5, ls="--")
    axr.set_xlabel("time [s]")
    axr.set_ylabel("outside temperature [degC]")
    axr.set_title("heater (red), ambient (blue)", fontsize=9)
    axr.grid(alpha=0.3)

    fig.suptitle("Heating case - boundary and initial conditions", fontsize=13)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "boundary_conditions.png", dpi=150)
    plt.close(fig)


def bc_table(fields, out):
    patches = list(fields["T"].keys())
    rows = []
    for name, bf in fields.items():
        row = [name]
        for p in patches:
            bc = bf.get(p, {})
            cell = str(bc.get("type", "-")).replace("HeatFluxTemperature", "HeatFlux\nTemperature")
            if "value" in bc and not cell.startswith("externalWall"):
                cell += "\n" + flat(bc["value"]).replace("uniform ", "= ")
            row.append(cell)
        rows.append(row)
    with open(out / "boundary_conditions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field"] + patches)
        w.writerows([[c.replace("\n", " ") for c in r] for r in rows])

    fig, ax = plt.subplots(figsize=(15, 0.62 * len(rows) + 1.6))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=["field"] + patches, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 2.3)
    for j, p in enumerate(patches):
        tab[0, j + 1].set_facecolor(PATCH_COLOURS.get(p, "#dddddd"))
        tab[0, j + 1].set_text_props(color="w", weight="bold")
    tab[0, 0].set_text_props(weight="bold")
    ax.set_title("Boundary-condition type of every field on every patch (from 0.orig/)\n"
                 "thermal details of the walls: see boundary_conditions.png", fontsize=11)
    fig.savefig(out / "boundary_conditions_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=HERE / "case")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    info, fields, check = read_case(args.case)
    plot_geometry(info, fields, args.out / "geometry")
    plot_mesh(info, check, args.out / "mesh")
    plot_boundary_conditions(info, fields, args.out / "boundary_conditions")
    bc_table(fields, args.out / "boundary_conditions")
    print("Set-up figures written to %s/{geometry,mesh,boundary_conditions}" % args.out)


if __name__ == "__main__":
    main()
