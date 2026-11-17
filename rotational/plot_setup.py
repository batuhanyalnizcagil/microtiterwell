#!/usr/bin/env python3
"""
Set-up documentation of the orbital-shaking (rotational) cases: geometry, mesh
and boundary conditions as figures and tables.

    python3 plot_setup.py              # uses the first case found in cases/
    python3 plot_setup.py --rpm 1500   # a specific case

All cases share geometry, mesh and boundary conditions; only the shaking speed
differs. Everything is read from the case itself (caseInfo.txt, 0.orig/*,
system/blockMeshDict, constant/*, log.checkMesh if the mesh was built).

Output in results/ :

    geometry/geometry.png, geometry.txt
    mesh/mesh.png, mesh_summary.txt
    boundary_conditions/boundary_conditions.png, boundary_conditions_table.png,
                        boundary_conditions.csv

Needs only numpy + matplotlib. It never runs OpenFOAM.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np

from case_info import read_case_info, FILE_NAME as CASE_INFO  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Circle  # noqa: E402

HERE = Path(__file__).resolve().parent
MM = 1e3
PATCH_COLOURS = {"bottom": "#d62728", "sideWall": "#ff7f0e", "atmosphere": "#1f77b4"}
WATER = "#9ec5e8"
AIR = "#f4f4f4"


# --------------------------------------------------------------------------- #
# Reading OpenFOAM dictionaries
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
            depth += tokens[i] == "("
            depth -= tokens[i] == ")"
            words.append(tokens[i])
            i += 1
        i += 1
        out[key] = " ".join(words)
    return out, i + 1


def parse_dict(path):
    """Whole dictionary file -> nested dict ('$name;' macros are expanded)."""
    text = strip_comments(path.read_text())
    text = re.sub(r"FoamFile\s*\{.*?\}", "", text, count=1, flags=re.S)
    tokens = ["{"] + re.findall(r"[{}();]|[^\s{}();]+", text) + ["}"]
    top = parse_block(tokens, 0)[0]

    def expand(d):
        for key in list(d):
            if isinstance(d[key], dict):
                expand(d[key])
            elif key.startswith("$") and isinstance(top.get(key[1:]), dict):
                del d[key]
                d.update(top[key[1:]])
    expand(top)
    return top


def read_case(case):
    info = read_case_info(case)
    fields = {p.name: parse_dict(p)["boundaryField"] for p in sorted((case / "0.orig").iterdir())}
    check = (case / "log.checkMesh").read_text() if (case / "log.checkMesh").is_file() else ""
    return info, fields, check


def read_blocks(case):
    """Cell counts of the O-grid from blockMeshDict: (n_core, n_ring, n_z) and
    the half-width of the core square at the bottom [m]."""
    text = strip_comments((case / "system" / "blockMeshDict").read_text())
    counts = [tuple(int(v) for v in m) for m in
              re.findall(r"hex\s*\([^)]*\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)", text)]
    verts = re.search(r"vertices\s*\((.*?)\)\s*;", text, re.S).group(1)
    first = [float(v) for v in re.findall(r"\(([^()]*)\)", verts)[0].split()]
    return counts[0][0], counts[1][1], counts[0][2], abs(first[0])


def read_acceleration(case):
    """constant/acceleration.dat -> t, ax, ay [SI]."""
    rows = re.findall(r"\(\s*([-+0-9.eE]+)\s+\(\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+[-+0-9.eE]+\s*\)",
                      (case / "constant" / "acceleration.dat").read_text())
    return np.array(rows, dtype=float).T


def radius(info, z):
    return info["bottom_radius"] + (info["top_radius"] - info["bottom_radius"]) * z / info["well_height"]


def frustum_volume(info, z0, z1):
    r0, r1 = radius(info, z0), radius(info, z1)
    return math.pi * (z1 - z0) / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1)


def dim_arrow(ax, p0, p1, text, offset=(0, 0), **kw):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
    mid = (0.5 * (p0[0] + p1[0]) + offset[0], 0.5 * (p0[1] + p1[1]) + offset[1])
    ax.text(mid[0], mid[1], text, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"),
            fontsize=9, bbox=dict(fc="w", ec="none", pad=1.0), **kw)


def draw_section(ax, info, lw=3):
    """Vertical section: water, air and the three patches in their colours."""
    rb, rt, h, fill = (info[k] * MM for k in ("bottom_radius", "top_radius", "well_height",
                                               "fill_height"))
    rf = radius(info, info["fill_height"]) * MM
    ax.add_patch(Polygon([(-rb, 0), (rb, 0), (rf, fill), (-rf, fill)], fc=WATER, ec="none"))
    ax.add_patch(Polygon([(-rf, fill), (rf, fill), (rt, h), (-rt, h)], fc=AIR, ec="none"))
    ax.plot([-rb, rb], [0, 0], color=PATCH_COLOURS["bottom"], lw=lw, solid_capstyle="butt")
    for s in (-1, 1):
        ax.plot([s * rb, s * rt], [0, h], color=PATCH_COLOURS["sideWall"], lw=lw,
                solid_capstyle="butt")
    ax.plot([-rt, rt], [h, h], color=PATCH_COLOURS["atmosphere"], lw=lw, ls=(0, (4, 2)))
    ax.plot([-rf, rf], [fill, fill], color="#0b3d91", lw=1.2)
    ax.set_aspect("equal")


# --------------------------------------------------------------------------- #
# 1. Geometry
# --------------------------------------------------------------------------- #
def plot_geometry(info, out):
    out.mkdir(parents=True, exist_ok=True)
    rb, rt, h, fill = (info[k] * MM for k in ("bottom_radius", "top_radius", "well_height",
                                               "fill_height"))
    orbit = info["orbit_diameter"] * MM
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw=dict(width_ratios=[1, 1.1]))

    draw_section(ax, info)
    dim_arrow(ax, (-rb, -0.7), (rb, -0.7), "bottom diameter %.2f mm" % (2 * rb))
    dim_arrow(ax, (-rt, h + 0.7), (rt, h + 0.7), "top diameter %.2f mm" % (2 * rt))
    dim_arrow(ax, (rt + 0.9, 0), (rt + 0.9, h), "height %.1f mm" % h, rotation=90)
    dim_arrow(ax, (-rt - 0.9, 0), (-rt - 0.9, fill), "water %.2f mm" % fill, rotation=90)
    ax.text(0, 0.5 * fill, "water", ha="center", va="center", fontsize=11)
    ax.text(0, 0.5 * (fill + h), "air", ha="center", va="center", fontsize=11)
    ax.text(0, h + 0.15, "open to the atmosphere", ha="center", va="bottom", fontsize=9,
            color=PATCH_COLOURS["atmosphere"])
    ax.annotate("gravity", xy=(-1.6, 6.2), xytext=(-1.6, 8.6), ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_xlim(-rt - 2.2, rt + 2.2)
    ax.set_ylim(-1.6, h + 1.6)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Vertical section through the axis")

    # top view with the orbit
    phi = np.linspace(0, 2 * np.pi, 200)
    axt.add_patch(Circle((0, 0), rt, fc=AIR, ec=PATCH_COLOURS["sideWall"], lw=2))
    axt.add_patch(Circle((0, 0), rb, fc="none", ec=PATCH_COLOURS["sideWall"], lw=1, ls=":"))
    axt.plot(0.5 * orbit * np.cos(phi), 0.5 * orbit * np.sin(phi), "k", lw=1.2)
    for a in (0.0, 0.6 * np.pi, 1.3 * np.pi):
        c = 0.5 * orbit * np.array([math.cos(a), math.sin(a)])
        axt.add_patch(Circle(c, rt, fc="none", ec="0.6", lw=0.8, ls="--"))
        axt.plot(*c, "o", color="0.4", ms=3)
    axt.annotate("", xy=(0.5 * orbit * math.cos(1.0), 0.5 * orbit * math.sin(1.0)),
                 xytext=(0.5 * orbit * math.cos(0.6), 0.5 * orbit * math.sin(0.6)),
                 arrowprops=dict(arrowstyle="->", lw=1.5, color="k"))
    dim_arrow(axt, (-0.5 * orbit, -rt - 1.0), (0.5 * orbit, -rt - 1.0),
              "orbit diameter %.1f mm" % orbit, offset=(0, -0.45))
    axt.text(0, rt + 1.25,
             "The well centre travels on the black circle;\nthe well itself does not spin "
             "(dashed: well at three instants)", ha="center", fontsize=9)
    axt.set_xlim(-rt - 1.6, rt + 1.6)
    axt.set_ylim(-rt - 2.2, rt + 2.2)
    axt.set_aspect("equal")
    axt.set_xlabel("x [mm]")
    axt.set_ylabel("y [mm]")
    axt.set_title("Top view: orbital shaking")

    v_well = frustum_volume(info, 0, info["well_height"]) * 1e9
    v_water = frustum_volume(info, 0, info["fill_height"]) * 1e9
    taper = math.degrees(math.atan2(info["top_radius"] - info["bottom_radius"], info["well_height"]))
    lines = [
        "Greiner 96-well plate, one well (polystyrene), 3-D",
        "bottom radius        %.3f mm" % rb,
        "top radius           %.3f mm" % rt,
        "height               %.2f mm" % h,
        "wall taper           %.2f deg" % taper,
        "well volume          %.1f microlitre" % v_well,
        "water fill height    %.2f mm" % fill,
        "water volume         %.1f microlitre (%.0f %% of the well)" % (v_water, 100 * v_water / v_well),
        "orbit diameter       %.1f mm (radius %.2f mm)" % (orbit, 0.5 * orbit),
        "shaking speed        0 -> %d rpm in %.0f s (linear), then constant" % (info["rpm"], info["ramp_time"]),
        "simulated time       %.0f s" % info["end_time"],
        "peak acceleration    %.2f m/s2 = %.2f g at %d rpm"
        % (info["max_centripetal_acceleration"], info["max_centripetal_acceleration"] / 9.81, info["rpm"]),
    ]
    fig.suptitle("Geometry of the orbital-shaking case (%d rpm case shown)" % info["rpm"])
    fig.tight_layout()
    fig.savefig(out / "geometry.png", dpi=150)
    plt.close(fig)
    (out / "geometry.txt").write_text("\n".join(lines) + "\n")
    return lines


# --------------------------------------------------------------------------- #
# 2. Mesh
# --------------------------------------------------------------------------- #
def ogrid_lines(n_core, n_ring, half, r_wall):
    """Line segments (list of (x, y) arrays, in m) of one horizontal layer of
    the O-grid: square core of half-width `half`, four ring blocks out to the
    circle of radius r_wall."""
    lines = []
    u = np.linspace(-half, half, n_core + 1)
    for v in u:                                           # core
        lines.append((np.array([v, v]), np.array([-half, half])))
        lines.append((np.array([-half, half]), np.array([v, v])))
    s = np.linspace(0, 1, n_core + 1)
    for q in range(4):                                    # ring blocks
        a0 = -0.75 * np.pi + q * 0.5 * np.pi
        rot = np.array([[math.cos(q * 0.5 * np.pi), -math.sin(q * 0.5 * np.pi)],
                        [math.sin(q * 0.5 * np.pi), math.cos(q * 0.5 * np.pi)]])
        inner = rot @ np.vstack([-half + 2 * half * s, np.full_like(s, -half)])
        ang = a0 + s * 0.5 * np.pi
        outer = r_wall * np.vstack([np.cos(ang), np.sin(ang)])
        for j in range(1, n_ring + 1):                    # circumferential lines
            w = j / n_ring
            p = (1 - w) * inner + w * outer
            lines.append((p[0], p[1]))
        for i in range(n_core + 1):                       # radial lines
            lines.append((np.array([inner[0, i], outer[0, i]]),
                          np.array([inner[1, i], outer[1, i]])))
    return lines


def plot_mesh(info, case, check, out):
    out.mkdir(parents=True, exist_ok=True)
    n_core, n_ring, n_z, half = read_blocks(case)
    rb, rt, h = info["bottom_radius"], info["top_radius"], info["well_height"]
    scale = half / rb                                     # core half-width / wall radius

    fig = plt.figure(figsize=(16, 8))
    ax_top = fig.add_axes([0.04, 0.09, 0.36, 0.78])
    ax_side = fig.add_axes([0.42, 0.09, 0.19, 0.78])
    ax_zoom = fig.add_axes([0.66, 0.50, 0.30, 0.37])
    ax_txt = fig.add_axes([0.655, 0.02, 0.34, 0.40])
    ax_txt.axis("off")
    decomp = (case / "system" / "decomposeParDict").read_text()
    n_proc = re.search(r"numberOfSubdomains\s+(\d+)", decomp).group(1)

    for ax, lw in ((ax_top, 0.45), (ax_zoom, 0.8)):
        for x, y in ogrid_lines(n_core, n_ring, half, rb):
            ax.plot(x * MM, y * MM, color="0.25", lw=lw)
        ax.add_patch(Circle((0, 0), rb * MM, fc="none", ec=PATCH_COLOURS["sideWall"], lw=2))
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    ax_top.set_title("Horizontal layer at the bottom (O-grid: square core + 4 ring blocks)")
    ax_zoom.set_xlim(0.45 * rb * MM, 1.03 * rb * MM)
    ax_zoom.set_ylim(0.45 * rb * MM, 1.03 * rb * MM)
    ax_zoom.set_title("Close-up at the wall (45 deg corner of the core)")

    # vertical section y = 0: column boundaries at every z
    zf = np.linspace(0, h, n_z + 1)
    frac = np.concatenate([-1 + (1 - scale) * np.linspace(0, 1, n_ring + 1)[:-1],
                           np.linspace(-scale, scale, n_core + 1),
                           scale + (1 - scale) * np.linspace(0, 1, n_ring + 1)[1:]])
    for f in frac:
        ax_side.plot(f * radius(info, zf) * MM, zf * MM, color="0.25", lw=0.4)
    for z in zf:
        ax_side.plot([-radius(info, z) * MM, radius(info, z) * MM], [z * MM, z * MM],
                     color="0.25", lw=0.4)
    ax_side.plot([-rb * MM, rb * MM], [0, 0], color=PATCH_COLOURS["bottom"], lw=2.5)
    for s in (-1, 1):
        ax_side.plot([s * rb * MM, s * rt * MM], [0, h * MM], color=PATCH_COLOURS["sideWall"], lw=2.5)
    ax_side.plot([-rt * MM, rt * MM], [h * MM, h * MM], color=PATCH_COLOURS["atmosphere"], lw=2.5)
    ax_side.axhline(info["fill_height"] * MM, color="#0b3d91", lw=1.2, ls="--")
    ax_side.text(0, info["fill_height"] * MM + 0.12, "initial water surface", color="#0b3d91",
                 ha="center", fontsize=8, bbox=dict(fc="w", ec="none", pad=0.5))
    ax_side.set_aspect("equal")
    ax_side.set_xlabel("x [mm]")
    ax_side.set_ylabel("z [mm]")
    ax_side.set_title("Vertical section y = 0")

    def grab(pattern, default="n/a"):
        m = re.search(pattern, check)
        return m.group(1) if m else default

    per_layer = n_core * n_core + 4 * n_core * n_ring
    dz = h / n_z
    lines = [
        "mesh level            %s" % info["mesh_level"],
        "type                  3-D block-structured O-grid, hexahedra only",
        "core block            %d x %d cells per layer" % (n_core, n_core),
        "ring blocks           4 x (%d x %d) cells per layer" % (n_core, n_ring),
        "cells per layer       %d" % per_layer,
        "layers (z)            %d, thickness %.3f mm (uniform)" % (n_z, dz * MM),
        "total cells           %d" % (per_layer * n_z),
        "smallest cell size    %.3f mm" % (info["min_cell_size"] * MM),
        "cells across water    %.0f layers over %.2f mm" % (info["fill_height"] / dz, info["fill_height"] * MM),
        "capillary time step   %.2e s (limits the time step)" % info["capillary_dt"],
        "parallel run          %s sub-domains (scotch)" % n_proc,
        "",
        "checkMesh:",
        "  max aspect ratio    %s" % grab(r"Max aspect ratio = ([-+0-9.e]+)"),
        "  max non-orthog.     %s deg (average %s)" % (grab(r"non-orthogonality Max: ([-+0-9.e]+)"),
                                                       grab(r"non-orthogonality Max: [-+0-9.e]+ average: ([-+0-9.e]+)")),
        "  max skewness        %s" % grab(r"Max skewness = ([-+0-9.e]+)"),
        "  result              %s" % ("Mesh OK" if "Mesh OK" in check else "not available"),
    ]
    ax_txt.text(0, 1, "\n".join(lines), family="monospace", fontsize=9, va="top")
    handles = [plt.Line2D([], [], color=c, lw=3, label="patch " + n) for n, c in PATCH_COLOURS.items()]
    ax_zoom.legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle("Mesh of the orbital-shaking case (identical for all speeds)")
    fig.savefig(out / "mesh.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (out / "mesh_summary.txt").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# 3. Boundary conditions
# --------------------------------------------------------------------------- #
def describe(field, bc):
    kind = bc.get("type", "?")
    if kind == "dynamicAlphaContactAngle":
        swing = float(bc["thetaA"]) - float(bc["thetaR"])
        t0 = float(bc["theta0"])
        return ("dynamic contact angle: %g deg receding,\n%g deg advancing "
                "(velocity scale %s m/s)" % (t0 + swing, t0 - swing, bc["uTheta"]))
    text = {
        "noSlip": "no slip (zero velocity relative to the well)",
        "fixedFluxPressure": "gradient balances gravity + shaking",
        "pressureInletOutletVelocity": "free in/outflow, set by the pressure",
        "totalPressure": "atmospheric (total pressure 0 Pa gauge)",
        "inletOutlet": "outflow free; inflow is air",
    }.get(kind, kind)
    return text


def plot_boundary_conditions(info, case, fields, out):
    out.mkdir(parents=True, exist_ok=True)
    rb, rt, h, fill = (info[k] * MM for k in ("bottom_radius", "top_radius", "well_height",
                                               "fill_height"))
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.9, 1.0], hspace=0.5, wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    draw_section(ax, info, lw=5)
    ax.text(0, 0.5 * fill, "water\nat rest", ha="center", va="center", fontsize=10)
    ax.text(0, 0.62 * h, "air at rest\n0 Pa gauge", ha="center", va="center", fontsize=10)

    names = [("U", "velocity"), ("p_rgh", "pressure"), ("alpha.water", "water fraction")]
    boxes = [("atmosphere", "atmosphere  (open top)", (0.4 * rt, h), (rt + 1.2, h + 0.3)),
             ("sideWall", "sideWall  (polystyrene wall)", (radius(info, 0.6 * info["well_height"]) * MM, 0.6 * h),
              (rt + 1.2, 0.56 * h)),
             ("bottom", "bottom  (polystyrene wall)", (0.4 * rb, 0), (rt + 1.2, 0.12 * h))]
    for patch, title, xy, xytext in boxes:
        txt = title + "\n" + "\n".join("%-14s %s" % (label + ":", describe(f, fields[f][patch]).replace(
            "\n", "\n" + " " * 15)) for f, label in names)
        ax.annotate(txt, xy=xy, xytext=xytext, fontsize=8.2, family="monospace", va="center",
                    bbox=dict(fc="w", ec=PATCH_COLOURS[patch], lw=1.5),
                    arrowprops=dict(arrowstyle="-", color=PATCH_COLOURS[patch], lw=1.2))
    ax.text(-rt - 0.3, -1.2,
            "Body force in every cell: gravity (9.81 m/s2 down) minus the acceleration of the well\n"
            "(orbital shaking, horizontal). The mesh does not move: the case is solved in the\n"
            "frame of reference that travels with the well.", fontsize=8.5, va="top")
    ax.set_xlim(-rt - 0.6, rt + 15.5)
    ax.set_ylim(-3.0, h + 1.5)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Boundary and initial conditions")

    t, ax_, ay_ = read_acceleration(case)
    amag = np.hypot(ax_, ay_)
    rpm = info["rpm"] * np.clip(t / info["ramp_time"], 0, 1)

    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(t, rpm, color="tab:red")
    a1.set_ylabel("shaking speed [rpm]")
    a1.set_title("Shaking: 0 -> %d rpm in %.0f s, then constant" % (info["rpm"], info["ramp_time"]))

    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(t, amag, color="k", lw=1.2, label="magnitude")
    a2.plot(t, ax_, color="tab:blue", lw=0.3, alpha=0.7, label="x component")
    a2.axhline(9.81, color="grey", ls=":", lw=0.8)
    a2.text(t[-1], 9.81, "gravity ", fontsize=8, va="top", ha="right", color="grey")
    a2.set_ylabel("well acceleration [m/s2]")
    a2.set_xlabel("time [s]")
    a2.legend(fontsize=8, loc="upper left")

    a3 = fig.add_subplot(gs[2, 1])
    last = t >= t[-1] - 60.0 / info["rpm"]                # one revolution at full speed
    a3.plot(ax_[last], ay_[last], color="k")
    a3.plot(ax_[last][0], ay_[last][0], "o", color="tab:red", ms=5)
    a3.set_aspect("equal")
    a3.set_xlabel("a_x [m/s2]")
    a3.set_ylabel("a_y [m/s2]")
    a3.set_title("Acceleration vector over one revolution at full speed", fontsize=10)
    for a in (a1, a2, a3):
        a.grid(alpha=0.3)

    fig.suptitle("Boundary conditions and forcing of the orbital-shaking case (%d rpm)" % info["rpm"])
    fig.savefig(out / "boundary_conditions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def bc_table(fields, out):
    patches = ["bottom", "sideWall", "atmosphere"]
    rows = []
    for f in sorted(fields):
        row = [f]
        for p in patches:
            bc = fields[f][p]
            extra = ", ".join("%s %s" % (k, v) for k, v in bc.items() if k not in ("type", "value"))
            row.append(bc.get("type", "?") + ("\n" + extra if extra else ""))
        rows.append(row)
    with open(out / "boundary_conditions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["field"] + patches)
        for r in rows:
            w.writerow([c.replace("\n", "; ") for c in r])

    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.9 * len(rows)))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=["field"] + patches, loc="center", cellLoc="left",
                   colWidths=[0.12, 0.29, 0.29, 0.30])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 3.2)
    for j, p in enumerate(patches):
        tab[0, j + 1].set_facecolor(PATCH_COLOURS[p])
        tab[0, j + 1].set_text_props(color="w", weight="bold")
    ax.set_title("Boundary conditions as written in 0.orig/ (OpenFOAM types and parameters)")
    fig.savefig(out / "boundary_conditions_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpm", type=int, help="case to document (default: first one found)")
    ap.add_argument("--cases-dir", type=Path, default=HERE / "cases")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    cases = sorted(d for d in args.cases_dir.glob("rpm_*") if (d / CASE_INFO).is_file())
    if args.rpm:
        cases = [c for c in cases if int(c.name.split("_")[1]) == args.rpm]
    if not cases:
        sys.exit("No case found in %s" % args.cases_dir)
    # prefer a case whose mesh has been built (checkMesh log available)
    case = next((c for c in cases if (c / "log.checkMesh").is_file()), cases[0])

    info, fields, check = read_case(case)
    lines = plot_geometry(info, args.out / "geometry")
    plot_mesh(info, case, check, args.out / "mesh")
    plot_boundary_conditions(info, case, fields, args.out / "boundary_conditions")
    bc_table(fields, args.out / "boundary_conditions")
    print("Case documented: %s" % case)
    print("\n".join(lines))
    print("\nWritten to %s/{geometry,mesh,boundary_conditions}/" % args.out)


if __name__ == "__main__":
    main()
