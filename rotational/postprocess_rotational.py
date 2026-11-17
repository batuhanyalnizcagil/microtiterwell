#!/usr/bin/env python3
"""
Post-processing of the orbital-shaking (rotational) cases - one script for all RPMs.

    python3 postprocess_rotational.py                 # every case in cases/
    python3 postprocess_rotational.py --rpm 900 1500  # selected cases
    python3 postprocess_rotational.py --animate       # also write GIF animations

Needs only numpy + matplotlib. It reads what the solver's function objects wrote:

    postProcessing/freeSurface/<time>/gradAlpha_interface.raw
        free surface (alpha.water = 0.5) as points with the gradient of alpha
    postProcessing/waterVolume/<startTime>/volFieldValue.dat
        water volume inside the well versus time

Results go to  results/ :

    rpm_XXXX_timeseries.png   surface height, water volume, contact angle vs time
    rpm_XXXX_surface.png      3-D views of the free surface at selected times
    rpm_XXXX_section.png      2-D vertical sections through the well axis showing
                              the water surface: in the plane of the largest tilt
                              (crest to trough) and in the fixed plane y = 0
    rpm_XXXX_surface.gif      animation of the free surface (with --animate)
    rpm_XXXX_section.gif      animation of the two sections (with --animate)
    summary_all_rpm.png       comparison of all speeds
    summary_all_rpm.csv       key numbers per speed

Works on finished and on still-running cases (uses whatever has been written).
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from case_info import read_case_info, FILE_NAME as CASE_INFO  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"
RESULTS = HERE / "results"

MM = 1e3      # m -> mm
UL = 1e9      # m3 -> microlitre


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def load_case_info(case):
    return read_case_info(case)


def wall_radius(info, z):
    return info["bottom_radius"] + (info["top_radius"] - info["bottom_radius"]) \
        * z / info["well_height"]


def surface_files(case):
    """Sorted list of (time, file) of the sampled free surface."""
    root = case / "postProcessing" / "freeSurface"
    found = []
    if root.is_dir():
        for d in root.iterdir():
            try:
                t = float(d.name)
            except ValueError:
                continue
            raw = sorted(d.glob("*.raw"))
            if raw:
                found.append((t, raw[0]))
    return sorted(found)


def read_surface(path):
    """Returns points (n,3) and gradient of alpha (n,3); empty arrays if no surface."""
    try:
        data = np.loadtxt(path, comments="#", ndmin=2)
    except (ValueError, OSError):
        data = np.empty((0, 6))
    if data.size == 0 or data.shape[1] < 6:
        return np.empty((0, 3)), np.empty((0, 3))
    return data[:, 0:3], data[:, 3:6]


def read_water_volume(case):
    """Time and integrated alpha.water [m3] (all restarts merged)."""
    root = case / "postProcessing" / "waterVolume"
    rows = []
    if root.is_dir():
        for f in sorted(root.glob("*/volFieldValue*.dat")):
            try:
                d = np.loadtxt(f, comments="#", ndmin=2)
            except ValueError:
                continue
            if d.size:
                rows.append(d[:, :2])
    if not rows:
        return np.empty(0), np.empty(0)
    d = np.vstack(rows)
    d = d[np.argsort(d[:, 0], kind="stable")]
    _, idx = np.unique(d[:, 0], return_index=True)
    return d[idx, 0], d[idx, 1]


# --------------------------------------------------------------------------- #
# Analysis of one free-surface snapshot
# --------------------------------------------------------------------------- #
def analyse_surface(points, grad, info):
    """
    Height statistics and apparent contact angle of one snapshot.

    Contact angle: angle (measured through the water) between the free surface
    and the side wall, evaluated from the interface normal on the surface
    points next to the wall:  cos(theta) = n_water . n_wall,  where n_water is
    the unit normal pointing into the water (direction of grad(alpha)) and
    n_wall the outward wall normal. It is an apparent angle resolved on the
    scale of ~1-2 cells, not a microscopic one.
    """
    res = dict(z_max=np.nan, z_min=np.nan, z_wall_max=np.nan, z_wall_min=np.nan,
               theta_mean=np.nan, theta_min=np.nan, theta_max=np.nan,
               dry_spot=False)
    if len(points) == 0:
        return res

    x, y, z = points.T
    r = np.hypot(x, y)
    res["z_max"], res["z_min"] = z.max(), z.min()
    # the free surface touching the bottom means the bottom is (partly) dry
    res["dry_spot"] = bool(z.min() < 0.5 * info["min_cell_size"])

    band = 1.5 * info["min_cell_size"]
    near = (wall_radius(info, z) - r < band) & (z > band)
    if near.sum() < 3:
        return res
    res["z_wall_max"], res["z_wall_min"] = z[near].max(), z[near].min()

    g = grad[near]
    gmag = np.linalg.norm(g, axis=1)
    ok = gmag > 1e-12
    if ok.sum() < 3:
        return res
    n_water = g[ok] / gmag[ok, None]
    phi = np.arctan2(y[near][ok], x[near][ok])
    taper = math.atan2(info["top_radius"] - info["bottom_radius"], info["well_height"])
    n_wall = np.column_stack([np.cos(phi) * math.cos(taper),
                              np.sin(phi) * math.cos(taper),
                              np.full(phi.shape, -math.sin(taper))])
    theta = np.degrees(np.arccos(np.clip(np.sum(n_water * n_wall, axis=1), -1, 1)))
    res["theta_mean"] = float(theta.mean())
    res["theta_min"], res["theta_max"] = np.percentile(theta, [5, 95])
    return res


def analyse_case(case):
    info = load_case_info(case)
    files = surface_files(case)
    keys = ["z_max", "z_min", "z_wall_max", "z_wall_min",
            "theta_mean", "theta_min", "theta_max", "dry_spot"]
    series = {k: [] for k in keys}
    times = []
    for t, f in files:
        pts, grad = read_surface(f)
        a = analyse_surface(pts, grad, info)
        times.append(t)
        for k in keys:
            series[k].append(a[k])
    out = {k: np.array(v, dtype=float) for k, v in series.items()}
    out["t"] = np.array(times)
    out["t_vol"], out["vol"] = read_water_volume(case)
    out["files"] = files
    out["info"] = info
    return out


def shaking_speed(info, t):
    """Instantaneous shaking speed [rpm] (linear ramp, then constant)."""
    return info["rpm"] * np.clip(np.asarray(t) / info["ramp_time"], 0, 1)


def key_numbers(res):
    """Scalar summary of one case."""
    info, t = res["info"], res["t"]
    out = dict(rpm=info["rpm"], t_last=float(t[-1]) if len(t) else np.nan)
    if len(t) == 0:
        return out
    h = info["well_height"]
    steady = t >= max(info["ramp_time"], t[-1] - 1.0)     # last second at full speed
    if not steady.any():
        steady = t >= t[-1] - 0.25 * (t[-1] - t[0])
    out["max_height_mm"] = float(np.nanmax(res["z_max"]) * MM)
    out["max_height_frac"] = float(np.nanmax(res["z_max"]) / h)
    out["steady_crest_mm"] = float(np.nanmean(res["z_max"][steady]) * MM)
    out["steady_trough_mm"] = float(np.nanmean(res["z_min"][steady]) * MM)
    out["steady_theta_deg"] = float(np.nanmean(res["theta_mean"][steady])) \
        if np.isfinite(res["theta_mean"][steady]).any() else np.nan
    out["bottom_dries"] = bool(np.nansum(res["dry_spot"][steady]) > 0.5 * steady.sum())
    out["reaches_rim"] = bool(np.nanmax(res["z_max"]) >= h - info["min_cell_size"])
    if len(res["vol"]):
        v0 = res["vol"][0]
        out["spilled_uL"] = float((v0 - res["vol"][-1]) * UL)
        out["spilled_percent"] = float(100.0 * (v0 - res["vol"][-1]) / v0)
        lost = np.where(res["vol"] < v0 * (1 - 0.005))[0]      # >0.5 % lost
        out["spill_start_s"] = float(res["t_vol"][lost[0]]) if len(lost) else np.nan
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_timeseries(res, path):
    info, t = res["info"], res["t"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.plot(t, res["z_max"] * MM, label="highest point of the free surface")
    ax.plot(t, res["z_wall_max"] * MM, "--", lw=1, label="highest point of the contact line")
    ax.plot(t, res["z_min"] * MM, label="lowest point of the free surface")
    ax.axhline(info["well_height"] * MM, color="k", ls=":", label="well rim")
    ax.axhline(info["fill_height"] * MM, color="grey", ls=":", label="initial level")
    ax.set_ylabel("height above bottom [mm]")
    ax.set_ylim(0, info["well_height"] * MM * 1.05)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("%d rpm, orbit diameter %.1f mm" % (info["rpm"], info["orbit_diameter"] * MM))
    ax2 = ax.twinx()
    ax2.plot(t, shaking_speed(info, t), color="tab:red", lw=0.8, alpha=0.6)
    ax2.set_ylabel("shaking speed [rpm]", color="tab:red")

    ax = axes[1]
    if len(res["vol"]):
        ax.plot(res["t_vol"], res["vol"] * UL)
        ax.axhline(res["vol"][0] * UL, color="grey", ls=":")
    ax.set_ylabel("water inside the well [uL]")

    ax = axes[2]
    ax.plot(t, res["theta_mean"], label="mean along the contact line")
    ax.fill_between(t, res["theta_min"], res["theta_max"], alpha=0.25,
                    label="5-95 % range along the contact line")
    ca = info["contact_angle"]
    ax.axhline(ca["advancing"], color="k", ls=":", lw=0.8)
    ax.axhline(ca["receding"], color="k", ls=":", lw=0.8)
    ax.set_ylabel("apparent contact angle [deg]")
    ax.set_xlabel("time [s]")
    ax.legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_well(ax, info):
    """Wireframe of the well (bottom and rim circles + a few wall lines)."""
    phi = np.linspace(0, 2 * np.pi, 73)
    for z, r in ((0.0, info["bottom_radius"]), (info["well_height"], info["top_radius"])):
        ax.plot(r * np.cos(phi) * MM, r * np.sin(phi) * MM, z * MM, color="k", lw=0.6)
    for p in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([info["bottom_radius"] * math.cos(p) * MM, info["top_radius"] * math.cos(p) * MM],
                [info["bottom_radius"] * math.sin(p) * MM, info["top_radius"] * math.sin(p) * MM],
                [0, info["well_height"] * MM], color="k", lw=0.3)
    lim = info["top_radius"] * MM * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, info["well_height"] * MM)
    ax.set_box_aspect((1, 1, info["well_height"] / (2 * info["top_radius"])))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")


def draw_surface(ax, points, info):
    """Free surface coloured by height. Uses a triangulated sheet when the
    surface is a single-valued z(x,y); otherwise (overturning / film on the
    wall) a point cloud."""
    if len(points) < 4:
        return None
    x, y, z = (points * MM).T
    vmin, vmax = 0, info["well_height"] * MM
    single_valued = (z.max() - z.min()) < 0.6 * info["well_height"] * MM
    if single_valued:
        try:
            tri = Triangulation(x, y)
            # drop long triangles that would bridge dry patches
            xt, yt = x[tri.triangles], y[tri.triangles]
            edge = np.max(np.hypot(xt - np.roll(xt, 1, axis=1), yt - np.roll(yt, 1, axis=1)), axis=1)
            tri.set_mask(edge > 6 * info["min_cell_size"] * MM)
            return ax.plot_trisurf(tri, z, cmap="viridis", vmin=vmin, vmax=vmax,
                                   linewidth=0, antialiased=True)
        except (RuntimeError, ValueError):
            pass
    return ax.scatter(x, y, z, c=z, cmap="viridis", vmin=vmin, vmax=vmax, s=3)


def plot_snapshots(res, path, n=4):
    info, files = res["info"], res["files"]
    if not files:
        return
    t = res["t"]
    # last frame, frame of maximum height, and evenly spaced ones in between
    picks = {len(files) - 1, int(np.nanargmax(res["z_max"]))}
    for frac in np.linspace(0.25, 0.75, max(n - 2, 0)):
        picks.add(int(frac * (len(files) - 1)))
    picks = sorted(picks)
    fig = plt.figure(figsize=(4.2 * len(picks), 5.5))
    for i, k in enumerate(picks):
        ax = fig.add_subplot(1, len(picks), i + 1, projection="3d")
        pts, _ = read_surface(files[k][1])
        draw_well(ax, info)
        draw_surface(ax, pts, info)
        ax.set_title("t = %.3f s (%.0f rpm)\nsurface %.2f - %.2f mm"
                     % (t[k], shaking_speed(info, t[k]),
                        res["z_min"][k] * MM, res["z_max"][k] * MM), fontsize=9)
        ax.view_init(elev=22, azim=-60)
    fig.suptitle("Free surface, target %d rpm" % info["rpm"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2-D sections
# --------------------------------------------------------------------------- #
def tilt_direction(points):
    """Azimuth [rad] of the uphill direction of the free surface (least-squares
    plane through the surface points); 0 if the surface is flat."""
    if len(points) < 10:
        return 0.0
    x, y, z = points.T
    coef, *_ = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), z, rcond=None)
    if math.hypot(coef[0], coef[1]) < 1e-6:
        return 0.0
    return math.atan2(coef[1], coef[0])


def section_profile(points, info, phi):
    """Water surface in the vertical plane through the axis with azimuth phi.

    Returns a list of (s, z) pieces [m]: s is the signed distance from the axis
    along the plane (positive towards azimuth phi). Surface points within one
    cell of the plane are averaged in bins of one cell; gaps (dry bottom or the
    surface leaving the plane) split the profile into pieces."""
    if len(points) == 0:
        return []
    dx = info["min_cell_size"]
    x, y, z = points.T
    dist = -x * math.sin(phi) + y * math.cos(phi)
    near = np.abs(dist) < dx
    s, zs = (x * math.cos(phi) + y * math.sin(phi))[near], z[near]
    if len(s) < 3:
        return []
    width = 1.5 * dx
    k = np.round(s / width).astype(int)
    bins = np.unique(k)
    sb = np.array([s[k == b].mean() for b in bins])
    zb = np.array([zs[k == b].mean() for b in bins])
    pieces, start = [], 0
    for i in range(1, len(bins) + 1):
        if i == len(bins) or bins[i] - bins[i - 1] > 2:
            if i - start >= 2:
                pieces.append((sb[start:i], zb[start:i]))
            start = i
    return pieces


def draw_section(ax, points, info, phi, title):
    """Well cross-section with the water (blue) below the surface line."""
    h, rb, rt = info["well_height"], info["bottom_radius"], info["top_radius"]
    for sign in (-1, 1):
        ax.plot([sign * rb * MM, sign * rt * MM], [0, h * MM], "k", lw=1.5)
    ax.plot([-rb * MM, rb * MM], [0, 0], "k", lw=1.5)
    ax.plot([-rt * MM, rt * MM], [h * MM, h * MM], "k", lw=0.8, ls=":")
    ax.axhline(info["fill_height"] * MM, color="grey", lw=0.8, ls="--")
    for s, z in section_profile(points, info, phi):
        # extend a piece to the wall if it ends within two cells of it
        s, z = list(s), list(z)
        for end, sign in ((0, -1), (-1, 1)):
            wall = sign * wall_radius(info, z[end])
            if abs(wall - s[end]) < 2.5 * info["min_cell_size"]:
                if end == 0:
                    s.insert(0, wall)
                    z.insert(0, z[0])
                else:
                    s.append(wall)
                    z.append(z[-1])
        s, z = np.array(s) * MM, np.array(z) * MM
        ax.fill_between(s, 0, z, color="#4a90d9", alpha=0.55, lw=0)
        ax.plot(s, z, color="#0b3d91", lw=1.6)
    ax.set_xlim(-rt * MM * 1.08, rt * MM * 1.08)
    ax.set_ylim(-0.2, h * MM * 1.03)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def plot_sections(res, path, n=6):
    info, files, t = res["info"], res["files"], res["t"]
    if not files:
        return
    picks = {len(files) - 1, int(np.nanargmax(res["z_max"]))}
    for frac in np.linspace(0.15, 0.85, max(n - 2, 0)):
        picks.add(int(frac * (len(files) - 1)))
    picks = sorted(picks)
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 8.6), squeeze=False)
    for j, k in enumerate(picks):
        pts, _ = read_surface(files[k][1])
        phi = tilt_direction(pts)
        draw_section(axes[0, j], pts, info, phi,
                     "t = %.3f s (%.0f rpm)\ncrest towards %.0f deg"
                     % (t[k], shaking_speed(info, t[k]), math.degrees(phi) % 360))
        draw_section(axes[1, j], pts, info, 0.0, "plane y = 0")
        axes[1, j].set_xlabel("x [mm]")
        axes[0, j].set_xlabel("along the tilt direction [mm]")
    axes[0, 0].set_ylabel("plane of the largest tilt (rotates with the wave)\nz [mm]")
    axes[1, 0].set_ylabel("fixed plane y = 0\nz [mm]")
    fig.suptitle("Vertical sections through the well axis, target %d rpm "
                 "(dashed: initial level, dotted: rim)" % info["rpm"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_section_animation(res, path, max_frames=200):
    info, files = res["info"], res["files"]
    if not files:
        return
    step = max(1, len(files) // max_frames)
    frames = list(range(0, len(files), step))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 6))

    def update(k):
        pts, _ = read_surface(files[k][1])
        phi = tilt_direction(pts)
        for ax in axes:
            ax.cla()
        draw_section(axes[0], pts, info, phi, "plane of the largest tilt\n(crest towards %.0f deg)"
                     % (math.degrees(phi) % 360))
        draw_section(axes[1], pts, info, 0.0, "fixed plane y = 0\n")
        axes[0].set_xlabel("along the tilt direction [mm]")
        axes[1].set_xlabel("x [mm]")
        axes[0].set_ylabel("z [mm]")
        fig.suptitle("t = %.3f s, %.0f rpm" % (res["t"][k], shaking_speed(info, res["t"][k])))

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=80)
    try:
        anim.save(path, writer=animation.PillowWriter(fps=12))
    except Exception as err:                       # Pillow missing etc.
        print("    animation skipped (%s)" % err)
    plt.close(fig)


def make_animation(res, path, max_frames=200):
    info, files = res["info"], res["files"]
    if not files:
        return
    step = max(1, len(files) // max_frames)
    frames = list(range(0, len(files), step))
    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(k):
        ax.cla()
        pts, _ = read_surface(files[k][1])
        draw_well(ax, info)
        draw_surface(ax, pts, info)
        ax.view_init(elev=22, azim=-60)
        ax.set_title("t = %.3f s, %.0f rpm" % (res["t"][k], shaking_speed(info, res["t"][k])))

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=80)
    try:
        anim.save(path, writer=animation.PillowWriter(fps=12))
    except Exception as err:                       # Pillow missing etc.
        print("    animation skipped (%s)" % err)
    plt.close(fig)


def plot_summary(all_keys, all_res, path):
    done = [k for k in all_keys if "max_height_mm" in k]
    if not done:
        return
    rpm = np.array([k["rpm"] for k in done])
    info = all_res[0]["info"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(rpm, [k["max_height_mm"] for k in done], "o-", label="highest point reached (whole run)")
    ax.plot(rpm, [k["steady_crest_mm"] for k in done], "s--", label="crest, last second (mean)")
    ax.plot(rpm, [k["steady_trough_mm"] for k in done], "v--", label="trough, last second (mean)")
    ax.axhline(info["well_height"] * MM, color="k", ls=":", label="well rim")
    ax.axhline(info["fill_height"] * MM, color="grey", ls=":", label="initial level")
    ax.set_xlabel("shaking speed [rpm]")
    ax.set_ylabel("free-surface height [mm]")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.bar(rpm, [k.get("spilled_percent", np.nan) for k in done], width=0.5 * np.min(np.diff(rpm)) if len(rpm) > 1 else 100)
    ax.set_xlabel("shaking speed [rpm]")
    ax.set_ylabel("water lost over the rim [% of initial volume]")

    ax = axes[1, 0]
    ax.plot(rpm, [k["steady_theta_deg"] for k in done], "o-")
    ca = info["contact_angle"]
    ax.axhline(ca["advancing"], color="k", ls=":", lw=0.8)
    ax.axhline(ca["receding"], color="k", ls=":", lw=0.8)
    ax.set_xlabel("shaking speed [rpm]")
    ax.set_ylabel("mean apparent contact angle, last second [deg]")

    ax = axes[1, 1]
    for res in all_res:
        if len(res["t"]):
            ax.plot(res["t"], res["z_max"] * MM, lw=0.9, label="%d rpm" % res["info"]["rpm"])
    ax.axhline(info["well_height"] * MM, color="k", ls=":")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("highest point of the free surface [mm]")
    ax.legend(fontsize=8, ncol=2)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.suptitle("Orbital shaking of the microtiter well - all speeds")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary_csv(all_keys, path):
    cols = ["rpm", "t_last", "max_height_mm", "max_height_frac", "steady_crest_mm",
            "steady_trough_mm", "steady_theta_deg", "reaches_rim", "bottom_dries",
            "spilled_uL", "spilled_percent", "spill_start_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for k in all_keys:
            w.writerow(k)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpm", type=int, nargs="+", help="only these speeds")
    ap.add_argument("--animate", action="store_true", help="write GIF animations")
    ap.add_argument("--cases-dir", type=Path, default=CASES)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    cases = sorted(d for d in args.cases_dir.glob("rpm_*") if (d / CASE_INFO).is_file())
    if args.rpm:
        cases = [c for c in cases if int(c.name.split("_")[1]) in args.rpm]
    if not cases:
        sys.exit("No cases found in %s" % args.cases_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    all_res, all_keys = [], []
    for case in cases:
        res = analyse_case(case)
        keys = key_numbers(res)
        all_res.append(res)
        all_keys.append(keys)
        if len(res["t"]) == 0:
            print("%s: no results yet" % case.name)
            continue
        print("%s: %d surface snapshots up to t = %.3f s" % (case.name, len(res["t"]), res["t"][-1]))
        print("    highest point %.2f mm (%.0f %% of well height)%s, spilled %.2f %%"
              % (keys["max_height_mm"], 100 * keys["max_height_frac"],
                 " - REACHES THE RIM" if keys["reaches_rim"] else "",
                 keys.get("spilled_percent", float("nan"))))
        plot_timeseries(res, args.out / (case.name + "_timeseries.png"))
        plot_snapshots(res, args.out / (case.name + "_surface.png"))
        plot_sections(res, args.out / (case.name + "_section.png"))
        if args.animate:
            make_animation(res, args.out / (case.name + "_surface.gif"))
            make_section_animation(res, args.out / (case.name + "_section.gif"))

    plot_summary(all_keys, all_res, args.out / "summary_all_rpm.png")
    write_summary_csv(all_keys, args.out / "summary_all_rpm.csv")
    print("\nFigures and summary written to %s" % args.out)


if __name__ == "__main__":
    main()
