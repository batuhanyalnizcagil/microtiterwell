#!/usr/bin/env python3
"""
Animation of the orbital shaking in the laboratory frame, using the simulated
free surface of one case.

    python3 make_shaking_gif.py                      # rpm 1500, t = 4.0 ... 4.4 s
    python3 make_shaking_gif.py --rpm 1500 --t0 2.0 --t1 2.4 --fps 10

Left : top view to scale. The well (rim circle) travels on the 1 mm orbit
       without turning (the black tick keeps its direction). Coloured dots: free
       surface, coloured by height. Blue arrow: acceleration of the well a_w,
       purple arrow: inertial force per unit mass -a_w felt by the liquid.
Right: vertical section in the fixed laboratory plane y = 0; the well walls move
       with the well. Inset arrow: direction of the effective gravity g - a_w
       in this plane.

The solver works in the frame of the well (see constant/fvOptions of the case); here
the surface is shifted by the well position x_w(t) to show it as seen in the lab.
Writes results/rpm_XXXX_orbital_shaking.gif.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

import postprocess_rotational as pr

HERE = Path(__file__).resolve().parent
MM = 1e3
G = 9.81


def shaking_kinematics(t, omega_target, ramp_time):
    """Angular speed, angular acceleration and phase angle of the orbit
    (linear speed ramp 0 -> target in ramp_time, then constant speed)."""
    if t < ramp_time:
        alpha = omega_target / ramp_time
        return alpha * t, alpha, 0.5 * alpha * t * t
    return omega_target, 0.0, 0.5 * omega_target * ramp_time + omega_target * (t - ramp_time)


def well_state(t, rpm, orbit_diameter, ramp_time):
    om_t = rpm * 2 * math.pi / 60
    om, al, th = shaking_kinematics(t, om_t, ramp_time)
    r = 0.5 * orbit_diameter
    pos = np.array([r * math.cos(th), r * math.sin(th)])
    acc = r * np.array([-om * om * math.cos(th) - al * math.sin(th),
                        -om * om * math.sin(th) + al * math.cos(th)])
    return pos, acc, om


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rpm", type=int, default=1500)
    ap.add_argument("--t0", type=float, default=4.0)
    ap.add_argument("--t1", type=float, default=4.4)
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    case = HERE / "cases" / ("rpm_%04d" % args.rpm)
    info = pr.load_case_info(case)
    files = [(t, f) for t, f in pr.surface_files(case) if args.t0 - 1e-9 <= t <= args.t1 + 1e-9]
    if not files:
        raise SystemExit("no free-surface data between %g and %g s in %s" % (args.t0, args.t1, case))
    out = Path(args.out) if args.out else HERE / "results" / ("rpm_%04d_orbital_shaking.gif" % args.rpm)
    out.parent.mkdir(parents=True, exist_ok=True)

    h, rb, rt = info["well_height"], info["bottom_radius"], info["top_radius"]
    d_orb = info["orbit_diameter"]
    r_orb = 0.5 * d_orb
    period = 60.0 / args.rpm
    dt_snap = files[1][0] - files[0][0] if len(files) > 1 else period
    slow = (1.0 / args.fps) / dt_snap

    fig = plt.figure(figsize=(10.5, 5.6))
    ax_top = fig.add_axes([0.05, 0.11, 0.42, 0.76])
    ax_sec = fig.add_axes([0.55, 0.11, 0.33, 0.76])
    cax = fig.add_axes([0.905, 0.2, 0.015, 0.6])
    norm = plt.Normalize(0, 6.0)
    cmap = plt.cm.viridis
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cax, label="surface height z [mm]")
    trail = []

    def update(i):
        t, f = files[i]
        pts, _ = pr.read_surface(f)
        pos, acc, om = well_state(t, args.rpm, d_orb, info["ramp_time"])
        trail.append(pos.copy())

        # ---------------- top view (lab frame) ----------------
        ax = ax_top
        ax.cla()
        ax.add_patch(Circle((0, 0), r_orb * MM, fill=False, ls="--", color="tab:red", lw=1))
        tr = np.array(trail[-12:]) * MM
        ax.plot(tr[:, 0], tr[:, 1], color="tab:red", lw=2, alpha=0.6)
        if len(pts):
            sel = slice(None, None, max(1, len(pts) // 4000))
            ax.scatter(pts[sel, 0] * MM + pos[0] * MM, pts[sel, 1] * MM + pos[1] * MM,
                       c=pts[sel, 2] * MM, cmap=cmap, norm=norm, s=3, lw=0)
            phi = pr.tilt_direction(pts)
            rc = pr.wall_radius(info, pts[:, 2].max())
            ax.plot(pos[0] * MM + rc * MM * math.cos(phi), pos[1] * MM + rc * MM * math.sin(phi),
                    "o", ms=9, mfc="orange", mec="k", mew=1.2, zorder=5)
        ax.add_patch(Circle(pos * MM, rt * MM, fill=False, color="k", lw=2))
        ax.add_patch(Circle(pos * MM, rb * MM, fill=False, color="k", lw=0.6, ls=":"))
        ax.plot([0, pos[0] * MM], [0, pos[1] * MM], color="0.3", lw=1)
        ax.plot(0, 0, "k+", ms=10)
        ax.plot(*(pos * MM), "ko", ms=3)
        # orientation tick: fixed direction (the well does not turn)
        ax.plot([pos[0] * MM + rt * MM, pos[0] * MM + rt * MM + 0.5],
                [pos[1] * MM, pos[1] * MM], "k", lw=3)
        a = np.hypot(*acc)
        if a > 1e-9:
            sc = 2.2 / (r_orb * (args.rpm * 2 * math.pi / 60) ** 2)   # full speed -> 2.2 mm
            ax.annotate("", xy=pos * MM + acc * sc, xytext=pos * MM,
                        arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=2))
            ax.annotate("", xy=pos * MM - acc * sc, xytext=pos * MM,
                        arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=2))
        lim = rt * MM + r_orb * MM + 0.9
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm] (laboratory)")
        ax.set_ylabel("y [mm]")
        ax.set_title("Top view, to scale (red dashed: 1 mm orbit)", fontsize=10)
        ax.text(-lim + 0.2, -lim + 0.25,
                "blue: acceleration of the well\npurple: inertial force on the liquid\n"
                "orange dot: wave crest at the wall", fontsize=7.5, va="bottom")

        # ---------------- vertical section y = 0 (lab frame) ----------------
        ax = ax_sec
        ax.cla()
        xw = pos[0] * MM
        for sign in (-1, 1):
            ax.plot([xw + sign * rb * MM, xw + sign * rt * MM], [0, h * MM], "k", lw=1.8)
        ax.plot([xw - rb * MM, xw + rb * MM], [0, 0], "k", lw=1.8)
        ax.axhline(info["fill_height"] * MM, color="grey", lw=0.8, ls="--")
        ax.axvline(0, color="0.8", lw=0.8)
        # the section plane y = 0 in the lab is y = -y_w in the well frame
        if len(pts):
            shifted = pts.copy()
            shifted[:, 1] += pos[1]
            for s, z in pr.section_profile(shifted, info, 0.0):
                s = np.asarray(s) * MM + xw
                z = np.asarray(z) * MM
                ax.fill_between(s, 0, z, color="#4a90d9", alpha=0.55, lw=0)
                ax.plot(s, z, color="#0b3d91", lw=1.8)
        # effective gravity in the plane y = 0
        gx, gz = -acc[0], -G
        ax.annotate("", xy=(xw + 2.0 * gx / G, 9.6 + 2.0 * gz / G), xytext=(xw, 9.6),
                    arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=2))
        ax.text(xw + 0.25, 9.9, "g - a_w", color="tab:purple", fontsize=8)
        ax.set_xlim(-rt * MM - r_orb * MM - 0.3, rt * MM + r_orb * MM + 0.3)
        ax.set_ylim(-0.2, h * MM + 0.3)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm] (laboratory)")
        ax.set_ylabel("z [mm]")
        ax.set_title("Vertical section, fixed plane y = 0", fontsize=10)

        fig.suptitle("Orbital shaking, %d rpm (target), orbit diameter %.0f mm   |   "
                     "t = %.3f s, %.0f rpm, %.1f orbits shown at %.0fx slow motion"
                     % (args.rpm, d_orb * MM, t, om * 60 / (2 * math.pi),
                        (t - files[0][0]) / period, slow), fontsize=10.5)

    anim = animation.FuncAnimation(fig, update, frames=len(files), interval=1000 / args.fps)
    anim.save(out, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print("written %s (%d frames, %.3f s per frame of simulated time)" % (out, len(files), dt_snap))


if __name__ == "__main__":
    main()
