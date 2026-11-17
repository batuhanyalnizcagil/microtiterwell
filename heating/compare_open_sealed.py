#!/usr/bin/env python3
"""
Side-by-side comparison of the sealed and the open-top heating runs.

    python3 compare_open_sealed.py
    python3 compare_open_sealed.py --sealed results --open results_open --out results_compare

Reads the heating_timeseries.csv files written by postprocess_heating.py and
writes results_compare/open_vs_sealed.png and open_vs_sealed.csv.

Evaporated mass compared:
  sealed : vapour added to the gas as simulated (evaporated_mass), and the
           post-processing estimate with wall condensation (evaporated_mass_cond,
           GasSpaceModel, an upper estimate - not simulation output)
  open   : vapour budget = vapour stored in the gas + vapour that left through
           the mouth (evaporated_mass), the solver's interface source dmdt
           (evaporated_mass_solver) and the loss of liquid volume
           (evaporated_mass_from_liquid_volume)
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
K0 = 273.15
UG = 1e9


def load(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sealed", default="results")
    ap.add_argument("--open", default="results_open")
    ap.add_argument("--out", default="results_compare")
    args = ap.parse_args()
    s = load(HERE / args.sealed / "heating_timeseries.csv")
    o = load(HERE / args.open / "heating_timeseries.csv")
    out = HERE / args.out
    out.mkdir(exist_ok=True)
    m_water = o["liquid_volume"][0] * 998.2   # initial water mass [kg]

    cs, co = "tab:red", "tab:blue"
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    ts, to = s["time_s"], o["time_s"]

    a = ax[0, 0]
    a.plot(ts, s["T_heater"] - K0, "k:", lw=1, label="heater")
    a.plot(ts, s["T_water_mean"] - K0, color=cs, label="water mean, sealed")
    a.plot(to, o["T_water_mean"] - K0, color=co, label="water mean, open")
    a.plot(ts, s["T_gas_mean"] - K0, color=cs, ls="--", label="gas mean, sealed")
    a.plot(to, o["T_gas_mean"] - K0, color=co, ls="--", label="gas mean, open")
    a.set_ylabel("temperature [degC]")
    a.set_title("Temperatures")
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(ts, s["p_vapour"] / 1e3, color=cs, label="sealed")
    a.plot(to, o["p_vapour"] / 1e3, color=co, label="open")
    a.set_ylabel("mean vapour partial pressure [kPa]")
    a.set_title("Vapour pressure in the gas")
    a.legend(fontsize=8)

    a = ax[0, 2]
    a.plot(ts, 100 * s["rh_mean"], color=cs, label="mean, sealed")
    a.plot(to, 100 * o["rh_mean"], color=co, label="mean, open")
    a.plot(ts, 100 * s["rh_max"], color=cs, ls=":", label="max, sealed")
    a.plot(to, 100 * o["rh_max"], color=co, ls=":", label="max, open")
    a.axhline(100, color="0.5", lw=0.8)
    a.set_ylabel("relative humidity [%]")
    a.set_title("Humidity of the gas")
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(ts, s["evaporated_mass"] * UG, color=cs, label="sealed, simulated")
    a.plot(ts, s["evaporated_mass_cond"] * UG, color=cs, ls="--",
           label="sealed + wall condensation (model, upper est.)")
    a.plot(to, o["evaporated_mass"] * UG, color=co, label="open, vapour budget")
    a.plot(to, o["evaporated_mass_solver"] * UG, color=co, ls="--", label="open, solver dmdt")
    a.plot(to, o["evaporated_mass_from_liquid_volume"] * UG, color=co, ls=":",
           label="open, liquid-volume loss")
    a.set_ylabel("evaporated water [ug]")
    a.set_title("Evaporated water")
    a.legend(fontsize=7.5)
    sec = a.secondary_yaxis("right", functions=(lambda m: 100 * m / (m_water * UG),
                                                 lambda p: p / 100 * m_water * UG))
    sec.set_ylabel("% of the water")

    a = ax[1, 1]
    a.plot(ts, s["p_total"] / 1e3, color=cs, label="sealed (ideal gas, post-processing)")
    a.plot(to, o["p_total"] / 1e3, color=co, label="open (ambient)")
    a.set_ylabel("gas pressure [kPa]")
    a.set_title("Pressure")
    a.legend(fontsize=8)

    a = ax[1, 2]
    a.plot(ts, s["U_max"] * 1e3, color=cs, label="sealed")
    a.plot(to, o["U_max"] * 1e3, color=co, label="open")
    a.set_ylabel("max velocity [mm/s]")
    a.set_title("Natural convection")
    a.legend(fontsize=8)

    for a in ax.flat:
        a.set_xlabel("time [s]")
        a.grid(alpha=0.3)
    fig.suptitle("Heating 20 -> 45 degC: sealed vs open-top well (same mesh and settings)")
    fig.tight_layout()
    fig.savefig(out / "open_vs_sealed.png", dpi=130)

    keys = ["T_water_mean", "T_gas_mean", "p_total", "p_vapour", "rh_mean", "rh_max", "U_max"]
    common = np.intersect1d(ts, to)
    with open(out / "open_vs_sealed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + ["%s_%s" % (k, c) for k in keys for c in ("sealed", "open")]
                   + ["evaporated_mass_sealed", "evaporated_mass_sealed_cond",
                      "evaporated_mass_open", "evaporated_mass_open_solver",
                      "evaporated_mass_open_from_liquid_volume"])
        for t in common:
            i, j = np.where(ts == t)[0][0], np.where(to == t)[0][0]
            w.writerow([t] + [v for k in keys for v in (s[k][i], o[k][j])]
                       + [s["evaporated_mass"][i], s["evaporated_mass_cond"][i],
                          o["evaporated_mass"][j], o["evaporated_mass_solver"][j],
                          o["evaporated_mass_from_liquid_volume"][j]])
    print("written", out / "open_vs_sealed.png", "and", out / "open_vs_sealed.csv")
    print("initial water mass %.1f mg" % (m_water * 1e6))


if __name__ == "__main__":
    main()
