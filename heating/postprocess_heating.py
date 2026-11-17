#!/usr/bin/env python3
"""
Post-processing of the heating/evaporation case of the microtiter well
(sealed or open top; the kind is read from caseInfo.txt).

    python3 postprocess_heating.py                # reads case/, writes results/
    python3 postprocess_heating.py --case case_open     # -> results_open/
    python3 postprocess_heating.py --times 60 240 450   # times of the maps [s]
    python3 postprocess_heating.py --no-gif       # static figures only (faster)

Needs only numpy + matplotlib. It reads the ASCII fields written by the solver
in the time directories of the case (T, alpha.liquid, vapour.gas, U) and the
mesh layout stored in caseInfo.txt. Works on finished and on still-running
cases (uses the time directories present).

Output in results/ :

    heating_timeseries.png   temperatures, pressure, evaporation, humidity vs time
    heating_maps.png         cross-sections: temperature, liquid/gas/vapour, humidity
    heating_profiles.png     temperature and vapour content along the well axis
    heating_timeseries.csv   all time series
    heating_temperature.gif  animated temperature field, axis profile, time series
    heating_vapour.gif       animated water-vapour content of the gas, evaporated mass
    heating_evaporation.gif  animated relative humidity, driving force at the water
                             surface (saturation vs. actual vapour pressure), rate
    heating_flow.gif         animated natural-convection velocity field
    heating_3d.png           whole well in 3-D (axisymmetric result revolved, one
                             quarter cut away): temperature and water/vapour
    heating_3d.gif           the same, animated

    heating_condensation.png wall-condensation estimate (see below)

About the pressure: icoReactingMultiphaseInterFoam is an incompressible
(low-Mach) solver, so the pressure rise of the sealed gas volume is not part
of its solution. It is evaluated here from the ideal-gas law for the closed
gas space: the amount of dry air is fixed at its initial value, the local
temperature and vapour mole fraction come from the simulation, and the
pressure is taken as uniform in the gas (it is, to within a few Pa).

Corrections for known artefacts of the solver:

  * Gas volume. In a sealed domain the solver removes the volume of the
    vapour it creates from the liquid (about 1.3 microlitre per microgram),
    so its gas region grows by ~2 %. All gas inventories (dry air, vapour) are
    therefore scaled back to the geometric gas volume of t = 0. Without this
    the pressure is ~1.2 kPa too low.
  * Liquid volume and level are derived from the evaporated mass; the values
    of the solver's alpha field are kept in the *_solver columns.
  * Condensation. The solver exchanges mass only at the water surface, so the
    gas becomes supersaturated (RH up to ~180 %) where it is colder than the
    water. What would condense on the lid and on the side wall is estimated
    by a quasi-steady model of the gas space (class GasSpaceModel): coupled
    heat conduction and vapour diffusion between the simulated water surface
    and the walls, the walls being saturated where vapour arrives, with the
    latent heat released there. It uses the simulated water-surface
    temperature; the extra evaporative cooling of the water is not fed back.
    Columns/curves ending in _cond belong to this estimate.

Open-top cases (case_open): the pressure is the room pressure and
the gas-volume correction and the condensation estimate are not applied (the
artefact does not occur with an open top). The evaporated mass is the vapour
stored in the gas plus the vapour that left through the mouth (diffusion to
the room air + outflow), integrated over the written times; the loss of
liquid volume times the water density is given as an independent check.
"""

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np

from case_info import read_case_info, FILE_NAME as CASE_INFO  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

warnings.filterwarnings("ignore", message="No contour levels were found")

HERE = Path(__file__).resolve().parent
R_GAS = 8314.47     # J/kmol/K
MM = 1e3


# --------------------------------------------------------------------------- #
# Mesh (structured wedge, reconstructed from caseInfo.txt)
# --------------------------------------------------------------------------- #
class Mesh:
    """Full-revolution geometry of the axisymmetric mesh; arrays are [ir, iz]."""

    def __init__(self, info):
        self.nr, self.nz = info["n_radial"], info["n_axial"]
        self.h = info["well_height"]
        rb, rt = info["bottom_radius"], info["top_radius"]
        self.wall_radius = lambda z: rb + (rt - rb) * z / self.h
        self.axial_first = info.get("cell_order", "axial_first") == "axial_first"

        xi = np.linspace(0.0, 1.0, self.nr + 1)          # fraction of wall radius
        zf = np.linspace(0.0, self.h, self.nz + 1)
        rf = self.wall_radius(zf)
        self.z_nodes = np.tile(zf, (self.nr + 1, 1))
        self.r_nodes = np.outer(xi, rf)
        self.zc = 0.5 * (zf[1:] + zf[:-1])
        self.rc = np.outer(0.5 * (xi[1:] + xi[:-1]), self.wall_radius(self.zc))
        # exact volume of each ring-shaped cell (whole well, not only the wedge)
        ring = np.pi * (xi[1:] ** 2 - xi[:-1] ** 2)
        slab = (zf[1:] - zf[:-1]) / 3.0 * (rf[:-1] ** 2 + rf[:-1] * rf[1:] + rf[1:] ** 2)
        self.vol = np.outer(ring, slab)
        self.dz = zf[1:] - zf[:-1]
        self.xi, self.zf = xi, zf

    @property
    def n_cells(self):
        return self.nr * self.nz

    def to_grid(self, flat):
        """Cell list in OpenFOAM order -> array [ir, iz] (works for (n,3) too)."""
        shape = flat.shape[1:]
        if self.axial_first:
            return flat.reshape((self.nr, self.nz) + shape)
        return np.swapaxes(flat.reshape((self.nz, self.nr) + shape), 0, 1)


# --------------------------------------------------------------------------- #
# OpenFOAM ASCII field reader
# --------------------------------------------------------------------------- #
def read_field(path, n_cells):
    """internalField of an ASCII vol field -> array (n,) or (n,3)."""
    text = path.read_text()
    start = text.index("internalField")
    end = text.find("boundaryField", start)
    body = text[start + len("internalField"):end if end > 0 else None]
    m = re.match(r"\s*uniform\s+(.*?);", body, re.S)
    if m:
        vals = np.array(m.group(1).replace("(", " ").replace(")", " ").split(), dtype=float)
        return np.tile(vals, (n_cells, 1)).squeeze()
    m = re.match(r"\s*nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", body)
    if not m:
        raise ValueError("cannot parse %s (binary file? writeFormat must be ascii)" % path)
    kind, n = m.group(1), int(m.group(2))
    data = body[m.end():body.rindex(")")]
    vals = np.array(data.replace("(", " ").replace(")", " ").split(), dtype=float)
    if n != n_cells:
        raise ValueError("%s has %d cells, expected %d (mesh changed?)" % (path, n, n_cells))
    return vals.reshape(n, 3) if kind == "vector" else vals


def time_dirs(case):
    out = []
    for d in case.iterdir():
        if d.is_dir() and (d / "T").is_file():
            try:
                out.append((float(d.name), d))
            except ValueError:
                pass
    return sorted(out)


# --------------------------------------------------------------------------- #
# Physics helpers
# --------------------------------------------------------------------------- #
def p_sat(T, antoine):
    a, b, c = antoine
    return np.exp(a + b / (c + T))


def is_open(info):
    return info.get("top", "sealed") == "open"


def top_name(info):
    return "open top" if is_open(info) else "sealed top"


def heater_temperature(info, t):
    t = np.asarray(t, dtype=float)
    frac = np.clip(t / info["heat_ramp_time"], 0, 1)
    return info["T_ambient"] + frac * (info["T_heater_end"] - info["T_ambient"])


def load_state(mesh, info, d):
    """Fields of one time directory as [ir, iz] arrays plus derived fields."""
    n = mesh.n_cells
    s = dict(T=mesh.to_grid(read_field(d / "T", n)),
             al=np.clip(mesh.to_grid(read_field(d / "alpha.liquid", n)), 0, 1),
             Y=np.clip(mesh.to_grid(read_field(d / "vapour.gas", n)), 0, 1))
    s["ag"] = 1.0 - s["al"]
    wv, wa = info["W_water"], info["W_air"]
    s["X"] = (s["Y"] / wv) / (s["Y"] / wv + (1 - s["Y"]) / wa)     # vapour mole fraction
    if (d / "U").is_file():
        s["U"] = mesh.to_grid(read_field(d / "U", n))          # (ur, -, uz)
    else:
        s["U"] = np.zeros(s["T"].shape + (3,))
    s["Umag"] = np.linalg.norm(s["U"], axis=-1)
    # interfacial mass-transfer rate of the solver [kg/m3/s] (not written at t = 0)
    f = d / "dmdt.liquidToGas"
    s["dmdt"] = mesh.to_grid(read_field(f, n)) if f.is_file() else None
    f = d / "p"
    s["p_field"] = mesh.to_grid(read_field(f, n)) if f.is_file() else None
    return s


class GasSpaceModel:
    """Quasi-steady heat conduction and vapour diffusion in the gas space with
    condensation on the lid and on the side wall.

    The gas is treated as a stagnant conducting/diffusing medium (Rayleigh
    number ~200, diffusion times ~1 s against minutes of heating). Unknowns are
    the temperature and the vapour partial pressure in the gas cells above the
    initial fill height. Boundary conditions:

      water surface : simulated temperature of the uppermost water cells,
                      vapour saturated at that temperature
      lid, side wall: heat loss to the ambient through the wall resistance;
                      where the adjacent gas holds more vapour than saturation
                      at the wall temperature, the wall is saturated and the
                      latent heat of the condensing vapour is released there
    """

    def __init__(self, mesh, info):
        self.info = info
        self.k0 = int(round(info["fill_height"] / mesh.h * mesh.nz))   # first gas row
        nr, ng = mesh.nr, mesh.nz - self.k0
        self.nr, self.ng, self.n = nr, ng, nr * ng
        xi = mesh.xi
        xc = 0.5 * (xi[1:] + xi[:-1])
        ring = np.pi * (xi[1:] ** 2 - xi[:-1] ** 2)
        zc, zf, dz = mesh.zc[self.k0:], mesh.zf[self.k0:], mesh.dz[self.k0:]
        rw_c, rw_f = mesh.wall_radius(zc), mesh.wall_radius(zf)
        self.vol = mesh.vol[:, self.k0:]
        self.zc, self.r_lid, self.z_side = zc, xc * rw_f[-1], zc

        # geometric conductances area/distance [m]
        g_r = 2 * np.pi * np.outer(xi[1:-1] / (xc[1:] - xc[:-1]), dz)           # (nr-1, ng)
        g_z = np.outer(ring, rw_f[1:-1] ** 2 / (zc[1:] - zc[:-1]))              # (nr, ng-1)
        self.a_surf, self.a_lid = ring * rw_f[0] ** 2, ring * rw_f[-1] ** 2
        self.a_side = 2 * np.pi * rw_c * dz
        self.g_surf = self.a_surf / (0.5 * dz[0])
        self.g_lid = self.a_lid / (0.5 * dz[-1])
        self.g_side = self.a_side / ((1 - xc[-1]) * rw_c)

        idx = np.arange(self.n).reshape(nr, ng)
        lap = np.zeros((self.n, self.n))
        for a, b, g in ((idx[:-1, :], idx[1:, :], g_r), (idx[:, :-1], idx[:, 1:], g_z)):
            a, b, g = a.ravel(), b.ravel(), g.ravel()
            np.add.at(lap, (a, b), g)
            np.add.at(lap, (b, a), g)
            np.add.at(lap, (a, a), -g)
            np.add.at(lap, (b, b), -g)
        self.lap = lap
        self.c_surf, self.c_lid, self.c_side = idx[:, 0], idx[:, -1], idx[-1, :]
        self.c_wall = np.concatenate([self.c_lid, self.c_side])
        self.g_wall = np.concatenate([self.g_lid, self.g_side])
        self.a_wall = np.concatenate([self.a_lid, self.a_side])

        self.k_gas = info.get("k_gas", 0.0262)                     # W/m/K
        h = info["h_ambient"]
        u_lid = 1.0 / (1.0 / h + info.get("lid_thickness", 0.0) / info["ps_conductivity"])
        u_side = 1.0 / (1.0 / h + info["side_thickness"] / info["ps_conductivity"])
        self.u_wall = np.concatenate([np.full(nr, u_lid), np.full(ng, u_side)])
        self._inverse = {}

    def _inverse_for(self, g_wall):
        """Inverse of the Laplace operator with Dirichlet links at the water
        surface (always) and at the walls (conductances g_wall, 0 = no link)."""
        key = g_wall.tobytes()
        if key not in self._inverse:
            if len(self._inverse) >= 12:
                self._inverse.clear()
            m = -self.lap.copy()
            np.add.at(m, (self.c_surf, self.c_surf), self.g_surf)
            np.add.at(m, (self.c_wall, self.c_wall), g_wall)
            self._inverse[key] = np.linalg.inv(m)
        return self._inverse[key]

    def solve(self, t_surface, condensation=True):
        """t_surface: temperature of the water surface per radial column [K].

        Newton iteration on the wall-face temperatures; the gas fields follow
        from them linearly. The set of condensing faces is updated outside."""
        info = self.info
        ant, ta = info["antoine"], info["T_ambient"]
        latent, d_vap, wv = info["latent_heat"], info["D_vapour"], info["W_water"]
        cw, gw, n_wall = self.c_wall, self.g_wall, self.c_wall.size
        kq = self.k_gas * gw / self.a_wall                         # gas side [W/m2/K]
        pv_surface = p_sat(t_surface, ant)

        inv_t = self._inverse_for(gw)
        s_t = inv_t[np.ix_(cw, cw)] * gw                           # d T_cell / d T_wall
        t_base = inv_t[np.ix_(cw, self.c_surf)] @ (self.g_surf * t_surface)

        t_wall = np.full(n_wall, ta)
        active = np.full(n_wall, bool(condensation))
        eye = np.eye(n_wall)
        for _ in range(12):                                        # active-set loop
            g_act = np.where(active, gw, 0.0)
            inv_v = self._inverse_for(g_act)
            s_v = inv_v[np.ix_(cw, cw)] * g_act
            v_base = inv_v[np.ix_(cw, self.c_surf)] @ (self.g_surf * pv_surface)
            for _ in range(60):
                ps = p_sat(t_wall, ant)
                dps = ps * (-ant[1] / (ant[2] + t_wall) ** 2)
                tc, pc = t_base + s_t @ t_wall, v_base + s_v @ ps
                lb = np.where(active, latent * d_vap * wv / (R_GAS * t_wall) * gw / self.a_wall, 0.0)
                res = kq * (tc - t_wall) + lb * (pc - ps) - self.u_wall * (t_wall - ta)
                jac = (kq[:, None] * (s_t - eye) + lb[:, None] * ((s_v - eye) * dps[None, :])
                       - np.diag(self.u_wall))
                step = np.linalg.solve(jac, -res)
                t_wall = t_wall + step
                if np.abs(step).max() < 1e-7:
                    break
            if not condensation:
                break
            ps = p_sat(t_wall, ant)
            pc = v_base + s_v @ ps
            # condensing faces must receive vapour; dry faces must not be
            # supersaturated
            wrong = np.where(active, pc < ps, pc > ps * (1 + 1e-9))
            if not wrong.any():
                break
            active = active ^ wrong

        rhs = np.zeros(self.n)
        np.add.at(rhs, self.c_surf, self.g_surf * t_surface)
        np.add.at(rhs, cw, gw * t_wall)
        T = inv_t @ rhs
        rhs = np.zeros(self.n)
        np.add.at(rhs, self.c_surf, self.g_surf * pv_surface)
        np.add.at(rhs, cw, g_act * p_sat(t_wall, ant))
        pv = inv_v @ rhs
        beta = d_vap * wv / (R_GAS * t_wall) * gw / self.a_wall
        flux = np.where(active, beta * (pv[cw] - p_sat(t_wall, ant)), 0.0)   # kg/m2/s

        T = T.reshape(self.nr, self.ng)
        pv = pv.reshape(self.nr, self.ng)
        rate = flux * self.a_wall                                  # kg/s per wall face
        beta_s = d_vap * wv / (R_GAS * t_surface) * self.g_surf
        return dict(T=T, pv=pv, rh=pv / p_sat(T, ant),
                    t_lid=t_wall[:self.nr], t_side=t_wall[self.nr:],
                    flux_lid=flux[:self.nr], flux_side=flux[self.nr:],
                    rate_lid=rate[:self.nr].sum(), rate_side=rate[self.nr:].sum(),
                    rate_evap=np.sum(beta_s * (pv_surface - pv[:, 0])),
                    vapour_mass=np.sum(self.vol * pv * wv / (R_GAS * T)))


def analyse(mesh, info, states, times, raw=False):
    """Time series (dict of arrays) and per-state humidity/pressure fields.

    raw=True: no post-processing corrections or models - gas inventories in the
    solver's own gas volume, pressure = mean of the solver's p field, liquid
    volume and level as the solver has them, no wall-condensation model."""
    V = mesh.vol
    wv, wa = info["W_water"], info["W_air"]
    p0 = info["p_initial"]
    keys = ["T_heater", "T_water_mean", "T_water_min", "T_water_max", "T_gas_mean",
            "T_bottom_fluid", "T_lid_gas", "p_total", "p_vapour", "p_air",
            "vapour_mass", "evaporated_mass", "rh_mean", "rh_max",
            "supersat_volume_frac", "liquid_volume", "level", "U_max",
            "liquid_volume_solver", "gas_volume_solver", "level_solver_axis",
            "level_solver_wall", "p_total_uncorrected",
            "T_lid_wall_cond", "evaporation_rate_cond", "condensation_rate_lid",
            "condensation_rate_side", "condensed_mass_lid", "condensed_mass_side",
            "vapour_mass_cond", "evaporated_mass_cond", "p_total_cond", "p_vapour_cond",
            "rh_mean_cond", "rh_max_cond", "liquid_volume_cond", "model_T_gas_deviation",
            "mouth_loss_rate", "mouth_loss_mass", "evaporated_mass_from_liquid_volume",
            "evaporation_rate_solver", "evaporated_mass_solver"]
    ts = {k: np.full(len(times), np.nan) for k in keys}
    model = GasSpaceModel(mesh, info)
    k0 = model.k0
    open_top = is_open(info)
    if open_top:
        # room air at the mouth plane (boundary values of the case)
        x_amb = info["rh_ambient"] * p_sat(info["T_ambient"], info["antoine"]) / p0
        y_amb = x_amb * wv / (x_amb * wv + (1 - x_amb) * wa)
        rho_amb = p0 / (y_amb / wv + (1 - y_amb) / wa) / (R_GAS * info["T_ambient"])
        a_top = np.pi * (mesh.xi[1:] ** 2 - mesh.xi[:-1] ** 2) * mesh.wall_radius(mesh.h) ** 2

    air_moles = air_ref_raw = gas_volume_0 = None
    for i, (t, s) in enumerate(zip(times, states)):
        T, al, ag, X, Y = s["T"], s["al"], s["ag"], s["X"], s["Y"]
        wl, wg = al * V, ag * V
        liquid, gas = al > 0.5, al < 0.5

        ts["T_heater"][i] = heater_temperature(info, t)
        ts["T_water_mean"][i] = np.sum(wl * T) / np.sum(wl)
        ts["T_water_min"][i], ts["T_water_max"][i] = T[liquid].min(), T[liquid].max()
        ts["T_gas_mean"][i] = np.sum(wg * T) / np.sum(wg)
        ts["T_bottom_fluid"][i] = np.sum(V[:, 0] * T[:, 0]) / np.sum(V[:, 0])
        ts["T_lid_gas"][i] = np.sum(V[:, -1] * T[:, -1]) / np.sum(V[:, -1])

        # The solver's gas region grows artificially (see module docstring):
        # scale all gas inventories back to the geometric gas volume.
        if gas_volume_0 is None:
            gas_volume_0 = np.sum(wg)
        f = 1.0 if (open_top or raw) else gas_volume_0 / np.sum(wg)

        # Vapour: the solver carries the vapour density X*p0/(R*T), so the
        # local vapour partial pressure is X*p0.
        # Dry air: its amount in the sealed gas space is conserved,
        #   n_air*R = p_air * sum(V_gas/T) = const   (V_gas: geometric volume)
        # Total pressure (uniform) = dry-air + mean vapour partial pressure.
        s["p"] = p0
        s["rh"] = X * p0 / p_sat(T, info["antoine"])
        inv_temp = f * np.sum(wg / T)
        pv_mean = p0 * np.sum(wg * X / T) / np.sum(wg / T)    # mole-weighted
        air_raw = np.sum(wg * (1 - X) / T)
        if air_moles is None:
            air_moles, air_ref_raw = (p0 - pv_mean) * inv_temp, air_raw
        p = p0 if open_top else air_moles / inv_temp + pv_mean
        if raw:
            # what the (incompressible) solver itself has: no pressure rise
            p = np.sum(wg * s["p_field"]) / np.sum(wg) if s["p_field"] is not None else p0
        ts["p_total"][i], ts["p_vapour"][i] = p, pv_mean
        ts["p_air"][i] = p - pv_mean
        ts["p_total_uncorrected"][i] = p0 * air_ref_raw / air_raw   # version-1 value

        # vapour inventory as transported by the solver (gas density at the
        # reference pressure)
        w_mix = 1.0 / (Y / wv + (1 - Y) / wa)
        rho_gas = p0 * w_mix / (R_GAS * T)
        ts["vapour_mass"][i] = f * np.sum(wg * rho_gas * Y)

        ts["rh_mean"][i] = np.sum(wg * s["rh"]) / np.sum(wg)
        ts["rh_max"][i] = s["rh"][gas].max()
        ts["supersat_volume_frac"][i] = np.sum(wg[s["rh"] > 1.0]) / np.sum(wg)
        ts["liquid_volume_solver"][i] = np.sum(wl)
        ts["gas_volume_solver"][i] = np.sum(wg)
        ts["level_solver_axis"][i] = np.sum(al[0, :] * mesh.dz)
        ts["level_solver_wall"][i] = np.sum(al[-1, :] * mesh.dz)
        ts["U_max"][i] = s["Umag"].max()
        if s.get("dmdt") is not None:
            ts["evaporation_rate_solver"][i] = np.sum(s["dmdt"] * mesh.vol)

        if open_top:
            # vapour leaving through the mouth: diffusion to the room air over
            # half a cell + flow through the mouth carrying the boundary value
            diff = rho_amb * info["D_vapour"] * (Y[:, -1] - y_amb) / (0.5 * mesh.dz[-1])
            conv = rho_amb * s["U"][:, -1, 2] * y_amb
            ts["mouth_loss_rate"][i] = np.sum(a_top * (diff + conv))
            continue
        if raw:
            continue

        # ---- wall-condensation estimate ----------------------------------- #
        t_surface = T[:, k0 - 1]
        c = model.solve(t_surface)
        s["cond"] = c
        ts["T_lid_wall_cond"][i] = np.sum(model.a_lid * c["t_lid"]) / model.a_lid.sum()
        ts["evaporation_rate_cond"][i] = c["rate_evap"]
        ts["condensation_rate_lid"][i] = c["rate_lid"]
        ts["condensation_rate_side"][i] = c["rate_side"]
        ts["vapour_mass_cond"][i] = c["vapour_mass"]
        inv_t = np.sum(model.vol / c["T"])
        if i == 0:
            air_cond = ts["p_air"][0] * inv_t                  # dry air is conserved
        if ts["rh_max"][i] < 0.999:
            # still under-saturated everywhere: the transient of the simulation
            # applies, the quasi-steady model does not
            for key in ("evaporation_rate_cond", "condensation_rate_lid", "condensation_rate_side"):
                ts[key][i] = 0.0
            ts["vapour_mass_cond"][i] = ts["vapour_mass"][i]
            ts["p_total_cond"][i], ts["p_vapour_cond"][i] = p, pv_mean
            ts["rh_mean_cond"][i], ts["rh_max_cond"][i] = ts["rh_mean"][i], ts["rh_max"][i]
        else:
            pv_mean = np.sum(model.vol * c["pv"] / c["T"]) / inv_t
            ts["p_total_cond"][i] = air_cond / inv_t + pv_mean
            ts["p_vapour_cond"][i] = pv_mean
            ts["rh_mean_cond"][i] = np.sum(model.vol * c["rh"]) / model.vol.sum()
            ts["rh_max_cond"][i] = c["rh"].max()
        # how well does pure conduction reproduce the simulated gas temperature?
        ref = model.solve(t_surface, condensation=False)["T"]
        ts["model_T_gas_deviation"][i] = np.sqrt(np.mean((ref - T[:, k0:]) ** 2))

    rho_l = info["rho_water"]
    area = np.pi * mesh.wall_radius(info["fill_height"]) ** 2
    ts["evaporated_mass"] = ts["vapour_mass"] - ts["vapour_mass"][0]
    ts["evaporated_mass_from_liquid_volume"] = rho_l * (ts["liquid_volume_solver"][0]
                                                        - ts["liquid_volume_solver"])
    if open_top:
        r = ts["mouth_loss_rate"]
        ts["mouth_loss_mass"] = np.concatenate(
            [[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(times))])
        ts["evaporated_mass"] = ts["evaporated_mass"] + ts["mouth_loss_mass"]
    ts["liquid_volume"] = ts["liquid_volume_solver"][0] - ts["evaporated_mass"] / rho_l
    ts["level"] = info["fill_height"] - ts["evaporated_mass"] / rho_l / area
    if raw:
        ts["liquid_volume"] = ts["liquid_volume_solver"].copy()
        ts["level"] = ts["level_solver_axis"].copy()
    for key in ("lid", "side"):
        r = ts["condensation_rate_" + key]
        ts["condensed_mass_" + key] = np.concatenate(
            [[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(times))])
    ts["evaporated_mass_cond"] = (ts["vapour_mass_cond"] - ts["vapour_mass"][0]
                                  + ts["condensed_mass_lid"] + ts["condensed_mass_side"])
    ts["liquid_volume_cond"] = ts["liquid_volume_solver"][0] - ts["evaporated_mass_cond"] / rho_l
    if raw:
        for k in [k for k in ts if k.endswith("_cond") or "condens" in k
                  or k in ("model_T_gas_deviation", "p_total_uncorrected")]:
            del ts[k]
    # evaporation according to the solver's own interface source term; the rate
    # at t = 0 is not written and is taken equal to the first written value
    r = ts["evaporation_rate_solver"]
    ok = np.isfinite(r)
    if ok.any():
        r = np.interp(times, times[ok], r[ok])
        ts["evaporation_rate_solver"] = r
        ts["evaporated_mass_solver"] = np.concatenate(
            [[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(times))])
    ts["gas_model"] = model
    return ts


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_timeseries(times, ts, info, path, raw=False):
    c = 273.15
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

    ax = axes[0, 0]
    ax.plot(times, ts["T_heater"] - c, "k--", label="heater (outside of the bottom wall)")
    ax.plot(times, ts["T_bottom_fluid"] - c, label="water at the bottom wall")
    ax.plot(times, ts["T_water_mean"] - c, label="water, mean")
    ax.fill_between(times, ts["T_water_min"] - c, ts["T_water_max"] - c, alpha=0.2,
                    label="water, min-max")
    ax.plot(times, ts["T_gas_mean"] - c, label="gas, mean")
    ax.plot(times, ts["T_lid_gas"] - c, label="gas at the %s" % top_name(info))
    ax.set_ylabel("temperature [degC]")
    ax.legend(fontsize=8)
    ax.set_title("Temperatures")

    if is_open(info):
        _open_top_panels(axes, times, ts)
        for ax in axes.flat:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    if raw:
        _raw_sealed_panels(axes, times, ts)
        for ax in axes.flat:
            ax.grid(alpha=0.3)
        fig.suptitle("Raw simulation data (no post-processing corrections)", fontsize=11)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    ax = axes[0, 1]
    ax.plot(times, ts["p_total"] / 1e3, label="total, as simulated (no wall condensation)")
    ax.plot(times, ts["p_total_cond"] / 1e3, "--", label="total, with wall condensation (estimate)")
    ax.plot(times, ts["p_air"] / 1e3, color="0.45", label="dry air (partial pressure)")
    ax.set_ylabel("pressure [kPa]")
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(times, ts["p_vapour"] / 1e3, color="tab:green")
    ax2.plot(times, ts["p_vapour_cond"] / 1e3, "--", color="tab:green")
    ax2.set_ylabel("water vapour partial pressure [kPa]\n(solid: as simulated, dashed: estimate)",
                   color="tab:green")
    ax.set_title("Pressure in the sealed well (ideal gas, fixed gas volume)")

    ax = axes[1, 0]
    um = 1e9
    ax.semilogy(times[1:], ts["evaporated_mass"][1:] * um,
                label="evaporated, as simulated (no wall condensation)")
    ax.semilogy(times[1:], ts["evaporated_mass_cond"][1:] * um, "--",
                label="evaporated, with wall condensation (estimate)")
    for key, name in (("side", "side wall"), ("lid", "lid")):
        m = ts["condensed_mass_" + key] * um
        ax.semilogy(times[m > 0], m[m > 0], ":", label="condensed on the %s (estimate)" % name)
    ax.set_ylim(0.5, None)
    ax.set_ylabel("mass [microgram]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Evaporation and condensation (cumulative)")

    ax = axes[1, 1]
    ax.plot(times, 100 * ts["rh_mean"], label="gas mean, as simulated")
    ax.plot(times, 100 * ts["rh_max"], label="gas maximum, as simulated")
    ax.plot(times, 100 * ts["rh_max_cond"], "--", label="gas maximum, with wall condensation (estimate)")
    ax.axhline(100, color="k", ls=":", lw=0.8)
    ax.set_ylabel("relative humidity [%]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="lower right")
    ax2 = ax.twinx()
    ax2.plot(times, 100 * ts["supersat_volume_frac"], color="tab:purple", lw=0.8)
    ax2.set_ylabel("gas volume above 100 % RH, as simulated [%]", color="tab:purple")
    ax.set_title("Humidity of the gas")

    for ax in axes.flat:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _raw_sealed_panels(axes, times, ts):
    """Sealed case, raw data: only quantities the solver itself carries."""
    um = 1e9
    ax = axes[0, 1]
    ax.plot(times, ts["p_total"] - 101325.0,
            label="mean pressure p of the solver in the gas, minus 101325 Pa")
    ax.set_ylabel("p - 101325 Pa [Pa]")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(times, ts["p_vapour"] / 1e3, color="tab:green")
    ax2.set_ylabel("water vapour partial pressure X p0 [kPa]", color="tab:green")
    ax.set_title("Pressure (incompressible solver: no pressure rise is computed)", fontsize=10)

    ax = axes[1, 0]
    ax.plot(times, (ts["vapour_mass"] - ts["vapour_mass"][0]) * um,
            label="vapour added to the gas (solver gas volume)")
    ax.plot(times, ts["evaporated_mass_solver"] * um, "--",
            label="solver interface source dmdt, integrated")
    ax.set_ylabel("mass [microgram]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="upper left")
    ax3 = ax.twinx()
    ax3.plot(times, ts["liquid_volume_solver"] * um, color="tab:purple")
    ax3.set_ylabel("liquid volume of the solver [uL]", color="tab:purple")
    ax.set_title("Evaporation and liquid volume")

    ax = axes[1, 1]
    ax.plot(times, 100 * ts["rh_mean"], label="gas mean")
    ax.plot(times, 100 * ts["rh_max"], label="gas maximum")
    ax.axhline(100, color="k", ls=":", lw=0.8)
    ax.set_ylabel("relative humidity [%]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Humidity of the gas (X p0 / p_sat(T))")


def _open_top_panels(axes, times, ts):
    """Pressure/evaporation/humidity panels of the time-series figure, open top."""
    um = 1e9
    rate = np.gradient(ts["evaporated_mass"], times) if len(times) > 1 else np.zeros_like(times)

    ax = axes[0, 1]
    ax.plot(times, ts["evaporation_rate_solver"] * um, color="tab:red",
            label="evaporation, solver interface source (dmdt)")
    ax.plot(times, rate * um, label="evaporation, vapour budget of the gas")
    ax.plot(times, ts["mouth_loss_rate"] * um, "--", label="vapour leaving through the open top")
    ax.set_ylabel("rate [microgram/s]")
    if len(times) > 3:
        hi = np.nanmax([rate[2:].max(), ts["mouth_loss_rate"].max(),
                        np.nanmax(ts["evaporation_rate_solver"])])
        ax.set_ylim(min(0, rate[2:].min() * um), 1.25 * um * hi)
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(times, ts["p_vapour"] / 1e3, color="tab:green", lw=0.9)
    ax2.set_ylabel("mean vapour partial pressure in the well [kPa]", color="tab:green")
    ax.set_title("Mass-transfer rates (pressure = room pressure)")

    ax = axes[1, 0]
    ax.plot(times, ts["evaporated_mass_solver"] * um, color="tab:red",
            label="evaporated (solver interface source dmdt)")
    ax.plot(times, ts["evaporated_mass"] * um, label="evaporated (vapour budget)")
    ax.plot(times, ts["evaporated_mass_from_liquid_volume"] * um, ":",
            label="evaporated (loss of liquid volume, check)")
    ax.plot(times, ts["mouth_loss_mass"] * um, "--", label="left through the open top")
    ax.plot(times, (ts["vapour_mass"] - ts["vapour_mass"][0]) * um, "-.",
            label="stored in the gas of the well")
    ax.set_ylabel("mass [microgram]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Evaporation (cumulative)")

    ax = axes[1, 1]
    ax.plot(times, 100 * ts["rh_mean"], label="gas mean")
    ax.plot(times, 100 * ts["rh_max"], label="gas maximum")
    ax.axhline(100, color="k", ls=":", lw=0.8)
    ax.set_ylabel("relative humidity [%]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Humidity of the gas in the well")


def mirrored(ax, mesh, field, **kw):
    """Full cross-section (r < 0 mirrored) of an axisymmetric field."""
    r, z = mesh.r_nodes * MM, mesh.z_nodes * MM
    ax.pcolormesh(-r, z, field, shading="flat", **kw)
    return ax.pcolormesh(r, z, field, shading="flat", **kw)


def decorate(ax, mesh, s):
    zw = np.array([0.0, mesh.h])
    for sign in (-1, 1):
        ax.plot(sign * mesh.wall_radius(zw) * MM, zw * MM, "k", lw=1.2)
        ax.contour(sign * mesh.rc * MM, np.tile(mesh.zc * MM, (mesh.nr, 1)), s["al"],
                   levels=[0.5], colors="w", linewidths=1.2)
    r0, r1 = mesh.wall_radius(0.0) * MM, mesh.wall_radius(mesh.h) * MM
    ax.plot([-r0, r0], [0, 0], "k", lw=1.2)
    ax.plot([-r1, r1], [mesh.h * MM] * 2, "k", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])


def plot_maps(times, states, mesh, info, picks, path):
    c = 273.15
    n = len(picks)
    fig, axes = plt.subplots(3, n, figsize=(2.3 * n + 2.2, 13), squeeze=False)
    t_lo = min(s["T"].min() for s in states) - c
    t_hi = max(s["T"].max() for s in states) - c
    x_hi = 100 * max(np.max(np.where(s["al"] < 0.5, s["X"], 0)) for s in states)
    rh_hi = max(120.0, 100 * max(np.max(np.where(s["al"] < 0.5, s["rh"], 0)) for s in states))
    water = ListedColormap(["#1f4e9c"])

    for j, k in enumerate(picks):
        s = states[k]
        gas_only = np.ma.masked_where(s["al"] >= 0.5, np.ones_like(s["al"]))
        liquid_only = np.ma.masked_where(s["al"] < 0.5, np.ones_like(s["al"]))

        ax = axes[0, j]
        im_t = mirrored(ax, mesh, s["T"] - c, cmap="inferno", vmin=t_lo, vmax=max(t_hi, t_lo + 0.1))
        decorate(ax, mesh, s)
        ax.set_title("t = %g s" % times[k])

        ax = axes[1, j]
        mirrored(ax, mesh, liquid_only, cmap=water)
        im_x = mirrored(ax, mesh, 100 * s["X"] * gas_only, cmap="YlGn", vmin=0, vmax=max(x_hi, 0.1))
        decorate(ax, mesh, s)

        ax = axes[2, j]
        mirrored(ax, mesh, liquid_only, cmap=water)
        im_rh = mirrored(ax, mesh, 100 * s["rh"] * gas_only, cmap="coolwarm_r", vmin=0, vmax=rh_hi)
        ax.contour(mesh.rc * MM, np.tile(mesh.zc * MM, (mesh.nr, 1)),
                   np.where(s["al"] < 0.5, s["rh"], 0), levels=[1.0], colors="k", linewidths=0.8)
        ax.contour(-mesh.rc * MM, np.tile(mesh.zc * MM, (mesh.nr, 1)),
                   np.where(s["al"] < 0.5, s["rh"], 0), levels=[1.0], colors="k", linewidths=0.8)
        decorate(ax, mesh, s)

        if j:
            for row in range(3):
                axes[row, j].set_yticks([])
    for row, label in enumerate(["temperature", "liquid water (blue) / gas", "liquid water (blue) / gas"]):
        axes[row, 0].set_ylabel("%s\nz [mm]" % label)

    fig.colorbar(im_t, ax=axes[0, :], label="temperature [degC]", fraction=0.03, pad=0.02)
    fig.colorbar(im_x, ax=axes[1, :], label="water vapour in the gas [mol %]", fraction=0.03, pad=0.02)
    fig.colorbar(im_rh, ax=axes[2, :],
                 label="relative humidity of the gas [%]\n(black line: 100 %, condensation beyond)",
                 fraction=0.03, pad=0.02)
    fig.suptitle("%s well heated from below - cross-sections (white line: water surface)"
                 % ("Open" if is_open(info) else "Sealed"))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_condensation(times, states, ts, mesh, info, picks, path):
    """Wall-condensation estimate of the quasi-steady gas-space model."""
    c0, um = 273.15, 1e9
    model = ts["gas_model"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0, 0]
    ax.plot(times, ts["evaporation_rate_cond"] * um, "k", label="evaporation at the water surface")
    ax.plot(times, ts["condensation_rate_side"] * um, label="condensation on the side wall")
    ax.plot(times, ts["condensation_rate_lid"] * um, label="condensation on the lid")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rate [microgram/s]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Mass-transfer rates with condensing walls")

    ax = axes[0, 1]
    ax.plot(times, ts["T_lid_gas"] - c0, label="gas under the lid, as simulated")
    ax.plot(times, ts["T_lid_wall_cond"] - c0, "--", label="lid surface, with condensation (estimate)")
    t_surf = np.array([np.mean(s["T"][:, model.k0 - 1]) for s in states])
    ax.plot(times, t_surf - c0, "k:", label="water surface (= dew point of the gas above it)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("temperature [degC]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("The released latent heat warms the lid")

    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(picks)))
    for col, k in zip(colors, picks):
        c = states[k]["cond"]
        if c["flux_side"].max() <= 0 and c["flux_lid"].max() <= 0:
            continue
        # unrolled wall: side wall from the water surface up to the lid, then
        # the lid from the wall inwards to the axis
        s_side = (model.z_side - model.z_side[0]) * MM
        s_lid = s_side[-1] + (model.r_lid[-1] - model.r_lid[::-1]) * MM + 0.1
        ax.plot(np.concatenate([s_side, s_lid]),
                np.concatenate([c["flux_side"], c["flux_lid"][::-1]]) * 1e6,
                color=col, label="t = %g s" % times[k])
    ax.axvline(s_side[-1] + 0.05, color="k", lw=0.8, ls=":")
    ax.text(0.5 * s_side[-1], 0.02, "side wall (0 = water surface)", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=8)
    ax.text(s_side[-1] + 0.5 * model.r_lid[-1] * MM, 0.02, "lid (towards the axis)", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xlabel("distance along the wall [mm]")
    ax.set_ylabel("condensation flux [mg/m2/s]")
    ax.legend(fontsize=8)
    ax.set_title("Where the vapour condenses")

    ax = axes[1, 1]
    s = states[-1]
    rh_sim = np.ma.masked_where(s["al"] >= 0.5, 100 * s["rh"])
    rh_est = np.ma.masked_all(s["rh"].shape)
    rh_est[:, model.k0:] = 100 * s["cond"]["rh"]
    liquid = np.ma.masked_where(s["al"] < 0.5, np.ones_like(s["al"]))
    r, z = mesh.r_nodes * MM, mesh.z_nodes * MM
    kw = dict(shading="flat", cmap="coolwarm_r", vmin=0, vmax=max(120.0, rh_sim.max()))
    for sign in (-1, 1):
        ax.pcolormesh(sign * r, z, liquid, shading="flat", cmap=WATER_BLUE)
    ax.pcolormesh(-r, z, rh_sim, **kw)
    im = ax.pcolormesh(r, z, rh_est, **kw)
    decorate(ax, mesh, s)
    ax.set_xticks([-3, 0, 3])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("r [mm]      left: as simulated | right: with wall condensation")
    ax.set_ylabel("height above the bottom [mm]")
    fig.colorbar(im, ax=ax, label="relative humidity of the gas [%]", fraction=0.05)
    ax.set_title("Relative humidity at t = %g s" % times[-1])

    for ax in axes.flat[:3]:
        ax.grid(alpha=0.3)
    fig.suptitle("Estimated condensation on the lid and side wall (quasi-steady gas-space model; "
                 "upper estimate,\nthe extra evaporative cooling of the water is not fed back)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiles(times, states, mesh, picks, path):
    c = 273.15
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(picks)))
    for col, k in zip(colors, picks):
        s = states[k]
        axes[0].plot(s["T"][0, :] - c, mesh.zc * MM, color=col, label="t = %g s" % times[k])
        x = np.where(s["al"][0, :] < 0.5, 100 * s["X"][0, :], np.nan)
        axes[1].plot(x, mesh.zc * MM, color=col)
    axes[0].set_xlabel("temperature on the well axis [degC]")
    axes[0].set_ylabel("height above the bottom [mm]")
    axes[1].set_xlabel("water vapour in the gas on the well axis [mol %]")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Animations (GIF)
# --------------------------------------------------------------------------- #
WATER_BLUE = ListedColormap(["#1f4e9c"])


def surface_state(mesh, info, s):
    """Per radial column: liquid temperature at the water surface, saturation
    pressure there and the vapour partial pressure in the first gas cell above."""
    k = np.argmax(s["al"] < 0.5, axis=1)                 # first gas cell
    k = np.clip(k, 1, mesh.nz - 1)
    cols = np.arange(mesh.nr)
    t_surf = s["T"][cols, k - 1]
    return (mesh.rc[cols, k] * MM, t_surf, p_sat(t_surf, info["antoine"]),
            s["X"][cols, k] * s["p"], mesh.zc[k] * MM)       # s["p"]: solver reference pressure


def _frame_layout(title):
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], hspace=0.38, wspace=0.36,
                          left=0.07, right=0.97, top=0.88, bottom=0.09)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])
    head = fig.suptitle(title, fontsize=12)
    return fig, ax_map, ax_top, ax_bot, head


def _finish_map(ax, mesh, s):
    decorate(ax, mesh, s)
    ax.set_xticks([-3, 0, 3])
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("height above the bottom [mm]")


def _cursor_axis(ax, times, t_now, ylabel):
    ax.axvline(t_now, color="k", lw=0.8)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _render(path, title, times, states, ts, mesh, info, draw, cbar_label, fps):
    """draw(ax_map, ax_top, ax_bot, i) -> mappable of the map."""
    fig, ax_map, ax_top, ax_bot, head = _frame_layout(title)
    cbar = None
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=90):
        for i in range(len(times)):
            for ax in (ax_map, ax_top, ax_bot):
                ax.clear()
            im = draw(ax_map, ax_top, ax_bot, i)
            _finish_map(ax_map, mesh, states[i])
            if cbar is None:
                cbar = fig.colorbar(im, ax=ax_map, label=cbar_label, fraction=0.08, pad=0.04)
            head.set_text("%s   |   t = %3.0f s   heater %.1f degC"
                          % (title, times[i], ts["T_heater"][i] - 273.15))
            writer.grab_frame()
        for _ in range(fps):                             # hold the last frame
            writer.grab_frame()
    plt.close(fig)


def animate_all(times, states, ts, mesh, info, out, fps=8):
    c = 273.15
    zc = mesh.zc * MM
    gas = [np.ma.masked_where(s["al"] >= 0.5, np.ones_like(s["al"])) for s in states]
    liq = [np.ma.masked_where(s["al"] < 0.5, np.ones_like(s["al"])) for s in states]
    level = [zc[np.argmax(s["al"][0, :] < 0.5)] for s in states]
    t_lo = info["T_ambient"] - c
    t_hi = info["T_heater_end"] - c
    surf = [surface_state(mesh, info, s) for s in states]

    # ---- temperature ----------------------------------------------------- #
    def draw_T(ax_map, ax_top, ax_bot, i):
        s = states[i]
        im = mirrored(ax_map, mesh, s["T"] - c, cmap="inferno", vmin=t_lo, vmax=t_hi)
        ax_top.plot(zc, s["T"][0, :] - c, color="tab:red", label="on the well axis")
        ax_top.plot(zc, s["T"][-1, :] - c, color="tab:orange", lw=1, label="at the side wall")
        ax_top.axvline(level[i], color="tab:blue", ls="--", lw=1, label="water surface")
        ax_top.set_xlim(0, mesh.h * MM)
        ax_top.set_ylim(t_lo - 0.5, t_hi + 0.5)
        ax_top.set_xlabel("height above the bottom [mm]")
        ax_top.set_ylabel("temperature [degC]")
        ax_top.legend(fontsize=8, loc="upper right")
        ax_top.grid(alpha=0.3)
        ax_bot.plot(times, ts["T_heater"] - c, "k--", label="heater")
        ax_bot.plot(times, ts["T_bottom_fluid"] - c, label="water at the bottom wall")
        ax_bot.plot(times, ts["T_water_mean"] - c, label="water, mean")
        ax_bot.plot(times, ts["T_gas_mean"] - c, label="gas, mean")
        ax_bot.plot(times, ts["T_lid_gas"] - c, label="gas at the %s" % top_name(info))
        ax_bot.legend(fontsize=7, loc="upper left")
        _cursor_axis(ax_bot, times, times[i], "temperature [degC]")
        return im

    _render(out / "heating_temperature.gif", "Temperature", times, states, ts, mesh, info,
            draw_T, "temperature [degC]", fps)

    # ---- water vapour ---------------------------------------------------- #
    x_hi = 100 * max(np.max(np.where(s["al"] < 0.5, s["X"], 0)) for s in states)

    def draw_X(ax_map, ax_top, ax_bot, i):
        s = states[i]
        mirrored(ax_map, mesh, liq[i], cmap=WATER_BLUE)
        im = mirrored(ax_map, mesh, 100 * s["X"] * gas[i], cmap="YlGn", vmin=0, vmax=x_hi)
        g = s["al"][0, :] < 0.5
        x_sat = 100 * p_sat(s["T"][0, :], info["antoine"]) / s["p"]
        ax_top.plot(zc[g], 100 * s["X"][0, g], color="tab:green", label="water vapour in the gas")
        ax_top.plot(zc[g], x_sat[g], "k--", lw=1, label="saturation at the local temperature")
        ax_top.set_xlim(0, mesh.h * MM)
        ax_top.set_ylim(0, 1.1 * x_hi)
        ax_top.axvline(level[i], color="tab:blue", ls="--", lw=1)
        ax_top.set_xlabel("height above the bottom, on the well axis [mm]")
        ax_top.set_ylabel("vapour content [mol %]")
        ax_top.legend(fontsize=8, loc="lower left")
        ax_top.grid(alpha=0.3)
        ax_bot.plot(times, ts["evaporated_mass"] * 1e9, color="tab:blue")
        ax_bot.plot(times[i], ts["evaporated_mass"][i] * 1e9, "o", color="tab:blue")
        _cursor_axis(ax_bot, times, times[i], "evaporated water [microgram]")
        return im

    _render(out / "heating_vapour.gif", "Water vapour (liquid water in blue)", times, states,
            ts, mesh, info, draw_X, "water vapour in the gas [mol %]", fps)

    # ---- humidity / evaporation ------------------------------------------ #
    rh_hi = max(120.0, 100 * max(np.max(np.where(s["al"] < 0.5, s["rh"], 0)) for s in states))
    p_hi = 1.05 * max(max(sf[2].max(), sf[3].max()) for sf in surf) / 1e3
    rate = np.gradient(ts["evaporated_mass"], times) * 1e9 if len(times) > 1 else np.zeros_like(times)
    rr = np.tile(mesh.zc * MM, (mesh.nr, 1))

    def draw_RH(ax_map, ax_top, ax_bot, i):
        s = states[i]
        mirrored(ax_map, mesh, liq[i], cmap=WATER_BLUE)
        im = mirrored(ax_map, mesh, 100 * s["rh"] * gas[i], cmap="coolwarm_r", vmin=0, vmax=rh_hi)
        rh_gas = np.where(s["al"] < 0.5, s["rh"], 0)
        for sign in (-1, 1):
            ax_map.contour(sign * mesh.rc * MM, rr, rh_gas, levels=[1.0], colors="k", linewidths=0.8)
        r, _, ps, pv, _ = surf[i]
        ax_top.plot(r, ps / 1e3, color="tab:red", label="saturation pressure at the water surface")
        ax_top.plot(r, pv / 1e3, color="tab:green", label="vapour pressure in the gas just above")
        ax_top.fill_between(r, pv / 1e3, ps / 1e3, where=ps > pv, color="tab:red", alpha=0.15)
        ax_top.set_xlim(0, mesh.wall_radius(mesh.h) * MM)
        ax_top.set_ylim(1.0, p_hi)
        ax_top.set_xlabel("radius along the water surface [mm]")
        ax_top.set_ylabel("pressure [kPa]")
        ax_top.set_title("driving force of evaporation (shaded: surface evaporates)", fontsize=9)
        ax_top.legend(fontsize=8, loc="upper left")
        ax_top.grid(alpha=0.3)
        ax_bot.plot(times, rate, color="tab:red")
        ax_bot.plot(times[i], rate[i], "o", color="tab:red")
        ax_bot.set_ylim(min(0, rate[1:].min()) if len(rate) > 1 else 0,
                        1.15 * rate[2:].max() if len(rate) > 3 else 1)
        _cursor_axis(ax_bot, times, times[i], "net evaporation rate [microgram/s]")
        return im

    _render(out / "heating_evaporation.gif",
            "Relative humidity of the gas (black line: 100 %)", times, states, ts, mesh, info,
            draw_RH, "relative humidity [%]", fps)

    # ---- flow ------------------------------------------------------------ #
    u_hi = 1e3 * max(np.percentile(s["Umag"], 99.5) for s in states)
    step_r, step_z = max(1, mesh.nr // 8), max(1, mesh.nz // 26)
    qr = mesh.rc[::step_r, ::step_z] * MM
    qz = np.tile(zc, (mesh.nr, 1))[::step_r, ::step_z]

    def draw_U(ax_map, ax_top, ax_bot, i):
        s = states[i]
        im = mirrored(ax_map, mesh, 1e3 * s["Umag"], cmap="viridis", vmin=0, vmax=u_hi)
        ur, uz = s["U"][::step_r, ::step_z, 0], s["U"][::step_r, ::step_z, 2]
        norm = np.hypot(ur, uz) + 1e-30
        show = norm > 0.02 * u_hi * 1e-3
        for sign in (-1, 1):
            ax_map.quiver((sign * qr)[show], qz[show], (sign * ur / norm)[show], (uz / norm)[show],
                          color="w", scale=28, width=0.006, headwidth=4)
        ax_top.plot(zc, 1e3 * s["U"][0, :, 2], color="tab:purple", label="on the well axis")
        ax_top.plot(zc, 1e3 * s["U"][int(0.8 * mesh.nr), :, 2], color="tab:gray",
                    label="at 80 % of the radius")
        ax_top.axvline(level[i], color="tab:blue", ls="--", lw=1)
        ax_top.axhline(0, color="k", lw=0.6)
        ax_top.set_xlim(0, mesh.h * MM)
        ax_top.set_ylim(-u_hi, u_hi)
        ax_top.set_xlabel("height above the bottom [mm]")
        ax_top.set_ylabel("vertical velocity [mm/s]")
        ax_top.legend(fontsize=8, loc="upper right")
        ax_top.grid(alpha=0.3)
        ax_bot.plot(times, 1e3 * ts["U_max"], color="tab:purple")
        ax_bot.plot(times[i], 1e3 * ts["U_max"][i], "o", color="tab:purple")
        _cursor_axis(ax_bot, times, times[i], "maximum velocity [mm/s]")
        return im

    _render(out / "heating_flow.gif", "Natural convection (arrows: flow direction)", times,
            states, ts, mesh, info, draw_U, "velocity magnitude [mm/s]", fps)


# --------------------------------------------------------------------------- #
# 3-D views of the whole well (the axisymmetric solution revolved about the axis)
# --------------------------------------------------------------------------- #
def _well_3d(ax, mesh, field, liquid, cmap, vmin, vmax, n_theta=40):
    """Draw the revolved well with the quarter 0..90 deg cut away. `field` is
    [ir, iz]; cells with liquid=True are drawn in blue if cmap is not for T."""
    norm = plt.Normalize(vmin, vmax)
    colours = cmap(norm(field))
    if liquid is not None:
        colours[liquid] = (0.12, 0.31, 0.61, 1.0)
    r, z = mesh.r_nodes * MM, mesh.z_nodes * MM
    theta = np.linspace(np.pi / 2, 2 * np.pi, n_theta + 1)
    ax.computed_zorder = False          # draw in the order given: far parts first
    kw = dict(rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

    # outer wall (seen from outside), bottom and top discs
    th, zz = np.meshgrid(theta, z[-1, :], indexing="ij")
    rw = np.tile(r[-1, :], (n_theta + 1, 1))
    ax.plot_surface(rw * np.cos(th), rw * np.sin(th), zz,
                    facecolors=np.tile(colours[-1, :], (n_theta, 1, 1)), **kw)
    for iz, cell in ((0, 0), (-1, -1)):
        th, rr = np.meshgrid(theta, r[:, iz], indexing="ij")
        ax.plot_surface(rr * np.cos(th), rr * np.sin(th), np.full_like(rr, z[0, iz]),
                        facecolors=np.tile(colours[:, cell], (n_theta, 1, 1)), **kw)
    # the two cut faces show the computed cross-section
    for ang in (np.pi / 2, 0.0):
        ax.plot_surface(r * np.cos(ang), r * np.sin(ang), z, facecolors=colours, **kw)
        ax.plot(r[-1, :] * np.cos(ang), r[-1, :] * np.sin(ang), z[-1, :], "k", lw=1.0, zorder=10)
    ax.plot([0, 0], [0, 0], [0, mesh.h * MM], "k", lw=0.6, zorder=10)
    ax.plot(r[-1, -1] * np.cos(theta), r[-1, -1] * np.sin(theta),      # top rim
            np.full_like(theta, z[0, -1]), "k", lw=1.0, zorder=10)

    lim = mesh.wall_radius(mesh.h) * MM
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, mesh.h * MM)
    ax.set_box_aspect((1, 1, mesh.h * MM / (2 * lim)), zoom=1.25)
    ax.view_init(elev=24, azim=45)
    ax.set_axis_off()
    ax.patch.set_visible(False)         # neighbouring panels may overlap
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def _surface_ring(ax, mesh, s):
    """White line where the water surface meets the two cut faces."""
    k = int(np.argmax(s["al"][-1, :] < 0.5))
    zs = mesh.z_nodes[0, k] * MM
    rs = mesh.wall_radius(mesh.z_nodes[0, k]) * MM
    th = np.linspace(np.pi / 2, 2 * np.pi, 80)
    ax.plot([0, rs], [0, 0], [zs, zs], "w", lw=1.3, zorder=11)
    ax.plot([0, 0], [0, rs], [zs, zs], "w", lw=1.3, zorder=11)


def _draw_3d_pair(ax_t, ax_x, mesh, info, s, x_hi):
    c = 273.15
    m_t = _well_3d(ax_t, mesh, s["T"] - c, None, plt.cm.inferno,
                   info["T_ambient"] - c, info["T_heater_end"] - c)
    m_x = _well_3d(ax_x, mesh, 100 * s["X"], s["al"] >= 0.5, plt.cm.YlGn, 0, x_hi)
    for ax in (ax_t, ax_x):
        _surface_ring(ax, mesh, s)
    return m_t, m_x


def plot_3d(times, states, mesh, info, picks, path):
    x_hi = 100 * max(np.max(np.where(s["al"] < 0.5, s["X"], 0)) for s in states)
    n = len(picks)
    fig = plt.figure(figsize=(2.3 * n + 1.5, 10))
    axes = [[fig.add_subplot(2, n, row * n + j + 1, projection="3d") for j in range(n)]
            for row in range(2)]
    fig.subplots_adjust(left=0.0, right=0.9, top=0.93, bottom=0.0, wspace=-0.35, hspace=-0.1)
    for j, k in enumerate(picks):
        m_t, m_x = _draw_3d_pair(axes[0][j], axes[1][j], mesh, info, states[k], x_hi)
        axes[0][j].set_title("t = %g s" % times[k])
    fig.colorbar(m_t, ax=axes[0], label="temperature [degC]", fraction=0.02, pad=0.0,
                 shrink=0.6)
    fig.colorbar(m_x, ax=axes[1], label="water vapour in the gas [mol %]\n(blue: liquid water)",
                 fraction=0.02, pad=0.0, shrink=0.6)
    fig.suptitle("Whole well (height %.1f mm), one quarter cut away - axisymmetric result "
                 "revolved; white line: water surface" % (mesh.h * MM))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def animate_3d(times, states, ts, mesh, info, path, fps=8):
    x_hi = 100 * max(np.max(np.where(s["al"] < 0.5, s["X"], 0)) for s in states)
    fig = plt.figure(figsize=(11, 6.6))
    ax_t = fig.add_axes([0.00, 0.02, 0.44, 0.88], projection="3d")
    ax_x = fig.add_axes([0.48, 0.02, 0.44, 0.88], projection="3d")
    cax_t = fig.add_axes([0.445, 0.2, 0.012, 0.55])
    cax_x = fig.add_axes([0.925, 0.2, 0.012, 0.55])
    head = fig.suptitle("")
    fig.text(0.22, 0.89, "temperature", ha="center", fontsize=11)
    fig.text(0.70, 0.89, "water vapour in the gas (blue: liquid water)", ha="center", fontsize=11)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=90):
        for i, s in enumerate(states):
            ax_t.clear()
            ax_x.clear()
            m_t, m_x = _draw_3d_pair(ax_t, ax_x, mesh, info, s, x_hi)
            if i == 0:
                fig.colorbar(m_t, cax=cax_t, label="degC")
                fig.colorbar(m_x, cax=cax_x, label="mol %")
            head.set_text("Whole well, one quarter cut away   |   t = %3.0f s   heater %.1f degC   "
                          "evaporated %.2f microgram"
                          % (times[i], ts["T_heater"][i] - 273.15, ts["evaporated_mass"][i] * 1e9))
            writer.grab_frame()
        for _ in range(fps):
            writer.grab_frame()
    plt.close(fig)


def write_csv(times, ts, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + list(ts))
        for i, t in enumerate(times):
            w.writerow(["%g" % t] + ["%.8g" % ts[k][i] for k in ts])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=Path, default=HERE / "case")
    ap.add_argument("--out", type=Path,
                    help="output directory (default: results, or results_<name> for case_<name>)")
    ap.add_argument("--times", type=float, nargs="+",
                    help="times [s] of the cross-section maps (default: 6 evenly spaced)")
    ap.add_argument("--no-gif", action="store_true", help="skip the animated GIFs")
    ap.add_argument("--fps", type=int, default=8, help="frames per second of the GIFs")
    ap.add_argument("--raw", action="store_true",
                    help="raw simulation data only: no gas-volume correction, no condensation "
                         "model, solver pressure/liquid volume/level (default output "
                         "results_raw_sim_data[_<name>])")
    args = ap.parse_args()

    if not args.case.is_absolute() and not args.case.is_dir():
        args.case = HERE / args.case
    if args.out is None:
        name = args.case.resolve().name
        suffix = "" if name == "case" else "_" + name.replace("case_", "", 1)
        args.out = HERE / (("results_raw_sim_data" if args.raw else "results") + suffix
                           if suffix or args.raw else "results")
    info = read_case_info(args.case)
    mesh = Mesh(info)
    dirs = time_dirs(args.case)
    if not dirs:
        sys.exit("No time directories with results in %s" % args.case)

    times = np.array([t for t, _ in dirs])
    states = [load_state(mesh, info, d) for _, d in dirs]
    ts = analyse(mesh, info, states, times, raw=args.raw)

    wanted = args.times if args.times else np.linspace(times[0], times[-1], min(6, len(times)))
    picks = sorted({int(np.argmin(np.abs(times - w))) for w in wanted})

    args.out.mkdir(parents=True, exist_ok=True)
    plot_timeseries(times, ts, info, args.out / "heating_timeseries.png", raw=args.raw)
    plot_maps(times, states, mesh, info, picks, args.out / "heating_maps.png")
    plot_profiles(times, states, mesh, picks, args.out / "heating_profiles.png")
    plot_3d(times, states, mesh, info, picks, args.out / "heating_3d.png")
    if not is_open(info) and not args.raw:
        plot_condensation(times, states, ts, mesh, info, picks,
                          args.out / "heating_condensation.png")
    ts.pop("gas_model")
    write_csv(times, ts, args.out / "heating_timeseries.csv")
    if not args.no_gif and len(times) > 1:
        animate_all(times, states, ts, mesh, info, args.out, fps=args.fps)
        animate_3d(times, states, ts, mesh, info, args.out / "heating_3d.gif", fps=args.fps)

    c = 273.15
    print("%d time directories, t = %g ... %g s" % (len(times), times[0], times[-1]))
    print("At t = %g s:" % times[-1])
    print("  heater %.1f degC, water mean %.1f degC (%.1f-%.1f), gas mean %.1f degC"
          % (ts["T_heater"][-1] - c, ts["T_water_mean"][-1] - c, ts["T_water_min"][-1] - c,
             ts["T_water_max"][-1] - c, ts["T_gas_mean"][-1] - c))
    if is_open(info):
        print("  open top: evaporated water %.1f microgram (check from liquid volume: %.1f),"
              " of which %.1f left through the top"
              % (ts["evaporated_mass"][-1] * 1e9,
                 ts["evaporated_mass_from_liquid_volume"][-1] * 1e9,
                 ts["mouth_loss_mass"][-1] * 1e9))
        print("  solver interface source (dmdt): %.1f microgram evaporated"
              % (ts["evaporated_mass_solver"][-1] * 1e9))
        print("  vapour partial pressure %.2f kPa, mean RH %.0f %%, max RH %.0f %%"
              % (ts["p_vapour"][-1] / 1e3, 100 * ts["rh_mean"][-1], 100 * ts["rh_max"][-1]))
    elif args.raw:
        print("  raw simulation data (no corrections):")
        print("    solver pressure %.2f kPa, vapour partial pressure %.2f kPa"
              % (ts["p_total"][-1] / 1e3, ts["p_vapour"][-1] / 1e3))
        print("    vapour added to the gas %.2f microgram, dmdt integral %.2f microgram,"
              " liquid volume %.3f -> %.3f uL"
              % (ts["evaporated_mass"][-1] * 1e9, ts["evaporated_mass_solver"][-1] * 1e9,
                 ts["liquid_volume_solver"][0] * 1e9, ts["liquid_volume_solver"][-1] * 1e9))
        print("    mean RH %.0f %%, max RH %.0f %%" % (100 * ts["rh_mean"][-1], 100 * ts["rh_max"][-1]))
    else:
        print("  as simulated (no wall condensation):")
        print("    sealed-well pressure %.2f kPa (start %.2f), vapour partial pressure %.2f kPa"
              % (ts["p_total"][-1] / 1e3, ts["p_total"][0] / 1e3, ts["p_vapour"][-1] / 1e3))
        print("    evaporated water %.2f microgram, mean RH %.0f %%, max RH %.0f %%"
              % (ts["evaporated_mass"][-1] * 1e9, 100 * ts["rh_mean"][-1], 100 * ts["rh_max"][-1]))
        print("  with wall condensation (quasi-steady estimate, upper bound):")
        print("    sealed-well pressure %.2f kPa, vapour partial pressure %.2f kPa, max RH %.0f %%"
              % (ts["p_total_cond"][-1] / 1e3, ts["p_vapour_cond"][-1] / 1e3,
                 100 * ts["rh_max_cond"][-1]))
        print("    evaporated water %.0f microgram; condensed: lid %.0f, side wall %.0f microgram"
              % (ts["evaporated_mass_cond"][-1] * 1e9, ts["condensed_mass_lid"][-1] * 1e9,
                 ts["condensed_mass_side"][-1] * 1e9))
        print("    lid surface %.1f degC; conduction-only gas model deviates %.2f K (rms) from the"
              " simulated gas temperature"
              % (ts["T_lid_wall_cond"][-1] - c, np.nanmax(ts["model_T_gas_deviation"])))
    print("Figures and CSV written to %s" % args.out)


if __name__ == "__main__":
    main()
