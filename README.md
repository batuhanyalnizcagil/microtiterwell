# Microtiter well simulations (OpenFOAM v2412)

CFD cases for one well of a Greiner 96-well polystyrene microplate, with the
Python scripts used to evaluate the results.

| Folder | Physics | Solver | Model |
|---|---|---|---|
| `rotational/cases/rpm_XXXX` | orbital shaking of the partly filled well (7 speeds, 300-2100 rpm) | `interFoam` | 3-D, isothermal, two-phase (water/air) |
| `heating/case` | well heated from below, **sealed** top | `icoReactingMultiphaseInterFoam` | 2-D axisymmetric, heat transfer + evaporation |
| `heating/case_open` | the same, **open** top (mouth at room air) | `icoReactingMultiphaseInterFoam` | 2-D axisymmetric, heat transfer + evaporation |

The repository contains only the case set-up (initial fields `0.orig`, `constant`,
`system`, run scripts) and the post-processing scripts. No mesh, no results.
All case files are ordinary OpenFOAM dictionaries - edit them directly.

Well geometry (all cases): bottom radius 3.315 mm, top radius 3.5 mm, height 10.9 mm.

---

## 1. Requirements

- **OpenFOAM v2412** (ESI / openfoam.com version). Both solvers are standard
  applications of that release; nothing has to be compiled.
- An MPI installation (`mpirun`) for the parallel rotational cases.
- **Python 3** with `numpy` and `matplotlib` (Pillow, which matplotlib installs,
  writes the GIF animations). Tested with numpy 1.26 and matplotlib 3.6.
  ParaView is optional (for looking at the fields yourself).

Load OpenFOAM in every new terminal before running a case, for example

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
```

(the path depends on where OpenFOAM is installed on your machine).

After cloning, make sure the run scripts are executable (git normally keeps this):

```bash
chmod +x rotational/cases/*/All* heating/case*/All* rotational/*.sh
```

---

## 2. Running a case

Every case has three scripts:

| Script | What it does |
|---|---|
| `Allrun.pre` | mesh (`blockMesh`, `checkMesh`, `topoSet` for heating), copies `0.orig` -> `0`, `setFields` (initial water) |
| `Allrun` | runs `Allrun.pre` if there is no mesh yet, then the solver (and for the rotational cases `decomposePar` / `reconstructPar`) |
| `Allclean` | deletes mesh, results and logs; keeps the set-up |

### 2.1 Orbital shaking (rotational)

```bash
cd rotational/cases/rpm_1500
./Allrun
```

- Runs in parallel on the number of processes set in `system/decomposeParDict`
  (`numberOfSubdomains 4`). Change it there if your machine has more/fewer cores.
- Run several cases one after another: `rotational/run_all.sh` (all) or
  `rotational/run_all.sh 300 900` (selected speeds). `run_slurm_array.sh` is a
  template for a SLURM cluster.
- Simulated time 5 s. Measured run time on 4 cores (medium mesh, 64 480 cells):
  about 15 h for 1500 rpm (measured with a 0.5 mm orbit radius; the stronger shaking of the
  current 1 mm radius may take longer).
- The solver log keeps only one time step in every 200 (plus warnings/errors) to
  stay small.

### 2.2 Heating (sealed and open top)

```bash
cd heating/case          # or heating/case_open
./Allrun
```

- Serial (the 2-D mesh is too small to profit from MPI).
- Simulated time 450 s. Measured run time: about 7-8 h (about 10 min per 5 s of
  simulated time) when nothing else runs on the machine.
- The solver log keeps only one time step in every 2000 (plus warnings/errors).

### 2.3 Progress, stop, restart

```bash
grep "^Time =" log.icoReactingMultiphaseInterFoam | tail -1     # heating
grep "^Time =" log.interFoam | tail -1                          # rotational
```

Stop a run with Ctrl+C, or from another terminal

```bash
pkill -x icoReactingMult      # heating (process name is cut to 15 characters)
pkill -x interFoam            # rotational
```

`controlDict` has `startFrom latestTime`, so `./Allrun` continues from the last
written time directory (rotational: every 0.05 s, heating: every 5 s).
`Allrun` overwrites the log file, so rename the old log first if you want to keep it.
If the run was killed while writing, delete the last (incomplete) time directory
before restarting.

---

## 3. Post-processing

The scripts read the ASCII fields of the time directories and the file
`caseInfo.txt` of each case (geometry, mesh layout and physical constants).
They never run OpenFOAM. Output goes into a `results*` folder next to the script.

### 3.1 Rotational

```bash
cd rotational
python3 postprocess_rotational.py              # all cases that have results
python3 postprocess_rotational.py --animate    # + animated sections (GIF)
python3 plot_setup.py --rpm 1500               # geometry, mesh, boundary-condition figures
python3 make_shaking_gif.py --rpm 1500 --t0 4.0 --t1 4.4   # shaking seen in the lab frame
```

Results in `rotational/results/`: per speed a time series (`rpm_XXXX_timeseries.png`),
free-surface snapshots, vertical sections, and a summary over all speeds
(`summary_all_rpm.csv`, `summary_all_rpm.png`). The free surface is taken from the iso-surface
alpha.water = 0.5 written during the run into `postProcessing/` (see `functions`
in `system/controlDict`).

### 3.2 Heating

```bash
cd heating
python3 postprocess_heating.py                      # sealed case  -> results/
python3 postprocess_heating.py --case case_open     # open case    -> results_open/
python3 postprocess_heating.py --raw                # sealed, no corrections -> results_raw_sim_data/
python3 postprocess_heating.py --no-gif             # static figures only (faster)
python3 compare_open_sealed.py                      # sealed vs open -> results_compare/
python3 plot_setup.py                               # set-up figures (written for the sealed case)
```

Outputs: time series (temperatures, pressure, evaporation, humidity), cross-section
maps, axis profiles, a CSV with all time series, animated GIFs and a 3-D view.
`compare_open_sealed.py` needs the outputs of the first two commands.
Run `python3 <script> --help` for all options.

What the heating post-processing adds to the raw solver data (all switched off
by `--raw`):

- **Sealed case:** the solver is incompressible, so the pressure rise of the
  closed gas is computed afterwards from the ideal-gas law. The gas inventories are
  scaled to the geometric gas volume (the solver's liquid volume shrinks
  artificially, see 5.). A quasi-steady model of the gas space estimates how much
  vapour would condense on the cooler lid and side wall (`heating_condensation.png`,
  `*_cond` columns); this is an upper estimate, not simulation output.
- **Open case:** evaporated mass = vapour stored in the gas + vapour that left
  through the mouth. The solver's own interface source (`dmdt`) and the loss of
  liquid volume are shown for comparison.

---

## 4. Changing parameters by hand

All values are in SI units. `caseInfo.txt` is **not** read by OpenFOAM, only by
the post-processing. If you change geometry, mesh or a physical constant, change it
in the OpenFOAM dictionary **and** in `caseInfo.txt` of that case.

`caseInfo.txt` is plain text, one entry per line (`name  value`); text after `//` is a
comment. Each entry is commented with its unit and the OpenFOAM file it must agree with.
Values marked "information only" are not used in calculations. Example:

```
fill_height    0.00545                    // initial water depth [m] (must match system/setFieldsDict)
antoine        23.1964  -3816.44  -46.13  // saturation pressure: p_sat = exp(A + B/(T + C)) [Pa], T in K
contact_angle  advancing 95  receding 75  static 85
```

It is read by `case_info.py` (one copy in `rotational/`, one in `heating/`).

### 4.1 Rotational cases

| Parameter | File |
|---|---|
| shaking (speed, orbit, ramp) | `constant/acceleration.dat` (see below) |
| end time, write interval | `system/controlDict` (`endTime 5`, `writeInterval 0.05`) |
| time-step limit | `system/caseParameters` (`maxDeltaTValue`, capillary limit for the mesh) |
| geometry and mesh | `system/blockMeshDict` (O-grid; vertices and cell counts) |
| fill height (2.75 mm, 96.3 uL) | `system/setFieldsDict` |
| fluid properties, surface tension | `constant/transportProperties` |
| contact angle (75 deg receding / 95 deg advancing) | `0.orig/alpha.water`, `contactAngle` entry (used by `bottom` and `sideWall`) |
| number of processes | `system/decomposeParDict` |

The contact-angle entries look reversed (`thetaA 85; thetaR 95`) on purpose: in
`dynamicAlphaContactAngle` the receding contact line gets `theta0 + (thetaA - thetaR)`,
so this gives 75 deg receding and 95 deg advancing. The opposite sign makes the contact
line self-exciting (a numerical ripple of about +/-0.7 mm); see the comment in the file.

**Shaking table.** The well moves on a circle of radius R = 1 mm (orbit diameter
2 mm) without turning. The case is solved in the frame that translates with the well,
so the shaking is a uniform body acceleration added to gravity
(`tabulatedAccelerationSource` in `constant/fvOptions`). `constant/acceleration.dat`
lists the acceleration of the well every 0.25 ms from 0 to 5.1 s:

```
// comment lines
<number of rows>
(
(t ((ax ay 0) (0 0 0) (0 0 0)))
...
)
```

with omega_target = 2*pi*rpm/60, ramp time T_r = 3 s and

```
t <  T_r : alpha = omega_target/T_r,  omega = alpha*t,     theta = alpha*t^2/2
t >= T_r : alpha = 0,                 omega = omega_target, theta = omega_target*(T_r/2 + t - T_r)

ax = R*(-omega^2*cos(theta) - alpha*sin(theta))
ay = R*(-omega^2*sin(theta) + alpha*cos(theta))
```

The two zero vectors are the angular velocity and acceleration of the frame (zero:
no rotation, hence no Coriolis or centrifugal terms). For a new speed, copy a case
folder, write a new table with any tool you like, and set `rpm` in `caseInfo.txt`.
If you change orbit or ramp, also update `orbit_diameter` / `ramp_time` in
`caseInfo.txt` (used by `make_shaking_gif.py`).

### 4.2 Heating cases

| Parameter | File |
|---|---|
| heater temperature (20 -> 45 degC in 420 s, then constant) | `0.orig/T`, patch `bottom`, table `Ta` |
| heat loss of side wall / lid (1 mm PS, h = 10 W/m2K to 20 degC) | `0.orig/T`, patches `sideWall`, `top` |
| open top: room air 20 degC, 50 % RH | `0.orig/T`, `0.orig/vapour.gas`, `0.orig/air.gas`, `0.orig/p`, `0.orig/p_rgh`, `0.orig/U` (patch `top`) |
| initial humidity of the gas | `0.orig/vapour.gas`, `0.orig/air.gas` (`internalField`) |
| fill height (5.45 mm, half the well) | `system/setFieldsDict` |
| end time, write interval, time-step limit | `system/controlDict` |
| evaporation model, surface tension | `constant/phaseProperties` |
| material properties | `constant/thermophysicalProperties.*` |
| vapour diffusivity in the gas (2.6e-5 m2/s) | `constant/fvOptions`, `0.orig/k`, `0.orig/epsilon` (see below) |
| mesh (coarse: 78 axial x 24 radial cells, 5 deg wedge) | `system/blockMeshDict` (+ `n_axial`, `n_radial` in `caseInfo.txt`) |

Keep the block order in `blockMeshDict` (axial first, then radial) and do not run
`renumberMesh`: the post-processing relies on that cell numbering.
For a finer mesh, scale the cell counts in `blockMeshDict` and reduce `maxDeltaT` in
`controlDict` (capillary limit, about proportional to cell size^1.5).

---

## 5. Modelling notes and known solver artefacts

- **Orbital shaking:** solved in the translating frame of the well; the mesh does not
  move. Orbit radius 1 mm (diameter 2 mm), linear speed ramp 0 -> target in 3 s, total 5 s. Water
  2.75 mm deep, open top (`atmosphere` patch, total pressure 0 relative to ambient).
- **Heating, incompressible solver:** `icoReactingMultiphaseInterFoam` does not solve a
  pressure rise in the closed gas; it is evaluated in post-processing.
- **No molecular vapour diffusion in the bulk gas** in this solver: a k-epsilon model whose
  k and epsilon are frozen by `constant/fvOptions` supplies a constant diffusivity
  (2.6e-5 m2/s) in the gas only. The flow itself is laminar (natural convection,
  Boussinesq in the water).
- **Condensation only at the water surface:** the sealed gas becomes supersaturated
  (RH > 100 %); condensation on lid/walls is only estimated in post-processing.
- **Liquid-volume artefact (sealed case):** the solver's volume-fraction source assumes
  that the vapour volume leaves the domain (`includeVolChange` off in the sealed case,
  and the related `includeDivU` is fixed in the code). The liquid volume therefore shrinks
  by about 1.3 uL per microgram evaporated (about 2 % of the water over 450 s). The open
  case uses `includeVolChange on` and does not show this.
- **Open case mass balance:** about one third of the evaporation removed from the liquid
  by the solver's interface source does not appear as vapour in the gas. The vapour
  budget agrees with a hand estimate of the diffusion through the gas layer, so it is
  taken as the physical evaporation.
- **Surface tension is scaled by 0.01** in the heating cases (`constant/phaseProperties`).
  The water surface stays flat (contact angle 90 deg, no contact-angle model in this
  solver), so this only relaxes the capillary time-step limit (10x larger time step).
- 2-D axisymmetric heating cases cannot show non-axisymmetric convection plumes.

---

## 6. Repository layout

```
rotational/
  cases/rpm_0300 ... rpm_2100/   0.orig/ constant/ system/ Allrun* Allclean caseInfo.txt
  postprocess_rotational.py      results of all speeds
  plot_setup.py                  set-up figures
  make_shaking_gif.py            animation of the shaking in the lab frame
  case_info.py                   reader for caseInfo.txt
  run_all.sh, run_slurm_array.sh
heating/
  case/                          sealed top
  case_open/                     open top
  postprocess_heating.py         results of one heating case
  compare_open_sealed.py         sealed vs open
  plot_setup.py                  set-up figures
  case_info.py                   reader for caseInfo.txt
```

Mesh, time directories, logs, `postProcessing/` and `results*/` are created when you run
and are listed in `.gitignore`.
