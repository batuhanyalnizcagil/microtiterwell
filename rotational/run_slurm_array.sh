#!/bin/bash
# OPTIONAL template for an HPC cluster with SLURM: runs the 7 RPM cases as one
# job array (one case per array task). Adapt partition/module names to the
# cluster, and set numberOfSubdomains in cases/rpm_XXXX/system/decomposeParDict
# of every case to match --ntasks.
#
#   sbatch run_slurm_array.sh

#SBATCH --job-name=well_shaking
#SBATCH --array=0-6
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%A_%a.out

RPMS=(300 600 900 1200 1500 1800 2100)

# module load openfoam/v2412          # <- cluster specific
# source /path/to/openfoam2412/etc/bashrc

cd "$SLURM_SUBMIT_DIR" || exit 1
CASE=$(printf 'cases/rpm_%04d' "${RPMS[$SLURM_ARRAY_TASK_ID]}")
"$CASE/Allrun"
