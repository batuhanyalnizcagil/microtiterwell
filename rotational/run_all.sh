#!/bin/sh
# Runs the orbital-shaking cases one after another on this machine.
#
#   ./run_all.sh                 # all cases in cases/
#   ./run_all.sh 300 900         # only the listed RPMs
#
# OpenFOAM v2412 must be loaded first (e.g. "source <OpenFOAM install dir>/etc/bashrc"
# or start the "openfoam2412" shell). Cases that already finished are skipped.

cd "${0%/*}" || exit
: "${WM_PROJECT_DIR:?OpenFOAM environment is not loaded}"

if [ "$#" -gt 0 ]
then
    cases=""
    for rpm in "$@"; do cases="$cases cases/rpm_$(printf '%04d' "$rpm")"; done
else
    cases=$(ls -d cases/rpm_* 2>/dev/null)
fi

for c in $cases
do
    if [ ! -d "$c" ]
    then
        echo "$c: not found"
    elif [ -f "$c/log.reconstructPar" ]
    then
        echo "$c: already finished - skipped"
    else
        echo "$c: running ..."
        "$c/Allrun"
    fi
done
