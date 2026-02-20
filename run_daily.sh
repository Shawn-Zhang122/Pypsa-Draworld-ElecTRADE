#!/bin/bash
set -e

# load conda
source /Users/zhang/opt/anaconda3/etc/profile.d/conda.sh

# activate environment
conda activate pypsa-china

# go to project root
cd /Users/zhang/Resource/China-ElecTrade-APP/Pypsa-Draworld-ElecTRADE

# set Gurobi license (relative to project root)
export GRB_LICENSE_FILE=$PWD/solver/gurobi.lic

# run daily model
python model/run_daily.py

# push results only if changed
git add docs/out/
git diff --cached --quiet || git commit -m "daily update $(date +%F)"
git push