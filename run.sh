#!/bin/bash
#SBATCH --job-name=room-fem
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --licenses=ansys@ansys.it.auth.gr:4
#SBATCH --time=06:00:00
#SBATCH --output=room_fem.out

module load ansys/2026R1
source ~/pymapdl-env/bin/activate
rm -f ~/.conn/mapdl-*.sock
python -u room-fem.py