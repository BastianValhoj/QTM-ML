# LSBATCH: User input
#!/bin/bash
#BSUB -J RSSE
#BSUB -q hpc
#BSUB -R "rusage[mem=36GB]"
#BSUB -B
#BSUB -N
#BSUB -o SSE_%J.out
#BSUB -e SSE_%J.err
#BSUB -W 4:00
#BSUB -n 24
#BSUB -R "span[hosts=1]"

# Load conda module (required on DTU HPC)
source /zhome/3b/7/209984/miniconda3/etc/profile.d/conda.sh

# activate your conda envirnoment
conda activate /zhome/3b/7/209984/miniconda3/envs/hubbard_env

# use your specific python
export PYTHONPATH="${PYTHONPATH}:/dtu/hpc.data/3b/7/209984/SofiasProject/hubbard/"

# MPI
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mpirun python -u SSE_Gr_TBT.py

