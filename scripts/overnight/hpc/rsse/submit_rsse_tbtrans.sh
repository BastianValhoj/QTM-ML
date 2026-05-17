# LSBATCH: User input
#!/bin/bash
#BSUB -J RSSE
#BSUB -q hpc
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -N
#BSUB -W 0:1
#BSUB -n 24
#BSUB -R "span[hosts=1]"
### mail notification
#BSUB -u s192943@student.dtu.dk
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o out.rsse_%J.log
#BSUB -e err.rsse_%J.log


#source /dtu/sw/dcc/dcc-sw.bash

#module load mpi
source /zhome/b9/2/144925/speciale/.venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

#mpirun -n 1 python rsse_tbtrans_mpi.py
python rsse_tbtrans.py
#mpirun python SSE_Gr_TBT.py

