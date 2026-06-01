#!/bin/bash
# LSBATCH: User input
### Job name (shown in queue and in email notificaitons)
#BSUB -J Trans-NC2-30-AB-40
### Specify which queue (here, just the HPC)
#BSUB -q hpc
### Request flag: reserve 16GB available for the duration of the job
#BSUB -R "rusage[mem=16GB]"
### Notify when job begins
#BSUB -B
### Notify when job ends
#BSUB -N
### Wall-clock time (HH:MM)
#BSUB -W 01:00
### Number of processors
#BSUB -n 3
### Request ressrouce: Use single host for ressources
#BSUB -R "span[hosts=1]"
### mail notification
#BSUB -u s192943@student.dtu.dk
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o out.trans-mpi-NC2-30-AB-40.%J.log
#BSUB -e err.trans-mpi-NC2-30-AB-40.%J.log

module load mpi
source /zhome/b9/2/144925/speciale/.venv/bin/activate


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NC=2
NT=30
STACK=AB
ANGLE=40

for NS in 3 4 5 6 7 8 9 10 11 12; do
    if [ $NS -lt $((1 + 2*(1+NC))) ]; then # if NS less than 1 + 2*(1+NC) -- skip
        echo "### Skipping NS=$NS (invalid for NC=$NC)"
        continue
    fi
    mpirun -n 3 python write_bilayer_transport.py --nc $NC --ns $NS --nt $NT --stack $STACK --angle $ANGLE
done
echo "## All calculations finished"
