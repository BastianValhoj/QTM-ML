# LSBATCH: User input
#!/bin/bash
### Job name (shown in queue and in email notificaitons)
#BSUB -J SE0-12-100
### Specify which queue (here, just the HPC)
#BSUB -q hpc
### Request flag: reserve 16GB available for the duration of the job
#BSUB -R "rusage[mem=16GB]"
### Notify when job begins
#BSUB -B
### Notify when job ends
#BSUB -N
### Wall-clock time (HH:MM)
#BSUB -W 0:30
### Number of processors
#BSUB -n 1
### Request ressrouce: Use single host for ressources
#BSUB -R "span[hosts=1]"
### mail notification
#BSUB -u s192943@student.dtu.dk
### -- Specify the output and error file. %J is the job-id -- 
#BSUB -o out.write-NC0-12-100.%J.log
#BSUB -e err.write-NC0-12-100.%J.log


uv run write_rsse.py --nc 0 --ns 12 --nb 100
