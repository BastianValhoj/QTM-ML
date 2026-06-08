# LSBATCH: User input
#!/bin/bash
### Job name (shown in queue and in email notificaitons)
#BSUB -J bi-NC0-30
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
#BSUB -o out.H-NC0-30.%J.log
#BSUB -e err.H-NC0-30.%J.log

uv run write_bilayer.py --nc 0 --ns 3 --nb 30  
uv run write_bilayer.py --nc 0 --ns 4 --nb 30  
uv run write_bilayer.py --nc 0 --ns 5 --nb 30  
uv run write_bilayer.py --nc 0 --ns 6 --nb 30  
uv run write_bilayer.py --nc 0 --ns 7 --nb 30  
uv run write_bilayer.py --nc 0 --ns 8 --nb 30  
uv run write_bilayer.py --nc 0 --ns 9 --nb 30  
uv run write_bilayer.py --nc 0 --ns 10 --nb 30  
uv run write_bilayer.py --nc 0 --ns 11 --nb 30  
uv run write_bilayer.py --nc 0 --ns 12 --nb 30  
