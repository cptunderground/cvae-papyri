#!/bin/bash 
#SBATCH --job-name=myrun     #Name of your job
#SBATCH --cpus-per-task=1    #Number of cores to reserve
#SBATCH --mem-per-cpu=1G     #Amount of RAM/core to reserve
#SBATCH --time=00:01:00      #Maximum allocated time
#SBATCH --qos=30min         #Selected queue to allocate your job
#SBATCH --output=myrun.o   #Path and name to the file for the STDOUT
#SBATCH --error=myrun.e    #Path and name to the file for the STDERR

ml Miniconda2
source activate cvae-cluster-3.8                 
python main.py    
