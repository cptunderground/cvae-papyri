#!/bin/bash 
#SBATCH --job-name=prototype-testing     #Name of your job
#SBATCH --cpus-per-task=4    #Number of cores to reserve
#SBATCH --mem-per-cpu=16G     #Amount of RAM/core to reserve
#SBATCH --time=06:00:00      #Maximum allocated time
#SBATCH --qos=6hours         #Selected queue to allocate your job
#SBATCH --output=myrun.o   #Path and name to the file for the STDOUT
#SBATCH --error=myrun.e    #Path and name to the file for the STDERR

ml Miniconda2
source activate MA                 
python prototype.py
