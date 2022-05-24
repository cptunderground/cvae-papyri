#!/bin/bash 
#SBATCH --job-name=cvae-100-gpu    #Name of your job
#SBATCH --cpus-per-task=1    #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve
#SBATCH --time=6:00:00      #Maximum allocated time
#SBATCH --qos=6hours          #Selected queue to allocate your job
#SBATCH --output=cvae-cluster-100.o   #Path and name to the file for the STDOUT
#SBATCH --error=cvae-cluster-100.e    #Path and name to the file for the STDERR

#SBATCH --ntasks=1
#SBATCH --partition=pascal  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc

ml Python/3.6.6-foss-2018b
ml CUDA
source $HOME/venv_MA/bin/activate               
python main.py --config="./_config/config-cluster_100_gpu.json"
