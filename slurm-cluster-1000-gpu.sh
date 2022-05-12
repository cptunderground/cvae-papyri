#!/bin/bash 
#SBATCH --job-name=cvae-1000-gpu    #Name of your job
#SBATCH --cpus-per-task=1    #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve
#SBATCH --time=1-00:00:00      #Maximum allocated time
#SBATCH --qos=1day          #Selected queue to allocate your job
#SBATCH --output=cvae-cluster-1000.o   #Path and name to the file for the STDOUT
#SBATCH --error=cvae-cluster-1000.e    #Path and name to the file for the STDERR

#SBATCH --ntasks=1
#SBATCH --partition=pascal  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc

ml Python/3.6.6-foss-2018b
ml CUDA
source $HOME/venv_MA/bin/activate               
python main.py --config="./_config/cluster_config_1000_gpu.json"
