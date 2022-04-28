#!/bin/bash 
#SBATCH --job-name=cvae-cluster-30     #Name of your job
#SBATCH --cpus-per-task=4    #Number of cores to reserve
#SBATCH --mem-per-cpu=16G     #Amount of RAM/core to reserve
#SBATCH --time=06:00:00      #Maximum allocated time
#SBATCH --qos=6hours         #Selected queue to allocate your job
#SBATCH --output=cvae-cluster-30.o   #Path and name to the file for the STDOUT
#SBATCH --error=cvae-cluster-30.e    #Path and name to the file for the STDERR

#SBATCH --ntasks=1
#SBATCH --partition=shig2  # or titanx
##SBATCH --gres=gpu:2        # --gres=gpu:2 for two GPU, etc


ml Python/3.6.6-foss-2018b
ml CUDA
source $HOME/venv_MA/bin/activate               
python main.py --config="./_config/cluster_config_30.json"
