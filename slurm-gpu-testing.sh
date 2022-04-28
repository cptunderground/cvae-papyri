#!/bin/bash 
#SBATCH --job-name=cvae-gpu-test     #Name of your job
#SBATCH --cpus-per-task=1    #Number of cores to reserve
#SBATCH --mem-per-cpu=8G     #Amount of RAM/core to reserve
#SBATCH --time=00:30:00      #Maximum allocated time
#SBATCH --qos=30min         #Selected queue to allocate your job
#SBATCH --output=cvae-gpu-testing.o   #Path and name to the file for the STDOUT
#SBATCH --error=cvae-gpu-testing.e    #Path and name to the file for the STDERR

#SBATCH --ntasks=1
#SBATCH --partition=pascal  # or titanx
#SBATCH --gres=gpu:1        # --gres=gpu:2 for two GPU, etc


ml Python/3.6.6-foss-2018b
ml CUDA
source $HOME/venv_MA/bin/activate               
python main.py --config="./_config/testing.json"
