#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=gpulong     # put the job into the gpu partition/queue
#SBATCH --gres=gpu:1
#SBATCH --output=log.out        # file name for stdout/stderr
#SBATCH --mem=50G               # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=3-00:00:00       # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=SLIM         # job name (default is the name of this file)

#cd $HOME/lidar/models/spvnas
module purge # unload all loaded modules



ml PyTorch-Lightning/1.9.0-foss-2022a-CUDA-11.7.0
ml PyTorch3D/0.7.1-foss-2022a-CUDA-11.7.0
ml PyTorch/1.13.0-foss-2022a-CUDA-11.7.0
ml tensorboardX/2.5.1-foss-2022a-CUDA-11.7.0

python -u train.py --dataset kitti --accelerator gpu --fast_dev_run True \
  --data_path /home/pokorsi1/data/rawkitti/prepared
