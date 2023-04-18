#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=gpulong     # put the job into the gpu partition/queue
#SBATCH --gres=gpu:1
#SBATCH --output=log_kitti.out        # file name for stdout/stderr
#SBATCH --mem=100G               # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=3-00:00:00       # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=SLIM                     # job name (default is the name of this file)
#SBATCH --mail-user=pokorsi1@fel.cvut.cz    # where send info about job
#SBATCH --mail-type=ALL                     # what to send, valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

module purge # unload all loaded modules

# ml tensorboardX/2.5.1-foss-2022a-CUDA-11.7.0
#ml tensorboardX/2.4.1-foss-2021a-CUDA-11.3.1

#ml PyTorch3D/0.7.1-foss-2022a-CUDA-11.7.0
#ml PyTorch3D/0.7.1-foss-2021b-CUDA-11.4.1

#ml PyTorch/1.13.0-foss-2022a-CUDA-11.7.0
#ml PyTorch/1.12.0-foss-2021b-CUDA-11.4.1

#ml PyTorch-Lightning/1.9.0-foss-2022a-CUDA-11.7.0
#ml PyTorch-Lightning/1.6.5-foss-2021b-CUDA-11.4.1

ml tensorboardX/2.5.1-foss-2022a-CUDA-11.7.0
ml PyTorch3D/0.7.1-foss-2022a-CUDA-11.7.0
ml PyTorch-Lightning/1.8.4-foss-2022a-CUDA-11.7.0
ml PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

python -u train.py --dataset rawkitti --accelerator gpu --fast_dev_run false \
  --data_path /home/pokorsi1/data/rawkitti/prepared