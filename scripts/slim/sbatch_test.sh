#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1     # processes per node
#SBATCH --cpus-per-task=12      # number of CPU cores per process
#SBATCH --partition=gpulong     # put the job into the gpu partition/queue
#SBATCH --gres=gpu:1
#SBATCH --output=log_test.out   # file name for stdout/stderr
#SBATCH --mem=100G              # how much CPU memory can be allocated for the job (hardware limit: 384 GB per node)
#SBATCH --time=0-03:00:00       # maximum wall time allocated for the job (max 24h for the gpu partition)
#SBATCH --job-name=SLIM_test    # job name (default is the name of this file)
#SBATCH --mail-user=pokorsi1@fel.cvut.cz    # where send info about job
#SBATCH --mail-type=FAIL               # what to send, valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

module purge # unload all loaded modules

ml tensorboardX/2.5.1-foss-2022a-CUDA-11.7.0
ml PyTorch3D/0.7.1-foss-2022a-CUDA-11.7.0
ml PyTorch-Lightning/1.8.4-foss-2022a-CUDA-11.7.0
ml PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


python -u test.py \
  --accelerator gpu \
  --data_path /home/pokorsi1/data/waymo_flow/preprocess \
  --dataset waymo \
  --dataset_trained_on waymo \
  --resume_from_checkpoint /home/pokorsi1/motion_learning/scripts/slim/experiments/waymo/checkpoints/version_2/epoch=0-step=100000.ckpt



