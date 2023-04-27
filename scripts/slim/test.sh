#!/bin/bash

module purge # unload all loaded modules

ml tensorboardX/2.5.1-foss-2022a-CUDA-11.7.0
ml PyTorch3D/0.7.1-foss-2022a-CUDA-11.7.0
ml PyTorch-Lightning/1.8.4-foss-2022a-CUDA-11.7.0
ml PyTorch/1.12.1-foss-2022a-CUDA-11.7.0


python -u test.py \
  --accelerator gpu \
  --data_path /home/pokorsi1/data/nuscenes/preprocess_new \
  --dataset nuscenes \
  --dataset_trained_on waymo \
  --resume_from_checkpoint experiments/waymo/checkpoints/version_3/epoch=0-step=100000.ckpt

# /home/pokorsi1/data/waymo_flow/preprocess
# /home/pokorsi1/data/kitti_lidar_sf


#python -u test.py \
#  --accelerator gpu \
#  --data_path /mnt/personal/vacekpa2/data/kitti_lidar_sf \
# --dataset kittisf \
# --dataset_trained_on rawkitti \
# --resume_from_checkpoint /home/pokorsi1/motion_learning/scripts/slim/experiments/rawkitti/checkpoints/version_9/epoch=2-step=101000.ckpt

