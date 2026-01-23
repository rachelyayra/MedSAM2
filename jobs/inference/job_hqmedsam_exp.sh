#!/bin/bash

#SBATCH --output=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_hqmed.txt               # Standard error file
#SBATCH --error=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/error_hqmed.txt               # Standard error file
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task 
#SBATCH --mail-type=END,FAIL                # Notify on job completion or failure
#SBATCH --mail-user=adjoerachel@email.com    # Email for notifications

# Print job status
echo "SLURM test job running..."



# Activate your virtual environment if using one
source /scratch_net/ken/radjoe/conda/bin/activate /scratch_net/ken/radjoe/conda_envs/medsam2

# Inference script for GLI for bboxes and Points
# python experiments/MedSAM_multi.py \
#   --video_path /scratch_net/ken/radjoe/BraTS/Test/images_UNN \
#   --label_path /scratch_net/ken/radjoe/BraTS/Test/labels_UNN\
#   --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/F_NA_GLI \
#   --model_cfg configs/sam2.1_hiera_t512_original.yaml  \
#   --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/logs/full_finetuning/checkpoints/checkpoint_210.pt \
#   --model_type video


# python experiments/MedSAM_multi.py \
#   --video_path /scratch_net/ken/radjoe/BraTS_SSA/Validation/images_UNN \
#   --label_path /scratch_net/ken/radjoe/BraTS_SSA/Validation/labels_UNN\
#   --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/F_NA_SSA \
#   --model_cfg configs/sam2.1_hiera_t512_original.yaml  \
#   --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/logs/full_finetuning/checkpoints/checkpoint_210.pt \
#   --model_type video

# python experiments/MedSAM_multi.py \
#   --video_path /scratch_net/ken/radjoe/BraTS/Test/images_UNN \
#   --label_path /scratch_net/ken/radjoe/BraTS/Test/labels_UNN\
#   --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/F_A_GLI \
#   --model_cfg configs/sam2.1_hiera_t512_original.yaml  \
#   --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/logs/full_finetuning_augs/checkpoints/checkpoint_220.pt \
#   --model_type video

python experiments/MedSAM_multi.py \
  --video_path /scratch_net/ken/radjoe/BraTS_SSA/Validation/images_UNN \
  --label_path /scratch_net/ken/radjoe/BraTS_SSA/Validation/labels_UNN\
  --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/F_A_SSA \
  --model_cfg configs/sam2.1_hiera_t512_original.yaml  \
  --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/logs/full_finetuning_augs/checkpoints/checkpoint_220.pt \
  --model_type video

# Print job status

echo "SLURM test job running..."