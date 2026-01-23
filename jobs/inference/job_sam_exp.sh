#!/bin/bash

#SBATCH --output=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_sam.txt               # Standard error file
#SBATCH --error=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/error_sam.txt               # Standard error file
#SBATCH --gres=gpu:a6000:2
#SBATCH --ntasks=4                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task 
#SBATCH --mail-type=END,FAIL                # Notify on job completion or failure
#SBATCH --mail-user=adjoerachel@email.com    # Email for notifications

# Print job status
echo "SLURM test job running..."

# module load miniconda3
# module load python/3.10.16


# Activate your virtual environment if using one
source /scratch_net/ken/radjoe/conda/bin/activate

# Inference script for GLI for bboxes and Points
python experiments/SAM_experiment.py \
  --video_path /scratch_net/ken/radjoe/BraTS/Training/images_UNN\
  --label_path /scratch_net/ken/radjoe/BraTS/Training/labels_UNN\
  --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_sam/BraTS_SSA/SAM2 \
  --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml\
  --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAM_DIR/sam2/checkpoints/sam2.1_hiera_tiny.pt \
  --model_type video


# python experiments/SAM_experiment.py \
#   --video_path /scratch_net/ken/radjoe/training/images \
#   --label_path /scratch_net/ken/radjoe/training/1st_manual \
#   --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_sam/DRIVE \
#   --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml\
#   --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAM_DIR/sam2/checkpoints/sam2.1_hiera_tiny.pt \
#   --model_type image \
#   --dataset_type DRIVE

# python experiments/SAM_experiment.py \
#   --video_path /scratch_net/ken/radjoe/stare_dataset/images \
#   --label_path /scratch_net/ken/radjoe/stare_dataset/label_vk \
#   --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_sam/STARE_VK \
#   --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml\
#   --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAM_DIR/sam2/checkpoints/sam2.1_hiera_tiny.pt  \
#   --model_type image \
#   --dataset_type STARE

# python experiments/SAM_experiment.py \
#   --video_path /scratch_net/ken/radjoe/stare_dataset/images \
#   --label_path /scratch_net/ken/radjoe/stare_dataset/label_ah \
#   --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_sam/STARE_AH \
#   --model_cfg configs/sam2.1/sam2.1_hiera_t.yaml\
#   --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAM_DIR/sam2/checkpoints/sam2.1_hiera_tiny.pt  \
#   --model_type image \
#   --dataset_type STARE

# Print job status
echo "SLURM test job running..."


