#!/bin/bash

#SBATCH --output=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_hq.txt               # Standard error file
#SBATCH --error=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/error_hq.txt               # Standard error file
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task 
#SBATCH --mail-type=END,FAIL                # Notify on job completion or failure
#SBATCH --mail-user=adjoerachel@email.com    # Email for notifications

# Print job status
echo "SLURM test job running..."



# Activate your virtual environment if using one
source /scratch_net/ken/radjoe/conda/bin/activate /scratch_net/ken/radjoe/conda_envs/sam_hq2

# Inference script for GLI for bboxes and Points
python experiments/HQSAM_experiment.py \
  --video_path /scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/images_UNN \
  --label_path /scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/labels_UNN \
  --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_hq/BraTS_SSA/MedSAM2 \
  --model_cfg configs/sam2.1/sam2.1_hq_hiera_l.yaml\
  --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/HQSAM2/sam-hq/sam-hq2/checkpoints/sam2.1_hq_hiera_large.pt\
  --model_type video \
  --dataset_type MRI


python experiments/HQSAM_experiment.py \
  --video_path /scratch_net/ken/radjoe/training/images \
  --label_path /scratch_net/ken/radjoe/training/1st_manual \
  --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_hq/DRIVE \
  --model_cfg configs/sam2.1/sam2.1_hq_hiera_l.yaml\
  --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/HQSAM2/sam-hq/sam-hq2/checkpoints/sam2.1_hq_hiera_large.pt\
  --model_type image \
  --dataset_type DRIVE

python experiments/HQSAM_experiment.py \
  --video_path /scratch_net/ken/radjoe/stare_dataset/images \
  --label_path /scratch_net/ken/radjoe/stare_dataset/label_vk \
  --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_hq/STARE_VK \
  --model_cfg configs/sam2.1/sam2.1_hq_hiera_l.yaml\
  --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/HQSAM2/sam-hq/sam-hq2/checkpoints/sam2.1_hq_hiera_large.pt\
  --model_type image \
  --dataset_type STARE

python experiments/HQSAM_experiment.py \
  --video_path /scratch_net/ken/radjoe/stare_dataset/images \
  --label_path /scratch_net/ken/radjoe/stare_dataset/label_ah \
  --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results_hq/STARE_AH \
  --model_cfg configs/sam2.1/sam2.1_hq_hiera_l.yaml\
  --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/HQSAM2/sam-hq/sam-hq2/checkpoints/sam2.1_hq_hiera_large.pt\
  --model_type image \
  --dataset_type STARE

# Print job status

echo "SLURM test job running..."


