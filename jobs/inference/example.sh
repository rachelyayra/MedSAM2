#!/bin/bash

#SBATCH --output=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_med.txt               # Standard error file
#SBATCH --error=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/error_med.txt               # Standard error file
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task 
#SBATCH --mail-type=END,FAIL                # Notify on job completion or failure
#SBATCH --mail-user=adjoerachel@email.com    # Email for notifications

# Print job status
echo "SLURM test job running..."



source /scratch_net/ken/radjoe/conda/etc/profile.d/conda.sh
conda activate /scratch_net/ken/radjoe/conda_envs/medsam2

python experiments/MedSAM_experiment.py \
  --video_path /scratch_net/ken/radjoe/BraTS/Test/images_UNN \
  --label_path /scratch_net/ken/radjoe/BraTS/Test/labels_UNN \
  --save_path /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/TT_NA_SSA \
  --model_cfg configs/sam_sofmax.yaml \
  --checkpoint /scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/logs/decoder_only/checkpoint_210.pt \
  --model_type video