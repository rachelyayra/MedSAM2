#!/bin/bash
#SBATCH --output=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_logs/output_unettrain.txt               # Standard error file
#SBATCH --error=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_logs/error_unettrain.txt               # Standard error file
#SBATCH --gres=gpu:2
#SBATCH --ntasks=4                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task 
#SBATCH --mail-type=END,FAIL                # Notify on job completion or failure
#SBATCH --mail-user=adjoerachel@email.com    # Email for notifications

# Print job status
echo "SLURM test job running..."

module load miniconda3
module load python/3.10.16


# Activate your virtual environment if using one
source /scratch_net/ken/radjoe/conda/bin/activate /scratch_net/ken/radjoe/conda_envs/medsam2

# Run your inference script
python /scratch_net/ken/radjoe/Projects/Experiments/MedSAM2/training/train.py -c configs/adapter_Unet_BraTS.yaml --use-cluster 0  --num-gpus 2

# Print job status
echo "SLURM test job running..."