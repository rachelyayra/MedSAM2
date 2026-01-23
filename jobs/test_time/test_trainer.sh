#!/bin/bash
#SBATCH --output=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_logs/output_test2.txt               # Standard error file
#SBATCH --error=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_logs/error_test2.txt               # Standard error file
#SBATCH --gres=gpu:2

 
#SBATCH --mail-type=END,FAIL                # Notify on job completion or failure
#SBATCH --mail-user=adjoerachel@email.com    # Email for notifications

# Print job status
echo "SLURM test job running..."

# Activate your virtual environment if using one
source /scratch_net/ken/radjoe/conda/bin/activate /scratch_net/ken/radjoe/conda_envs/medsam2

# Activate your virtual environment if using one
# Run your inference script
PYTHONPATH=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/ python /scratch_net/ken/radjoe/Projects/SourceCode/MedSAM2/training/train.py -c configs/test_time_adaptation.yaml --use-cluster 0 --num-gpus 2
# Print job status
echo "SLURM test job running..."