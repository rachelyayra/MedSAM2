#!/bin/bash
#SBATCH --output=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_logs/tta_val_1_output.txt               # Standard error file
#SBATCH --error=/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/output_logs/tta_val_1_error.txt               # Standard error file
#SBATCH --gres=gpu:a6000:2
#SBATCH --ntasks=4                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task 
#SBATCH --mail-type=END,FAIL                # Notify on job completion or failure
#SBATCH --mail-user=adjoerachel@email.com    # Email for notifications

# Print job status
echo "SLURM test job running..."


echo "Current branch: $pwd"

# Optional: switch to the branch you want
branch_to_use="inference"
git checkout $branch_to_use

echo "Current branch: $(git branch --show-current)"
export PYTHONPATH="/scratch_net/ken/radjoe/Projects/SourceCode/MedSAM2:$PYTHONPATH"

# Activate your virtual environment if using one
source /scratch_net/ken/radjoe/conda/bin/activate /scratch_net/ken/radjoe/conda_envs/medsam2

# Run your inference script
python /scratch_net/ken/radjoe/Projects/SourceCode/MedSAM2/training/train.py -c configs/test_time_adaptation.yaml --use-cluster 0  --num-gpus 1

# Print job status
echo "SLURM test job running..."