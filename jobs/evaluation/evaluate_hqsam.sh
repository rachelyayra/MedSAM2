#!/bin/bash

# Arrays of paths
PRED_PATHS=(
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/DRIVE/boxes"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/DRIVE/points"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_AH/boxes"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_AH/points"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_VK/boxes"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_VK/points"
    "/scratch/radjoe/Projects/Experiments/SAMEXP/results_hqmed/BraTS_SSA/MedSAM2/boxes"
    "/scratch/radjoe/Projects/Experiments/SAMEXP/results_hqmed/BraTS_SSA/MedSAM2/points"
)

LABEL_PATHS=(
    # "/scratch_net/ken/radjoe/training/1st_manual"
    # "/scratch_net/ken/radjoe/training/1st_manual"
    # "/scratch_net/ken/radjoe/stare_dataset/label_ah"
    # "/scratch_net/ken/radjoe/stare_dataset/label_ah"
    # "/scratch_net/ken/radjoe/stare_dataset/label_vk"
    # "/scratch_net/ken/radjoe/stare_dataset/label_vk"
    "/scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/labels_UNN"
    "/scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/labels_UNN"
)

SAVE_PATHS=(
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/DRIVE/drive_boxes.csv"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/DRIVE/drive_points.csv"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_AH/stare_boxes.csv"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_AH/stare_points.csv"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_VK/stare_boxes.csv"
    # "/scratch/radjoe/Projects/Experiments/SAMEXP/results_med/STARE_VK/stare_points.csv"
    "/scratch/radjoe/Projects/Experiments/SAMEXP/results_hqmed/BraTS_SSA/MedSAM2/SSA_boxes.csv"
    "/scratch/radjoe/Projects/Experiments/SAMEXP/results_hqmed/BraTS_SSA/MedSAM2/SSA_points.csv"
)

for i in ${!PRED_PATHS[@]}; do
  echo "Running evaluation for index $i..."
  python evaluation/evaluate.py \
    --pred_path "${PRED_PATHS[$i]}" \
    --label_path "${LABEL_PATHS[$i]}" \
    --save_path "${SAVE_PATHS[$i]}" \
    --type brats
done