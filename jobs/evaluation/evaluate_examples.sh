#!/bin/bash
PRED_PATHS=(
    # '/scratch-second/TTA_results/val_combined_decoder_lr4/logs/BraTS_GLI/'
    # '/scratch-second/TTA_results/val_combined_decoder_lr4_augs/logs/BraTS_GLI/'
    # '/scratch-second/TTA_results/val_combined_decoder_lr5/logs/BraTS_GLI/'
    # '/scratch-second/TTA_results/val_combined_decoder_lr5_augs/logs/BraTS_GLI/'
    # '/scratch-second/TTA_results/val_combined_decoder_lr6/logs/BraTS_GLI/'
    # '/scratch-second/TTA_results/val_combined_decoder_lr6_augs/logs/BraTS_GLI/'

    # '/scratch-second/TTA_results/val_combined_first_layer_lr4/logs/BraTS_GLI/'
    # '/scratch-second/TTA_results/val_combined_first_layer_lr4_augs/logs/BraTS_GLI/'
    # '/scratch-second/TTA_results/val_combined_first_layer_lr5/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_first_layer_l5_augs/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_first_layer_lr6/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_first_layer_lr6_augs/logs/BraTS_GLI/'

    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr4/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr4_augs/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr5/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr5_augs/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr6/logs/BraTS_GLI/'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr6_augs/logs/BraTS_GLI/'

    "/scratch/radjoe/Projects/Experiments/SAMEXP/results/TT_A_SSA/boxes"
    "/scratch/radjoe/Projects/Experiments/SAMEXP/results/TT_NA_SSA/boxes"
    )

LABEL_PATHS=(
        "/scratch_net/ken/radjoe/BraTS/Test/labels_UNN"

    )

SAVE_PATHS=(
    # '/scratch-second/TTA_results/val_combined_decoder_lr4/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch-second/TTA_results/val_combined_decoder_lr4_augs/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch-second/TTA_results/val_combined_decoder_lr5/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch-second/TTA_results/val_combined_decoder_lr5_augs/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch-second/TTA_results/val_combined_decoder_lr6/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch-second/TTA_results/val_combined_decoder_lr6_augs/logs/BraTS_GLI/CCA_boxes.csv'

    '/scratch/radjoe/Projects/Experiments/SAMEXP/results/TT_A_SSA/boxes/SSA_boxes.csv'
    '/scratch/radjoe/Projects/Experiments/SAMEXP/results/TT_NA_SSA/boxes/SSA_boxes.csv'

    # '/scratch-second/TTA_results/val_combined_first_layer_lr4_augs/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch-second/TTA_results/val_combined_first_layer_lr5/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_first_layer_lr5_augs/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_first_layer_lr6/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_first_layer_lr6_augs/logs/BraTS_GLI/CCA_boxes.csv'

    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr4/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr4_augs/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr5/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr5_augs/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr6/logs/BraTS_GLI/CCA_boxes.csv'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/logs/val_combined_norm_lr6_augs/logs/BraTS_GLI/CCA_boxes.csv'

)    


for i in ${!PRED_PATHS[@]}; do
  echo "Running evaluation for index $i..."
  python evaluation/evaluate.py \
    --pred_path "${PRED_PATHS[$i]}" \
    --label_path "${LABEL_PATHS[0]}" \
    --save_path "${SAVE_PATHS[$i]}" \
    --type brats
done