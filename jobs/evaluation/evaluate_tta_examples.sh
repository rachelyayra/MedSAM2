PRED_PATHS=(
    '/scratch/radjoe/Projects/Experiments/SAMEXP/results/Med_BraTS_GLI/boxes'
    # '/scratch/radjoe/Projects/Experiments/SAMEXP/results/Med_BraTS_SSA/boxes'
)

LABEL_PATHS=(
    # "/scratch_net/ken/radjoe/BraTS/Validation/labels_UNN"
    # /scratch_net/ken/radjoe/Preprocess_BraTS2023_SSA_Training/labels_UNN
    )

SAVE_PATHS=(
    "/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/Med_GLI_EXP_boxes.csv"
    # "/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/results/SSA_EXP_boxes.csv"
)

for i in ${!PRED_PATHS[@]}; do
  echo "Running evaluation for index $i..."
  python evaluation/evaluate.py \
    --pred_path "${PRED_PATHS[$i]}" \
    --label_path "${LABEL_PATHS[$i]}" \
    --save_path "${SAVE_PATHS[$i]}" \
    --type brats
done