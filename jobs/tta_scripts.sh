#!/bin/bash

sbatch jobs/inference/example.sh

echo "Evaluation of results"

conda init
conda activate 

chmod +x jobs/evaluation/evaluate_examples.sh

./jobs/evaluation/evaluate_examples.sh

echo "Done Evaluation of inference
Starting TTA
"

sbatch jobs/testing/single_sample_tta.sh
echo "
Starting TTA Evaluation 
"
chmod +x jobs/evaluation/evaluate_tta_examples.sh
./jobs/evaluation/evaluate_tta_examples.sh

echo "
DONE
"