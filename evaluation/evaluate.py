# Import Relevant Libraries
import os
import sys
sys.path.append('/scratch_net/ken/radjoe/Projects/Experiments/SAMEXP/')

import numpy as np
from evaluation.evaluation_func import evaluate_retinal, evaluate_3D, evaluate_4D, evaluate_3D_CC
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Segmentation masks")
    
    parser.add_argument('--pred_path', type=str, required=True, help='Path to the input video directory')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label directory')
    parser.add_argument('--save_path', type=str, required=True, help='Where to save predictions')
    parser.add_argument('--type', type=str, required=True, choices=['retina', 'brats', 'brats3D', 'brats3D_CC'], default='retina', help='Types of labels')

    return parser.parse_args() 

def main():
    args = parse_args()
    if args.type == 'retina':
        evaluate_retinal(
        pred_path = args.pred_path, 
        label_path = args.label_path, 
        savepath = args.save_path
        )
    elif args.type == 'brats3D':
        evaluate_4D(
        pred_path = args.pred_path, 
        label_path = args.label_path, 
        savepath = args.save_path
        )
    elif args.type == 'brats3D_CC':
        evaluate_3D_CC(
        pred_path = args.pred_path, 
        label_path = args.label_path, 
        savepath = args.save_path
        )
    else:
        evaluate_3D(
        pred_path = args.pred_path, 
        label_path = args.label_path, 
        savepath = args.save_path
        )
if __name__ == "__main__":
   main()