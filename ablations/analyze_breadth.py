#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------------------------------#
########################################################################

CSV_DIRECTORIES = {
    'clip': 'CLIP_results/ViT-B-16_UND_scores_100_samples.csv',
    'clip-quickgelu': 'CLIP_results/ViT-B-16-quickgelu_UND_scores_100_samples.csv',
    'siglip': 'SigLIP_results/ViT-B-16-SigLIP_UND_scores_100_samples.csv',
    'siglip2': 'SigLIP2_results/ViT-B-16-SigLIP2_UND_scores_100_samples.csv',
    'radio': 'RADIO_results/radio_v2.5-b_UND_scores_100_samples.csv'
}

AVERAGES_MATRIX_PATH = 'analysis/breadth_caption_averages.csv'
OUTPUT_DIRECTORY = 'analysis'

########################################################################
#----------------------------------------------------------------------#

def generate_averages_for_captions(exp_name, results_dir):
    sim_cols = [
        'sim_original',
        'sim_quantity',
        'sim_location',
        'sim_object',
        'sim_gender-number',
        'sim_gender',
        'sim_full'
    ]

    averages_matrix = pd.DataFrame(index=CSV_DIRECTORIES.keys(), columns=sim_cols)

    for model_name, csv_path in CSV_DIRECTORIES.items():
        df = pd.read_csv(os.path.join(f"{exp_name}_results", csv_path))
        for col in sim_cols:
            vals = df[col].dropna()
            vals = vals[vals != 0]
            mean_val = vals.mean() if not vals.empty else float('nan')
            averages_matrix.loc[model_name, col] = mean_val

    averages_matrix = averages_matrix.astype(float)

    out_csv = os.path.join(results_dir, 'breadth_caption_averages.csv')
    averages_matrix.to_csv(out_csv)
    print(f"Saved matrix to {out_csv}")

#----------------------------------------------------------------------#
    
def generate_heatmap_of_differences(exp_name, avg_mat_path, results_dir):
    cols = {
        'Quantity': 'sim_quantity',
        'Location': 'sim_location',
        'Object': 'sim_object',
        'Gender-Number': 'sim_gender-number',
        'Gender': 'sim_gender',
        'Full': 'sim_full'
    }

    averages_matrix = pd.read_csv(avg_mat_path, index_col=0)
    differences_matrix = pd.DataFrame(index=averages_matrix.index, columns=cols)

    for model_name in averages_matrix.index:
        for col, sim_col in cols.items():
            differences_matrix.loc[model_name, col] = averages_matrix.loc[model_name, sim_col] - averages_matrix.loc[model_name, 'sim_original']

    differences_matrix = differences_matrix.astype(float)

    plt.figure(figsize=(10, 6))
    sns.heatmap(differences_matrix, annot=True, cmap='coolwarm', center=0, fmt=".3f")

    if exp_name == "inc":
        exp_title = "Incorrectness"
    else:
        exp_title = "Underspecification"

    plt.title(f"Mean Similarity Difference from True Caption by Model and {exp_title} Type", pad=20, fontsize=12)
    plt.xlabel(f"Caption {exp_title} Type", labelpad=15, fontsize=12)
    plt.ylabel("Model", labelpad=15, fontsize=12)

    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    plt.tight_layout()

    out_png = os.path.join(results_dir, 'breadth_differences_from_original_heatmap.png')
    plt.savefig(out_png)
    plt.close()
    print(f"Saved heatmap to {out_png}")
    
#----------------------------------------------------------------------#

def main():
    desc = "Helper module for analyzing CSVs with heatmap"
    help_exp = "the proof of concept experiment to be analyzed"

    parser = ArgumentParser(prog=f'{sys.argv[0]}', description=desc)
    parser.add_argument('exp', type=str.lower, choices=['und', 'inc'], help=help_exp)
    
    args = vars(parser.parse_args())
    exp = args.get('exp')

    output_dir = exp + "_analysis"
    avg_mat_path = exp + "_analysis/breadth_caption_averages.csv"

    print(f"Generating breadth heatmap for {exp} proof of concept.")
    generate_averages_for_captions(exp, output_dir)
    generate_heatmap_of_differences(exp, avg_mat_path, output_dir)

#----------------------------------------------------------------------#

if __name__ == '__main__':
    main()