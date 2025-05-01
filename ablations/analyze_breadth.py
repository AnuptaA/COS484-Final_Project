#!/usr/bin/env python

import os
import sys
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

def generate_averages_for_captions(results_dir):
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
        df = pd.read_csv(csv_path)
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
    
def generate_heatmap_of_differences(results_dir):
    cols = {
        'Quantity': 'sim_quantity',
        'Location': 'sim_location',
        'Object': 'sim_object',
        'Gender-Number': 'sim_gender-number',
        'Gender': 'sim_gender',
        'Full': 'sim_full'
    }

    averages_matrix = pd.read_csv(AVERAGES_MATRIX_PATH, index_col=0)
    differences_matrix = pd.DataFrame(index=averages_matrix.index, columns=cols)

    for model_name in averages_matrix.index:
        for col, sim_col in cols.items():
            differences_matrix.loc[model_name, col] = averages_matrix.loc[model_name, sim_col] - averages_matrix.loc[model_name, 'sim_original']

    differences_matrix = differences_matrix.astype(float)

    plt.figure(figsize=(10, 6))
    sns.heatmap(differences_matrix, annot=True, cmap='coolwarm', center=0, fmt=".3f")

    plt.title("Similarity Difference from True Caption by Model and Underspecification Type", pad=20, fontsize=12)
    plt.xlabel("Caption Underspecification Type", labelpad=15, fontsize=12)
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
    generate_averages_for_captions(OUTPUT_DIRECTORY)
    generate_heatmap_of_differences(OUTPUT_DIRECTORY)

#----------------------------------------------------------------------#

if __name__ == '__main__':
    main()