#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------------------------------#
########################################################################

UND_CSV_DIRECTORIES = {
    'clip': 'CLIP_results/ViT-B-16_und_scores_100_samples.csv',
    'clip-quickgelu': 'CLIP_results/ViT-B-16-quickgelu_und_scores_100_samples.csv',
    'siglip': 'SigLIP_results/ViT-B-16-SigLIP_und_scores_100_samples.csv',
    'siglip2': 'SigLIP2_results/ViT-B-16-SigLIP2_und_scores_100_samples.csv',
    'radio': 'RADIO_results/radio_v2.5-b_und_scores_100_samples.csv'
}

INC_CSV_DIRECTORIES = {
    'clip': 'CLIP_results/ViT-B-16_inc_scores_100_samples.csv',
    'clip-quickgelu': 'CLIP_results/ViT-B-16-quickgelu_inc_scores_100_samples.csv',
    'siglip': 'SigLIP_results/ViT-B-16-SigLIP_inc_scores_100_samples.csv',
    'siglip2': 'SigLIP2_results/ViT-B-16-SigLIP2_inc_scores_100_samples.csv',
    'radio': 'RADIO_results/radio_v2.5-b_inc_scores_100_samples.csv'
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

    if exp_name == "inc":
        csv_dict = INC_CSV_DIRECTORIES
    else:
        csv_dict = UND_CSV_DIRECTORIES

    averages_matrix = pd.DataFrame(index=csv_dict.keys(), columns=sim_cols)

    for model_name, csv_path in csv_dict.items():
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

def generate_averages_for_und_and_inc():
    sim_cols = [
        'sim_original',
        'sim_quantity',
        'sim_location',
        'sim_object',
        'sim_gender-number',
        'sim_gender',
        'sim_full'
    ]

    und_averages = pd.DataFrame(index=UND_CSV_DIRECTORIES.keys(), columns=sim_cols)
    for model_name, csv_path in UND_CSV_DIRECTORIES.items():
        df = pd.read_csv(os.path.join(f"und_results", csv_path))
        for col in sim_cols:
            vals = df[col].dropna()
            vals = vals[vals != 0]
            mean_val = vals.mean() if not vals.empty else float('nan')
            und_averages.loc[model_name, col] = mean_val
    und_averages = und_averages.astype(float)
    out_csv = os.path.join('both_analysis', 'breadth_und_caption_averages.csv')
    und_averages.to_csv(out_csv)
    print(f"Saved und matrix to {out_csv}")

    inc_averages = pd.DataFrame(index=INC_CSV_DIRECTORIES.keys(), columns=sim_cols)
    for model_name, csv_path in INC_CSV_DIRECTORIES.items():
        df = pd.read_csv(os.path.join(f"inc_results", csv_path))
        for col in sim_cols:
            vals = df[col].dropna()
            vals = vals[vals != 0]
            mean_val = vals.mean() if not vals.empty else float('nan')
            inc_averages.loc[model_name, col] = mean_val
    inc_averages = inc_averages.astype(float)
    out_csv = os.path.join('both_analysis', 'breadth_inc_caption_averages.csv')
    inc_averages.to_csv(out_csv)
    print(f"Saved inc matrix to {out_csv}")

#----------------------------------------------------------------------#
    
def generate_heatmap_of_und_and_inc():
    cols = {
        'Quantity': 'sim_quantity',
        'Location': 'sim_location',
        'Object': 'sim_object',
        'Gender-Number': 'sim_gender-number',
        'Gender': 'sim_gender',
        'Full': 'sim_full'
    }

    und_averages = pd.read_csv(os.path.join('both_analysis', 'breadth_und_caption_averages.csv'), index_col=0)
    inc_averages = pd.read_csv(os.path.join('both_analysis', 'breadth_inc_caption_averages.csv'), index_col=0)
    differences_matrix = pd.DataFrame(index=und_averages.index, columns=cols)

    for model_name in und_averages.index:
        for col, sim_col in cols.items():
            differences_matrix.loc[model_name, col] = und_averages.loc[model_name, sim_col] - inc_averages.loc[model_name, sim_col]

    differences_matrix = differences_matrix.astype(float)

    plt.figure(figsize=(10, 6))
    sns.heatmap(differences_matrix, annot=True, cmap='coolwarm', center=0, fmt=".3f")

    plt.title(f"Mean Similarity Difference of Underspecification and Incorrectness by Model and Alteration Type", pad=20, fontsize=12)
    plt.xlabel(f"Caption Alteration Type", labelpad=15, fontsize=12)
    plt.ylabel("Model", labelpad=15, fontsize=12)

    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    plt.tight_layout()

    out_png = os.path.join('both_analysis', 'breadth_und_and_inc_differences_heatmap.png')
    plt.savefig(out_png)
    plt.close()
    print(f"Saved heatmap to {out_png}")

#----------------------------------------------------------------------#
    
def generate_plot(exp_name, avg_mat_path, results_dir):
    cols = {
        'sim_original': 'Original',
        'sim_quantity': 'Quantity',
        'sim_location': 'Location',
        'sim_object': 'Object',
        'sim_gender-number': 'Gender-Number',
        'sim_gender': 'Gender',
        'sim_full': 'Full'
    }

    averages_matrix = pd.read_csv(avg_mat_path, index_col=0)

    plt.figure(figsize=(10, 6))
    for model_name in averages_matrix.index:
        averages = averages_matrix.loc[model_name]
        plt.plot([cols[col] for col in cols], averages, marker='o', label=model_name)

    if exp_name == "inc":
        exp_title = "Incorrectness"
    else:
        exp_title = "Underspecification"

    plt.title(f"Average Similarity Score by Caption for all Model Families for {exp_title} Proof of Concept")
    plt.xlabel(f"{exp_title} Type")
    plt.ylabel("Average Similarity Score")
    plt.ylim(0, 1.1)
    plt.legend(title="Model")
    plt.grid(True)
    plt.tight_layout()

    out_png = os.path.join(results_dir, f'breadth_similarity_for_all.png')
    plt.savefig(out_png)
    plt.close()
    print(f"Saved lineplot to {out_png}")
    
#----------------------------------------------------------------------#

def main():
    desc = "Helper module for analyzing CSVs with heatmap"
    help_exp = "the proof of concept experiment to be analyzed"
    help_out = "the plot being generated"

    parser = ArgumentParser(prog=f'{sys.argv[0]}', description=desc)
    parser.add_argument('exp', type=str.lower, choices=['und', 'inc', 'both'], help=help_exp)
    parser.add_argument('out', type=str.lower, choices=['plot', 'map'], help=help_out)
    
    args = vars(parser.parse_args())
    exp = args.get('exp')
    out = args.get('out')

    output_dir = exp + "_analysis"
    avg_mat_path = exp + "_analysis/breadth_caption_averages.csv"

    print(f"Generating breadth {out} for {exp} proof of concept.")

    if out == "plot":
        if exp == "both":
            generate_averages_for_captions("und", "und_analysis")
            generate_plot("und", "und_analysis/breadth_caption_averages.csv", "und_analysis")
            generate_averages_for_captions("inc", "inc_analysis")
            generate_plot("inc", "inc_analysis/breadth_caption_averages.csv", "inc_analysis")
        else:
            generate_averages_for_captions(exp, output_dir)
            generate_plot(exp, avg_mat_path, output_dir)
    elif exp == "both":
        generate_averages_for_und_and_inc()
        generate_heatmap_of_und_and_inc()
    else:
        generate_averages_for_captions(exp, output_dir)
        generate_heatmap_of_differences(exp, avg_mat_path, output_dir)

#----------------------------------------------------------------------#

if __name__ == '__main__':
    main()