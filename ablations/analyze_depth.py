#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------#
########################################################################

CSV_DIRECTORIES = {
    'clip': {
        'ViT-B-16': 'CLIP_results/ViT-B-16_UND_scores_100_samples.csv',
        'ViT-L-14': 'CLIP_results/ViT-L-14_UND_scores_100_samples.csv',
    },
    'clip-quickgelu': {
        'ViT-B-16-quickgelu': 'CLIP_results/ViT-B-16-quickgelu_UND_scores_100_samples.csv',
        'ViT-L-14-quickgelu': 'CLIP_results/ViT-L-14-quickgelu_UND_scores_100_samples.csv',
    },
    'siglip': {
        'ViT-B-16-SigLIP-384': 'SigLIP_results/ViT-B-16-SigLIP-384_UND_scores_100_samples.csv',
        'ViT-L-16-SigLIP-384': 'SigLIP_results/ViT-L-16-SigLIP-384_UND_scores_100_samples.csv',
        'ViT-SO400M-14-SigLIP-384': 'SigLIP_results/ViT-SO400M-14-SigLIP-384_UND_scores_100_samples.csv',
    },
    'siglip2': {
        'ViT-B-16-SigLIP2-384': 'SigLIP2_results/ViT-B-16-SigLIP2-384_UND_scores_100_samples.csv',
        'ViT-L-16-SigLIP2-384': 'SigLIP2_results/ViT-L-16-SigLIP2-384_UND_scores_100_samples.csv',
        'ViT-SO400M-16-SigLIP2-384': 'SigLIP2_results/ViT-SO400M-16-SigLIP2-384_UND_scores_100_samples.csv',
    },
    'radio': {
        'radio_v2.5-b': 'RADIO_results/radio_v2.5-b_UND_scores_100_samples.csv',
        'radio_v2.5-g': 'RADIO_results/radio_v2.5-g_UND_scores_100_samples.csv',
        'radio_v2.5-h': 'RADIO_results/radio_v2.5-h_UND_scores_100_samples.csv',
        'radio_v2.5-l': 'RADIO_results/radio_v2.5-l_UND_scores_100_samples.csv',
    }
}

OUTPUT_DIRECTORY = 'analysis'

#----------------------------------------------------------------------#

def generate_depth_lineplots(model_family, results_dir):
    csv_paths = CSV_DIRECTORIES[model_family]

    cols = {
        'sim_original': 'Original',
        'sim_quantity': 'Quantity',
        'sim_location': 'Location',
        'sim_object': 'Object',
        'sim_gender-number': 'Gender-Number',
        'sim_gender': 'Gender',
        'sim_full': 'Full'
    }

    averages_matrix = pd.DataFrame(index=csv_paths.keys(), columns=cols.keys())

    for model_name, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        for col in cols.keys():
            vals = df[col].dropna()
            vals = vals[vals != 0]
            mean_val = vals.mean() if not vals.empty else float('nan')
            averages_matrix.loc[model_name, col] = mean_val

    plt.figure(figsize=(10, 6))
    for model_name in averages_matrix.index:
        averages = averages_matrix.loc[model_name]
        plt.plot([cols[col] for col in cols], averages, marker='o', label=model_name)

    plt.title(f"Average Similarity Score by Caption for {model_family}")
    plt.xlabel("Caption Type")
    plt.ylabel("Average Similarity Score")
    plt.legend(title="Model")
    plt.grid(True)
    plt.tight_layout()

    out_png = os.path.join(results_dir, f'depth_similarity_for_{model_family}.png')
    plt.savefig(out_png)
    plt.close()
    print(f"Saved lineplot to {out_png}")

#----------------------------------------------------------------------#

def main():
    desc = "Helper module for analyzing CSVs with line plots"
    help = "the family whose results are to be analyzed"

    parser = ArgumentParser(prog=f'{sys.argv[0]}', description=desc)
    parser.add_argument('family', type=str.lower, choices=['clip', 'clip-quickgelu', 'siglip', 'siglip2', 'radio'], help=help)
    
    args = vars(parser.parse_args())
    family = args.get('family')

    print(f"Generating depth line plot for family: {family}")
    generate_depth_lineplots(family, OUTPUT_DIRECTORY)

#----------------------------------------------------------------------#

if __name__ == '__main__':
    main()
