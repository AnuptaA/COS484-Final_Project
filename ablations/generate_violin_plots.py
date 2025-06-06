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
    'clip': 'CLIP_results',
    'siglip': 'SigLIP_results',
    'siglip2': 'SigLIP2_results',
    'radio': 'RADIO_results'
}

OUTPUT_DIRECTORY = 'analysis'

########################################################################
#----------------------------------------------------------------------#
    
def get_csv_filenames(input_path):
    file_names = os.listdir(input_path)
    return [(f'{input_path}/{file_name}', file_name[:-27]) for file_name in file_names if file_name.endswith('.csv')]
    
#----------------------------------------------------------------------#

def generate_violin_plots(model_name, exp_name, results_dir):
    input_dir = os.path.join(f"{exp_name}_results", CSV_DIRECTORIES[model_name])
    out_dir = os.path.join(results_dir, model_name)
    file_paths = get_csv_filenames(input_dir)

    if exp_name == "inc":
        exp_title = "Incorrectness"
    else:
        exp_title = "Underspecification"

    sim_cols = [
        'sim_original',
        'sim_quantity',
        'sim_location',
        'sim_object',
        'sim_gender-number',
        'sim_gender',
        'sim_full'
    ]

    for csv_path, name in file_paths:
        df = pd.read_csv(csv_path)

        means = {}
        for col in sim_cols:
            vals = df[col].dropna()
            vals = vals[vals != 0]
            # means[col] = vals.mean() if not vals.empty else float('nan') # if we want to display means somehow

        melted = df[sim_cols].melt(var_name='type', value_name='score')
        melted = melted.dropna()
        melted = melted[melted['score'] != 0]

        plt.figure(figsize=(8, 5))
        palette = sns.color_palette("viridis", len(sim_cols))

        sns.violinplot(x='type', y='score', hue='type', data=melted, palette=palette, legend=False)
        plt.title(f'Violin Plot of Similarity Scores for {exp_title} — {name}')
        plt.ylabel('CLIPScore')
        plt.xlabel('')
        plt.ylim(-0.1, 1.4)
        plt.xticks(rotation=30, ha='right')

        plt.tight_layout()

        out_png = os.path.join(out_dir, f'{name}_violin.png')
        plt.savefig(out_png)
        plt.close()

    print(f"\nPlots and summaries have been written to '{out_dir}/'.\n")


#----------------------------------------------------------------------#

def main():
    desc = "Helper module for analyzing output CSVs"
    help_model = "the model whose results are to be analyzed"
    help_exp = "the proof of concept experiment to be analyzed"

    parser = ArgumentParser(prog=f'{sys.argv[0]}', description=desc)
    parser.add_argument('exp', type=str.lower, choices=['und', 'inc'], help=help_exp)
    parser.add_argument('model', type=str.lower, choices=['clip', 'siglip', 'siglip2', 'radio'], help=help_model)
    
    args = vars(parser.parse_args())
    exp = args.get('exp')
    model = args.get('model')

    output_dir = exp + "_analysis"

    print(f'Your selected model is {model} for the {exp} proof of concept.')
    generate_violin_plots(model, exp, output_dir)

#----------------------------------------------------------------------#

if __name__ == '__main__':
    main()