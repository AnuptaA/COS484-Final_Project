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
    'siglip': 'SigLIP_results',
    'siglip2': 'SigLIP2_results',
    # 'tulip': 'TULIP_results'
}

OUTPUT_DIRECTORY = 'analysis'

########################################################################
#----------------------------------------------------------------------#
    
def get_csv_filenames(input_path):
    file_names = os.listdir(input_path)
    return [(f'{input_path}/{file_name}', file_name[:-27]) for file_name in file_names if file_name.endswith('.csv')]
    
#----------------------------------------------------------------------#

def analyze_model(model_name, results_dir):
    print("Analyzing model")

    input_path = f'{CSV_DIRECTORIES[model_name]}/'
    file_paths = get_csv_filenames(input_path)

    for csv_path, file_name in file_paths:
        analysis_path = f'{results_dir}/{model_name}/{file_name}_results'
        sim_scores = pd.read_csv(csv_path).iloc[:,-7:]

        plt.figure(figsize=(10, 6))
        violin_plot = sns.violinplot(data=sim_scores)
        plt.title(f'Violin Plot of Similarity Scores - {file_name}')
        plt.ylabel('CLIPScore')
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.2)

        fig = violin_plot.get_figure()
        fig.savefig(f'{analysis_path}.png')

#----------------------------------------------------------------------#

def main():
    desc = "Helper module for analyzing output CSVs"
    help = "the model whose results are to be analyzed"

    parser = ArgumentParser(prog=f'{sys.argv[0]}', description=desc)
    parser.add_argument('model', type=str.lower, choices=['siglip', 'siglip2', 'tulip'], help=help)
    
    args = vars(parser.parse_args())
    model = args.get('model')

    print(f'Your selected model is {model}.')
    analyze_model(model, OUTPUT_DIRECTORY)

#----------------------------------------------------------------------#

if __name__ == '__main__':
    main()