#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser
import pandas as pd
import numpy as np
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
    return [(f'{input_path}/{file_name}', file_name[:-4]) for file_name in file_names if file_name.endswith('.csv')]
    
#----------------------------------------------------------------------#

def analyze_model(model_name, results_dir):
    print("Analyzing model")

    input_path = f'{CSV_DIRECTORIES[model_name]}/'
    file_paths = get_csv_filenames(input_path)

    for csv_path, file_name in file_paths:
        analysis_path = f'{results_dir}/{file_name}_results'
        sim_scores = pd.read_csv(csv_path).iloc[:,-7:]

        violin_plot = sns.violinplot(data=sim_scores)
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