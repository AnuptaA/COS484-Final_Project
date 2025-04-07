#!/usr/bin/env python

from B_16 import initialize_and_get_model, compute_und_scores

#----------------------------------------------------------------------#
########################################################################

MODELS = ['TULIP-B-16-224']

PRETRAINED_DATASET = 'webli'
SAMPLE_CSV_PATH = '100samples_with_FULL.csv'
OUTPUT_DIRECTORY = 'TULIP_results'

########################################################################
#----------------------------------------------------------------------#

if __name__ == '__main__':
    for model_name in MODELS:
        model, tokenizer, preprocess, device = initialize_and_get_model(model_name, PRETRAINED_DATASET)
        compute_und_scores(model_name, model, tokenizer, preprocess, device, SAMPLE_CSV_PATH, OUTPUT_DIRECTORY)

# TULIP-B-16-224, no pretrained checkpoint