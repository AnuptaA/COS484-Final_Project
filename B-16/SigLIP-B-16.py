#!/usr/bin/env python

from B_16 import initialize_and_get_model, compute_und_scores

#----------------------------------------------------------------------#
########################################################################

MODELS = ['ViT-B-16-SigLIP', 
          'ViT-B-16-SigLIP-256', 
          'ViT-B-16-SigLIP-i18n-256', 
          'ViT-B-16-SigLIP-384', 
          'ViT-B-16-SigLIP-512']

PRETRAINED_DATASET = 'webli'
SAMPLE_CSV_PATH = '100samples_with_FULL.csv'
OUTPUT_DIRECTORY = 'SigLIP_results'

########################################################################
#----------------------------------------------------------------------#

if __name__ == '__main__':
    for model_name in MODELS:
        model, tokenizer, preprocess, device = initialize_and_get_model(model_name, PRETRAINED_DATASET)
        compute_und_scores(model_name, model, tokenizer, preprocess, device, SAMPLE_CSV_PATH, OUTPUT_DIRECTORY)

# Model 96: ('ViT-B-16-SigLIP', 'webli')
# Model 97: ('ViT-B-16-SigLIP-256', 'webli')
# Model 98: ('ViT-B-16-SigLIP-i18n-256', 'webli')
# Model 99: ('ViT-B-16-SigLIP-384', 'webli')
# Model 100: ('ViT-B-16-SigLIP-512', 'webli')