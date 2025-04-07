#!/usr/bin/env python

from B_16 import initialize_and_get_model, compute_und_scores

#----------------------------------------------------------------------#
########################################################################

MODELS = ['ViT-B-16-SigLIP2',
          'ViT-B-16-SigLIP2-256', 
          'ViT-B-16-SigLIP2-384',
          'ViT-B-16-SigLIP2-512']

PRETRAINED_DATASET = 'webli'
SAMPLE_CSV_PATH = '100samples_with_FULL.csv'
OUTPUT_DIRECTORY = 'SigLIP2_results'

########################################################################
#----------------------------------------------------------------------#

if __name__ == '__main__':
    for model_name in MODELS:
        model, tokenizer, preprocess, device = initialize_and_get_model(model_name, PRETRAINED_DATASET)
        compute_und_scores(model_name, model, tokenizer, preprocess, device, SAMPLE_CSV_PATH, OUTPUT_DIRECTORY)

# Model 108: ('ViT-B-16-SigLIP2', 'webli')
# Model 109: ('ViT-B-16-SigLIP2-256', 'webli')
# Model 110: ('ViT-B-16-SigLIP2-384', 'webli')
# Model 111: ('ViT-B-16-SigLIP2-512', 'webli')