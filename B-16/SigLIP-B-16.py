#!/usr/bin/env python

import torch
from PIL import Image
import open_clip
import json
import os
import csv

#-----------------------------------------------------------------------

MODELS = ['ViT-B-16-SigLIP', 
          'ViT-B-16-SigLIP-256', 
          'ViT-B-16-SigLIP-i18n-256', 
          'ViT-B-16-SigLIP-384', 
          'ViT-B-16-SigLIP-512']

PRETRAINED_DATASET = 'webli'

#-----------------------------------------------------------------------

def initialize_and_get_model(model_name, dataset=PRETRAINED_DATASET):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, dataset)

#-----------------------------------------------------------------------


#-----------------------------------------------------------------------

    



# pretrained_models = open_clip.list_pretrained()

# print(f"There are {len(pretrained_models)} models available")
# for i, model in enumerate(pretrained_models):
#     print(f"Model {i}: {model}")

# Model 96: ('ViT-B-16-SigLIP', 'webli')
# Model 97: ('ViT-B-16-SigLIP-256', 'webli')
# Model 98: ('ViT-B-16-SigLIP-i18n-256', 'webli')
# Model 99: ('ViT-B-16-SigLIP-384', 'webli')
# Model 100: ('ViT-B-16-SigLIP-512', 'webli')