#!/usr/bin/env python

import torch
from PIL import Image
import open_clip
import json
import os
import sys
import csv
from argparse import ArgumentParser
from scipy import spatial
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from torch.nn import functional as F

#-----------------------------------------------------------------------
########################################################################

MODEL_PARAMS = {
    'siglip': {
        'MODELS': ['ViT-B-16-SigLIP', 
          'ViT-B-16-SigLIP-256', 
          'ViT-B-16-SigLIP-i18n-256', 
          'ViT-B-16-SigLIP-384', 
          'ViT-B-16-SigLIP-512'],
        'OUTPUT_DIRECTORY': 'SigLIP_results'
    },
    'siglip2': {
        'MODELS': ['ViT-B-16-SigLIP2',
          'ViT-B-16-SigLIP2-256', 
          'ViT-B-16-SigLIP2-384',
          'ViT-B-16-SigLIP2-512'],
        'OUTPUT_DIRECTORY': 'SigLIP2_results'
    },
    'radio': {
        'MODELS': ['radio_v2.5-b'],
        'OUTPUT_DIRECTORY': 'RADIO_results'
    }
}
PRETRAINED_DATASET = 'webli'
SAMPLE_CSV_PATH = '100samples_with_FULL.csv'

########################################################################
#-----------------------------------------------------------------------

def initialize_and_get_model(model_name, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=dataset)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    return model, tokenizer, preprocess, device

#-----------------------------------------------------------------------

def compute_sim_score(caption, image_features, model, tokenized_captions):
    if caption == 'NA':
        return 0.0
    
    with torch.no_grad():
        caption_features = model.encode_text(tokenized_captions).cpu().numpy()[0]

    return max(float((1 - spatial.distance.cosine(image_features, caption_features)) * 2.5), 0)
    # return 100 * max(1 - spatial.distance.cosine(image_features, text_features), 0)

#-----------------------------------------------------------------------

def compute_und_scores(model_name, model, tokenizer, preprocess, device, csv_path, out_dir):
    # read csv contents
    with open(csv_path, 'r') as f:
        content = f.readlines()

    # output path
    output_path = f'{out_dir}/{model_name}_UND_scores_100_samples.csv'
    with open(output_path, 'w', newline='') as myoutput:
        myoutput.write('imageURL,imageID,original,und_quantity,und_location,und_object,und_gender-number,und_gender,und_full,sim_original,sim_quantity,sim_location,sim_object,sim_gender-number,sim_gender,sim_full\n')
        writer = csv.writer(myoutput)

        # iterate through each line of csv (skip first line of col names)
        for c, l in enumerate(content):
            if c == 0:
                continue

            print(f"Processing line {c}: {l}")
            l = l.split(';')

            # get image
            imageURL = str(l[0])
            imageID = str(l[1])

            try:
                image = preprocess(Image.open(imageURL)).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error loading {imageURL}: {e}")
                continue

            # get captions
            orig_caption = str(l[2])
            und_some = str(l[3])
            und_locative = str(l[4])
            und_demonstrative = str(l[5])
            und_they = str(l[6])
            und_person = str(l[7])
            und_full = str(l[8])

            # get tokens
            text_O_tokens = tokenizer([orig_caption]).to(device)
            text_some_tokens = tokenizer([und_some]).to(device)
            text_loc_tokens = tokenizer([und_locative]).to(device)
            text_dem_tokens = tokenizer([und_demonstrative]).to(device)
            text_they_tokens = tokenizer([und_they]).to(device)
            text_person_tokens = tokenizer([und_person]).to(device)
            text_full_tokens = tokenizer([und_full]).to(device)


            with torch.no_grad():
                # get encoded image
                image_features = model.encode_image(image).cpu().numpy()[0]

                # similarity scores between encoded image and captions
                sim0 = compute_sim_score(orig_caption, image_features, model, text_O_tokens)
                sim1 = compute_sim_score(und_some, image_features, model, text_some_tokens)
                sim2 = compute_sim_score(und_locative, image_features, model, text_loc_tokens)
                sim3 = compute_sim_score(und_demonstrative, image_features, model, text_dem_tokens)
                sim4 = compute_sim_score(und_they, image_features, model, text_they_tokens)
                sim5 = compute_sim_score(und_person, image_features, model, text_person_tokens)
                sim6 = compute_sim_score(und_full, image_features, model, text_full_tokens)

            writer.writerow([
                imageURL,
                imageID,
                orig_caption, 
                und_some,
                und_locative,
                und_demonstrative,
                und_they,
                und_person,
                und_full,
                sim0,
                sim1,
                sim2,
                sim3,
                sim4,
                sim5,
                sim6
            ])

    print(f"Finished PoC1 for {model_name}")
    print()

#-----------------------------------------------------------------------

def run_model(model_name):
    model_info = MODEL_PARAMS[model_name]
    for sub_model in model_info['MODELS']:
        model, tokenizer, preprocess, device = initialize_and_get_model(sub_model, PRETRAINED_DATASET)
        compute_und_scores(sub_model, model, tokenizer, preprocess, device, SAMPLE_CSV_PATH, model_info['OUTPUT_DIRECTORY'])

#-----------------------------------------------------------------------

def main():
    desc = "Helper module for running models to generate CSVs in the same"
    desc += " format as main_PoC1.py"
    model_help = "the model whose results are being generated"

    parser = ArgumentParser(prog=f'{sys.argv[0]}', description=desc)
    parser.add_argument('model', type=str.lower, choices=['siglip', 'siglip2'], help=model_help)

    args = vars(parser.parse_args())
    model = args.get('model')

    print(f'Your selected model is {model}.')
    run_model(model)
    
#-----------------------------------------------------------------------

if __name__ == '__main__':
    main()