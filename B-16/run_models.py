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
    'tulip': {
        'MODELS': ['TULIP-B-16-224'],
        'OUTPUT_DIRECTORY': 'TULIP_results'
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

def compute_sim_score(caption, model, tokenizer, image_features, device):
    if caption == 'NA':
        return 0.0
    tokens = tokenizer([caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens).cpu().numpy()[0]
    return max(1 - spatial.distance.cosine(image_features, text_features), 0)

#-----------------------------------------------------------------------

def compute_und_scores(model_name, model, tokenizer, preprocess, device, csv_path, out_dir):
    with open(csv_path, 'r') as f:
        content = f.readlines()

    output_path = f'{out_dir}/{model_name}_UND_scores_100_samples.csv'
    with open(output_path, 'w', newline='') as myoutput:
        myoutput.write('imageURL,imageID,original,und_quantity,und_location,und_object,und_gender-number,und_gender,und_full,sim_original,sim_quantity,sim_location,sim_object,sim_gender-number,sim_gender,sim_full\n')
        writer = csv.writer(myoutput)

        for c, l in enumerate(content):
            if c == 0:
                continue

            print(f"Processing line {c}")
            l = l.strip().split(';')

            imageURL = l[0]
            imageID = l[1]
            orig_caption = l[2]
            und_some = l[3]
            und_locative = l[4]
            und_demonstrative = l[5]
            und_they = l[6]
            und_person = l[7]
            und_full = l[8]

            try:
                image = preprocess(Image.open(imageURL)).unsqueeze(0).to(device)
            except Exception as e:
                print(f"Error loading {imageURL}: {e}")
                continue

            with torch.no_grad():
                image_features = model.encode_image(image).cpu().numpy()[0]

                sim0 = compute_sim_score(orig_caption, model, tokenizer, image_features, device)
                sim1 = compute_sim_score(und_some, model, tokenizer, image_features, device)
                sim2 = compute_sim_score(und_locative, model, tokenizer, image_features, device)
                sim3 = compute_sim_score(und_demonstrative, model, tokenizer, image_features, device)
                sim4 = compute_sim_score(und_they, model, tokenizer, image_features, device)
                sim5 = compute_sim_score(und_person, model, tokenizer, image_features, device)
                sim6 = compute_sim_score(und_full, model, tokenizer, image_features, device)

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
                round(sim0 * 2.5, 4),
                round(sim1 * 2.5, 4), 
                round(sim2 * 2.5, 4),
                round(sim3 * 2.5, 4), 
                round(sim4 * 2.5, 4), 
                round(sim5 * 2.5, 4), 
                round(sim6 * 2.5, 4)
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
    parser.add_argument('model', type=str.lower, choices=['siglip', 'siglip2', 'tulip'], help=model_help)

    args = vars(parser.parse_args())
    model = args.get('model')

    print(f'Your selected model is {model}.')
    run_model(model)
    
#-----------------------------------------------------------------------
    
if __name__ == '__main__':
    main()