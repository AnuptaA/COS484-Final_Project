#!/usr/bin/env python

import torch
from PIL import Image
import open_clip
import json
import os
import sys
import csv
from argparse import ArgumentParser
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

def get_radio_model(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_name, adaptor_names='clip', progress=True)
    model.eval().to(device)
    adaptor = model.adaptors['clip']
    tokenizer = adaptor.tokenizer
    return model, adaptor, tokenizer, device

#-----------------------------------------------------------------------

def compute_sim_score(caption, image_features, model, tokenizer, device):
    if caption == 'NA':
        return 0.0
    tokens = tokenizer([caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
    sim = F.cosine_similarity(image_features, text_features, dim=-1)
    return sim.item() if sim.item() > 0 else 0.0

#-----------------------------------------------------------------------

def compute_rad_sim_score(caption, image_summary, adaptor, tokenizer, device):
    if caption == 'NA':
        return 0.0
    tokens = tokenizer([caption]).to(device)
    with torch.no_grad():
        text_summary = adaptor.encode_text(tokens)
    sim = F.cosine_similarity(image_summary, text_summary, dim=-1)
    return sim.item() if sim.item() > 0 else 0.0

#-----------------------------------------------------------------------

def compute_und_scores(model_name, model, tokenizer, preprocess, device, csv_path, out_dir):
    # read csv contents
    with open(csv_path, 'r') as f:
        content = f.readlines()

    print(f"Computing und_scores for model {model_name}")
    print('----------------------------------------------------------')

    output_path = f'{out_dir}/{model_name}_UND_scores_100_samples.csv'
    with open(output_path, 'w', newline='') as myoutput:
        myoutput.write('imageURL,imageID,original,und_quantity,und_location,und_object,und_gender-number,und_gender,und_full,sim_original,sim_quantity,sim_location,sim_object,sim_gender-number,sim_gender,sim_full\n')
        writer = csv.writer(myoutput)
        for c, l in enumerate(content):
            if c == 0:
                continue
            l = l.split(';')
            imageURL = str(l[0])
            imageID = str(l[1])

            print(f"Processing line {c}")

            try:
                image = preprocess(Image.open(imageURL)).unsqueeze(0).to(device)
            except:
                continue

            orig_caption, und_some, und_locative, und_demonstrative, und_they, und_person, und_full = [str(x) for x in l[2:9]]

            with torch.no_grad():
                image_features = model.encode_image(image)

            sim0 = compute_sim_score(orig_caption, image_features, model, tokenizer, device)
            sim1 = compute_sim_score(und_some, image_features, model, tokenizer, device)
            sim2 = compute_sim_score(und_locative, image_features, model, tokenizer, device)
            sim3 = compute_sim_score(und_demonstrative, image_features, model, tokenizer, device)
            sim4 = compute_sim_score(und_they, image_features, model, tokenizer, device)
            sim5 = compute_sim_score(und_person, image_features, model, tokenizer, device)
            sim6 = compute_sim_score(und_full, image_features, model, tokenizer, device)

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

def compute_radio_und_scores(model_name, model, adaptor, tokenizer, device, csv_path, out_dir):
    # read csv contents
    with open(csv_path, 'r') as f:
        content = f.readlines()

    print(f"Computing und_scores for model {model_name}")
    print('----------------------------------------------------------')

    output_path = f'{out_dir}/{model_name}_UND_scores_100_samples.csv'
    with open(output_path, 'w', newline='') as myoutput:
        myoutput.write('imageURL,imageID,original,und_quantity,und_location,und_object,und_gender-number,und_gender,und_full,sim_original,sim_quantity,sim_location,sim_object,sim_gender-number,sim_gender,sim_full\n')
        writer = csv.writer(myoutput)
        for c, l in enumerate(content):
            if c == 0:
                continue
            l = l.split(';')
            imageURL = str(l[0])
            imageID = str(l[1])

            print(f"Processing line {c}")

            try:
                image = Image.open(imageURL).convert('RGB')
                image = pil_to_tensor(image).to(torch.float32).div_(255.0).unsqueeze(0).to(device)
                nearest = model.get_nearest_supported_resolution(*image.shape[-2:])
                image = F.interpolate(image, nearest, mode='bilinear', align_corners=False)
            except:
                continue

            orig_caption, und_some, und_locative, und_demonstrative, und_they, und_person, und_full = [str(x) for x in l[2:9]]

            with torch.no_grad():
                image_summary, _ = model(image)['clip']

            sim0 = compute_rad_sim_score(orig_caption, image_summary, adaptor, tokenizer, device)
            sim1 = compute_rad_sim_score(und_some, image_summary, adaptor, tokenizer, device)
            sim2 = compute_rad_sim_score(und_locative, image_summary, adaptor, tokenizer, device)
            sim3 = compute_rad_sim_score(und_demonstrative, image_summary, adaptor, tokenizer, device)
            sim4 = compute_rad_sim_score(und_they, image_summary, adaptor, tokenizer, device)
            sim5 = compute_rad_sim_score(und_person, image_summary, adaptor, tokenizer, device)
            sim6 = compute_rad_sim_score(und_full, image_summary, adaptor, tokenizer, device)

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

#-----------------------------------------------------------------------

def run_model(model_name):
    model_info = MODEL_PARAMS[model_name]
    if model_name == 'radio':
        for sub_model in model_info['MODELS']:
            model, adaptor, tokenizer, device = get_radio_model(sub_model)
            compute_radio_und_scores(sub_model, model, adaptor, tokenizer, device, SAMPLE_CSV_PATH, model_info['OUTPUT_DIRECTORY'])
    else:
        for sub_model in model_info['MODELS']:
            model, tokenizer, preprocess, device = initialize_and_get_model(sub_model, PRETRAINED_DATASET)
            compute_und_scores(sub_model, model, tokenizer, preprocess, device, SAMPLE_CSV_PATH, model_info['OUTPUT_DIRECTORY'])

#-----------------------------------------------------------------------

def main():
    desc = "Helper module for running models to generate CSVs in the same format as main_PoC1.py"
    model_help = "the model whose results are being generated"

    parser = ArgumentParser(prog=f'{sys.argv[0]}', description=desc)
    parser.add_argument('model', type=str.lower, choices=['siglip', 'siglip2', 'radio'], help=model_help)

    args = vars(parser.parse_args())
    model = args.get('model')

    print(f'Your selected model is {model}.')
    run_model(model)

#-----------------------------------------------------------------------

if __name__ == '__main__':
    main()
