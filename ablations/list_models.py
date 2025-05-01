#!/usr/bin/env python

import open_clip

#-----------------------------------------------------------------------

def main():
    pretrained_models = open_clip.list_pretrained()
    print(f"There are {len(pretrained_models)} models available")
    for i, model in enumerate(pretrained_models):
        # print(f"Model {i}: {model}")

        # if model[1] == 'openai':
        #     print(f"CLIP OpenAI Model: {model}")

        if 'siglip' in model[0].lower():
            print(f"SigLIP or SigLIP2 Model: {model}")

#-----------------------------------------------------------------------

if __name__ == '__main__':
    main()

########################################################################
# Ablation (1) - Breadth
########################################################################
#
# For breadth we keep the architecture constant and vary the models. For
# the architecture, we choose B-16 with all having 224x224 resolution
#
# The exact models to be used for breadth comparison are:
#   - ViT-B-16, openai
#   - ViT-B-16-quickgelu, openai
#   - ViT-B-16-SigLIP, webli
#   - ViT-B-16-SigLIP2, webli
#   - radio_v2.5-b, N/A
#
########################################################################
# Ablation (2) - Depth
########################################################################
#
# For depth we keep the model constant and vary model parameters such as
# patch size (e.g. B/16 vs H/14)
#
# The exact models to be used for depth comparison are:
#   - CLIP
#       - ViT-B-16 (224x224)
#       - ViT-L-14 (224x224)
#   - CLIP with quickGELU activation
#       - ViT-B-16-quickgelu (224x224)
#       - ViT-L-14-quickgelu (224x224)
#   - SigLIP
#       - ViT-B-16-SigLIP-384 (384x384)
#       - ViT-L-16-SigLIP-384 (384x384)
#       - ViT-SO400M-14-SigLIP-384 (384x384)
#   - SigLIP2
#       - ViT-B-16-SigLIP2-384 (384x384)
#       - ViT-L-16-SigLIP2-384 (384x384)
#       - ViT-SO400M-16-SigLIP2-384 (384x384)
#   - RADIO:
#       - radio_v2.5-b
#       - radio_v2.5-l
#       - radio_v2.5-h
#       - radio_v2.5-g