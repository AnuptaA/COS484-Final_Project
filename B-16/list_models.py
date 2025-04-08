#!/usr/bin/env python

import open_clip

#-----------------------------------------------------------------------

def main():
    pretrained_models = open_clip.list_pretrained()
    print(f"There are {len(pretrained_models)} models available")
    for i, model in enumerate(pretrained_models):
        print(f"Model {i}: {model}")

#-----------------------------------------------------------------------

if __name__ == '__main__':
    main()