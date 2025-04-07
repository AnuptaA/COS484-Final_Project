# COS484-Final_Project

# Installation
1. Create a python venv

2. Activate your venv and run `pip install -r requirements.txt`

3. Also install OpenAI CLIP from the repository by running `pip install git+https://github.com/openai/CLIP.git`

4. In another directory, locally clone TULIP from Berkeley's fork: `https://github.com/tulip-berkeley/open_clip.git`

5. Make sure your environment is activate. Run `pip install timm --upgrade`, 
`pip install transformers`, `pip install -e .` (IMPORTANT: run from the Berkeley directory for TULIP support)

6. Run `main_PoC1.py` and `main_PoC2.py` in `/paper` to run paper's experiement

7. Run `python <model>.py` from `\B-16` to run Proof of Concept 1 from the paper with B/16 arch (TULIP is 
not complete because of repo and data checkpoint issues)

8. fun