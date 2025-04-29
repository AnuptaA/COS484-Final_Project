# COS484-Final_Project

# Installation
1. Create a python venv using python version 3.10 (necessary for RADIO) i.e. just run
`python3.10 -m venv /path/to/your/env`.

2. Activate your venv and run `pip install -r requirements.txt` from the root directory.

3. Also install OpenAI CLIP from the repository by running `pip install git+https://github.com/openai/CLIP.git`

4. Run `/Applications/Python\ 3.10/Install\ Certificates.command` for `torch.hub`

5. Run `main_PoC1.py` and `main_PoC2.py` in `/paper` to run paper's experiement

6. Run `python run_models.py <model_name>` in `\B-16` to run Proof of Concept 1 from the paper
with B/16 architecture (i.e. same CLIP dimension)

7. For faster execution, in `\B-16` run `make clean` followed by `make all` to re-run all models