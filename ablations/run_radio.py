import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

# try this on someone's computer iwth cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# almost the same as docs but just use clip adaptor
model = torch.hub.load(
    'NVlabs/RADIO', 
    'radio_model', 
    version="radio_v2.5-b",
    adaptor_names='clip',
    progress=True
)

model.eval().to(device)

# get image as normal (this part is in RADIO repo README)
x = Image.open('./coco-images/images/COCO_train2014_000000196545.jpg').convert('RGB')
x = pil_to_tensor(x).to(torch.float32).div_(255.0).unsqueeze(0).to(device)

nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

# this is scattered throughout repo, on README as well as other places
summary, spatial_feats = model(x)['clip']
adaptor = model.adaptors['clip']

# tokenizer documentation in RADIO/examples/common/model_loader.py
tokenizer = adaptor.tokenizer
caption = "a boogie wit da hoodie"
tokens = tokenizer([caption]).to(device)

with torch.no_grad():
    text_feats = adaptor.encode_text(tokens)

# cosine similarity already normalizes, change in run_models as well
sim = F.cosine_similarity(summary, text_feats, dim=-1)
print(f"Similarity: {sim.item()}")
