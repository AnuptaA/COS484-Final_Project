from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)

# get the model
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
model.eval()

# get processed image
x = Image.open('./coco-images/images/COCO_train2014_000000196545.jpg').convert('RGB')
x = pil_to_tensor(x).to(dtype=torch.float32)
x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
x = x.unsqueeze(0) # Add a batch dimension

nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

summary, spatial_features = model(x)

print("Model works!")
print(f"Summary: {summary}")
print(f"Features: {spatial_features}")