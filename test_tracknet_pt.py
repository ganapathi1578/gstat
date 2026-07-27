import cv2
import numpy as np
import torch
from dev.tracknet_model import TrackNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing on {device}")

model = TrackNet(in_dim=27, out_dim=8).to(device)
ckpt_path = "models/ckpts/TrackNet_best.pt"

print(f"Loading {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location=device)

if 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
elif 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

model.eval()

dummy = np.random.rand(1, 27, 288, 512).astype(np.float32)
inp = torch.from_numpy(dummy).to(device)

with torch.no_grad():
    out = model(inp)

print(f"Output shape: {out.shape}")
print("TrackNetV3 PyTorch model loaded and inferred successfully!")
