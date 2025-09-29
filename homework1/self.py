import torch 
from PIL import Image
import numpy as np

def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

def poly_features(rgb):
    r ,g ,b = rgb[...,0],rgb[...,1],rgb[...,2]
    feats = [
        torch.ones_like(r), r, g, b,
        r * r, g * g, b * b,
        r * g, r * b, g * b,
        r * g * b
    ]
    return torch.stack(feats, dim=-1)  # (..., 11)

if __name__ == "__main__":
    # 训练集
    src = load_image("homework1/input.png")   # 输入图
    tgt = load_image("homework1/output.png")  # 目标图
    
    