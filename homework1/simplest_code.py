import torch, numpy as np
from PIL import Image

def poly(x):
    r,g,b=x[...,0],x[...,1],x[...,2]
    return torch.stack([torch.ones_like(r),r,g,b,r*r,g*g,b*b,r*g,r*b,g*b,r*g*b],-1)

src = torch.from_numpy(np.asarray(Image.open("train_input.png")).astype(np.float32)/255)
tgt = torch.from_numpy(np.asarray(Image.open("train_target.png")).astype(np.float32)/255)
X,Y = poly(src).reshape(-1,11), tgt.reshape(-1,3)
W = torch.linalg.pinv(X) @ Y
test = torch.from_numpy(np.asarray(Image.open("test_input.png")).astype(np.float32)/255)
out = (poly(test) @ W).clamp(0,1).numpy()
Image.fromarray((out*255).astype(np.uint8)).save("test_output.png")
