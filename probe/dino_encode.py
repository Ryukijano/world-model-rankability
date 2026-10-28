"""Shared frozen DINOv3 ViT-L/16 (LVD-1689M, 303M) patch-feature encoder for the encoder-swap experiment.
Loads from a staged local dir (config.json + model.safetensors, facebook/dinov3-vitl16-pretrain-lvd1689m layout)
via transformers.DINOv3ViTModel; no network access required inside jobs."""
import numpy as np, torch, torch.nn.functional as F
from transformers import DINOv3ViTModel  # noqa: E402

PATCH = 16
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def load_dino(path, device="cuda"):
    m = DINOv3ViTModel.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def encode_dino(model, imgs, image_size, device="cuda"):
    """imgs [V,H,W,3] uint8 RGB -> [V, H/PATCH, W/PATCH, C] fp16 patch features (cls + register tokens dropped)."""
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=tuple(image_size), mode="bilinear", align_corners=False)
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    out = model(x.to(device, torch.bfloat16)).last_hidden_state.float()
    hp, wp = image_size[0] // PATCH, image_size[1] // PATCH
    z = out[:, -hp * wp:]
    assert z.shape[1] == hp * wp and z.shape[2] == model.config.hidden_size, z.shape
    return z.unflatten(1, (hp, wp)).half().cpu().numpy()
