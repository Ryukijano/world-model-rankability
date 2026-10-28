"""Minimal RoboJEPA inference for the live Space: frozen V-JEPA 2.1 ViT-G/384 encoder + RoboJEPA predictor.

Mirrors probe/score_anchors.py (1-frame context, one world-model step, score = -L1 to the encoded goal frames).
Vendored code: facebookresearch/robo_jepa (CC BY-NC-SA 4.0) and facebookresearch/vjepa2, see vendor/*/LICENSE.
"""
import json, os, sys, types
import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "vendor", "robo_jepa"), os.path.join(HERE, "vendor")]
sys.modules.setdefault("wandb", types.ModuleType("wandb"))  # imported by robo_jepa utils, unused at inference

from rjapp.robo_jepa.world_model import WorldModel, WorldModelState  # noqa: E402
from rjapp.robo_jepa.mpc import mpc_utils  # noqa: E402
from vjepa2.app.vjepa_2_1.models import vision_transformer as vjepa_vit  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

IMAGE_SIZE = (240, 320)
EMBODIMENT_ID = 2  # franka_omron
CAMERA_IDS = [0, 2]  # agentview-left, eye-in-hand


class _Encoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(x)


def build_vjepa2_1_vitG_384():
    return vjepa_vit.vit_gigantic_xformers(
        patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False,
        wide_SiLU=True, uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)


def load_encoder(weights_dir, device):
    vit = build_vjepa2_1_vitG_384()
    sd = {k: v.float() for k, v in load_file(os.path.join(weights_dir, "vjepa2_1_vitG_384_target_encoder_bf16.safetensors")).items()}
    vit.load_state_dict(sd, strict=True)
    enc = _Encoder(vit).to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


def load_predictor(config_path, weights_path, encoder, device, dtype=torch.float32):
    """A WorldModel around one predictor checkpoint; the (large) image encoder is shared between models."""
    with open(config_path) as f:
        cfg = json.load(f)
    version = cfg.pop("encoder_version", "v21")
    wm = WorldModel(transition_model_config=cfg, device="cpu", heads=None)
    wm.encoder_version = version
    missing, unexpected = wm.transition_model.load_state_dict(load_file(weights_path), strict=False)
    wm.transition_model.to(device=device, dtype=dtype).eval()
    for p in wm.transition_model.parameters():
        p.requires_grad = False
    wm.encoder = encoder
    return wm, dict(missing=list(missing), unexpected=list(unexpected))


def load_world_model(weights_dir, device):
    enc = load_encoder(weights_dir, device)
    return load_predictor(os.path.join(weights_dir, "config.json"),
                          os.path.join(weights_dir, "robojepa_22m_150k_predictor.safetensors"), enc, device)


@torch.no_grad()
def encode_state(encoder_wm, init_imgs, goal_imgs, device="cuda"):
    """Encode the current and goal views once; reuse the features for several predictors."""
    device = torch.device(device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    return _encode_views(encoder_wm, init_imgs, device, dtype), _encode_views(encoder_wm, goal_imgs, device, dtype)


@torch.no_grad()
def score_from_features(wm, obs_f, goal_f, init_pose, cands, device="cuda"):
    device = torch.device(device)
    tm = wm.transition_model
    poses = torch.as_tensor(np.asarray(init_pose), dtype=torch.float32)[None, None]
    view_map = torch.tensor([CAMERA_IDS]) if tm.view_map is not None else None
    state = WorldModelState.from_features(features=obs_f, states=poses if tm.state_tokens > 0 else None,
                                          view_map=view_map, embodiment_ids=torch.tensor([[EMBODIMENT_ID]])).to(device)
    act = torch.as_tensor(np.asarray(cands), dtype=torch.float32)[:, None]
    K = act.shape[0]
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred, _ = wm.step(act, state.repeat(K), use_cache=False, no_grad=True, batch=100, device=device)
    pred = pred.float().reshape(K, -1).to(device)
    return mpc_utils.l1(goal_f.reshape(1, -1).to(device), pred).float().cpu().numpy()


def _encode_views(wm, imgs, device, dtype):
    feats = [wm.encode(np.ascontiguousarray(img)[None], size=IMAGE_SIZE, device=device, dtype=dtype)[:, None, None] for img in imgs]
    return torch.cat(feats, dim=2).to(torch.float32)  # [1, T=1, V, h, w, C]


@torch.no_grad()
def score_candidates(wm, init_imgs, goal_imgs, init_pose, cands, device="cuda"):
    """init_imgs/goal_imgs: [V,240,320,3] uint8; init_pose: [7]; cands: [K,7]. Returns -L1 per candidate (higher = better)."""
    obs_f, goal_f = encode_state(wm, init_imgs, goal_imgs, device)
    return score_from_features(wm, obs_f, goal_f, init_pose, cands, device)
