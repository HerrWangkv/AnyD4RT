"""Frozen OpenD4RT wrapper (README §3.1, §7.1): eval mode, 256² input, native aspect-ratio token."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

OD_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "Open-d4rt"
if str(OD_ROOT) not in sys.path:
    sys.path.insert(0, str(OD_ROOT))

from src.core.config import load_yaml_config  # noqa: E402
from src.eval.tasks import _encode_model_memory, _run_model_for_queries  # noqa: E402
from src.model import build_model  # noqa: E402


def load_d4rt(exp_dir: str | Path, device: str = "cuda") -> torch.nn.Module:
    exp_dir = Path(exp_dir)
    cfg = load_yaml_config(exp_dir / "model.yaml")
    model = build_model(cfg["model"])
    payload = torch.load(exp_dir / "opend4rt.ckpt", map_location="cpu")
    state = next((payload[k] for k in ("state_dict", "model", "module", "network", "net") if isinstance(payload.get(k), dict)), payload)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        raise RuntimeError(f"missing keys when loading OpenD4RT: {res.missing_keys[:5]} ...")
    del payload, state
    return model.to(device).eval()  # eval(): encoder/decoder dropout=0.1 (README §3.1)


def prepare_video(frames_rgb: np.ndarray, device: str, image_hw=(256, 256)) -> tuple[torch.Tensor, torch.Tensor]:
    """frames_rgb [T,H,W,3] uint8 at native resolution -> ([1,T,3,256,256] in [0,1], aspect [1,1] = W/H)."""
    h, w = image_hw
    src_h, src_w = frames_rgb.shape[1:3]
    rs = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames_rgb])
    video = torch.from_numpy(rs).to(device, torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 255.0
    aspect = torch.tensor([[src_w / src_h]], device=device, dtype=torch.float32)
    return video, aspect


@torch.no_grad()
def encode(model, video, aspect):
    """Encode a video once; pass the result as `memory` to every run_queries call on the same video."""
    return _encode_model_memory(model=model, video_b=video, aspect_b=aspect)


@torch.no_grad()
def run_queries(model, video, aspect, uv, t_src, t_tgt, t_cam, chunk: int = 8192, memory=None) -> dict[str, np.ndarray]:
    dev = video.device
    q = {
        "u": torch.as_tensor(uv[:, 0], dtype=torch.float32, device=dev),
        "v": torch.as_tensor(uv[:, 1], dtype=torch.float32, device=dev),
        "t_src": torch.as_tensor(t_src, dtype=torch.long, device=dev),
        "t_tgt": torch.as_tensor(t_tgt, dtype=torch.long, device=dev),
        "t_cam": torch.as_tensor(t_cam, dtype=torch.long, device=dev),
    }
    if memory is None:
        memory = encode(model, video, aspect)
    pred = _run_model_for_queries(model=model, video_b=video, aspect_b=aspect, query=q, chunk_size=chunk, memory_b=memory)
    return {k: v.numpy() for k, v in pred.items()}
