import os
import base64
import io
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Dinov2Model

import runpod

# --- HF cache (Runpod volume if mounted) ---
# If /runpod-volume exists, use it so model downloads persist across worker restarts
HF_BASE = "/runpod-volume/hf" if os.path.exists("/runpod-volume") else "/tmp/hf"
os.environ.setdefault("HF_HOME", HF_BASE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_BASE, "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # optional speedup

# Hugging Face model id (default: dinov2-large)
MODEL_ID = os.getenv("MODEL_ID", "facebook/dinov2-large")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def init_model() -> None:
    global model
    if model is not None:
        return

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Use fp16 on GPU for speed/memory (keeps output float32 later)
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = Dinov2Model.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()


def preprocess_square(
    img: Image.Image,
    size: int = 512,
    crop_mode: str = "stretch",  # "stretch" (your current behavior) or "center_crop"
) -> torch.Tensor:
    """
    Returns tensor [3,H,W] normalized with ImageNet mean/std.

    crop_mode:
      - "stretch": resize directly to (size,size) (your original behavior)
      - "center_crop": preserve aspect ratio, center-crop to square, then resize to (size,size)
    """
    img = img.convert("RGB")

    if crop_mode == "center_crop":
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), resample=Image.BICUBIC)
    else:
        # Default: stretch to square (original)
        img = img.resize((size, size), resample=Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0   # HWC
    arr = np.transpose(arr, (2, 0, 1))                 # CHW
    x = torch.from_numpy(arr)                          # [3,H,W]
    x = (x.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD
    return x.squeeze(0)                                # [3,H,W]


def _decode_b64_to_pil(b64: str) -> Image.Image:
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes))


def _embed_batch(images_b64: List[str], size: int, crop_mode: str) -> torch.Tensor:
    tensors = []
    for b64 in images_b64:
        img = _decode_b64_to_pil(b64)
        t = preprocess_square(img, size=size, crop_mode=crop_mode)
        tensors.append(t)

    x = torch.stack(tensors, dim=0).to(device)  # [B,3,H,W]

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        out = model(pixel_values=x)
        emb = out.last_hidden_state[:, 0, :]     # [B,1024]
        emb = F.normalize(emb, p=2, dim=1)

    return emb.float().detach().cpu()


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input supports either:
      {"image_b64": "...", "size": 512, "crop_mode": "stretch|center_crop"}
    or:
      {"images_b64": ["...", "..."], "size": 512, "crop_mode": "stretch|center_crop"}
    """
    init_model()

    inp = event.get("input", event)
    size = int(inp.get("size", 512))
    crop_mode = str(inp.get("crop_mode", "stretch"))

    try:
        if "images_b64" in inp:
            images_b64 = inp["images_b64"]
            if not isinstance(images_b64, list) or len(images_b64) == 0:
                return {"error": "'images_b64' must be a non-empty list."}

            emb = _embed_batch(images_b64, size=size, crop_mode=crop_mode)
            return {
                "embeddings": emb.tolist(),
                "dim": int(emb.shape[1]),
                "count": int(emb.shape[0]),
                "size": size,
                "crop_mode": crop_mode,
                "model_id": MODEL_ID,
            }

        if "image_b64" in inp:
            emb = _embed_batch([inp["image_b64"]], size=size, crop_mode=crop_mode)
            return {
                "embedding": emb[0].tolist(),
                "dim": int(emb.shape[1]),
                "size": size,
                "crop_mode": crop_mode,
                "model_id": MODEL_ID,
            }

        return {"error": "Provide either 'image_b64' or 'images_b64'."}

    except Exception as e:
        return {"error": f"Embedding failed: {e}"}


runpod.serverless.start({"handler": handler})
