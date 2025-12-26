import os, base64, io
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Dinov2Model

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- HF cache (Runpod volume if mounted) ---
HF_BASE = "/runpod-volume/hf" if os.path.exists("/runpod-volume") else "/tmp/hf"
os.environ.setdefault("HF_HOME", HF_BASE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_BASE, "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

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

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = Dinov2Model.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

def preprocess_square(img: Image.Image, size: int = 512, crop_mode: str = "stretch") -> torch.Tensor:
    img = img.convert("RGB")

    if crop_mode == "center_crop":
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size), resample=Image.BICUBIC)
    else:
        img = img.resize((size, size), resample=Image.BICUBIC)

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr)
    x = (x.unsqueeze(0) - IMAGENET_MEAN) / IMAGENET_STD
    return x.squeeze(0)

def _decode_b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))

def _embed_batch(images_b64: List[str], size: int, crop_mode: str) -> torch.Tensor:
    tensors = []
    for b64 in images_b64:
        img = _decode_b64_to_pil(b64)
        tensors.append(preprocess_square(img, size=size, crop_mode=crop_mode))

    x = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        out = model(pixel_values=x)
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)

    return emb.float().detach().cpu()

# --------- FastAPI ---------
app = FastAPI()

class EmbedRequest(BaseModel):
    image_b64: Optional[str] = None
    images_b64: Optional[List[str]] = None
    size: int = 512
    crop_mode: str = "stretch"

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/embed")
def embed(req: EmbedRequest):
    init_model()

    if not req.image_b64 and not req.images_b64:
        raise HTTPException(status_code=400, detail="Provide 'image_b64' or 'images_b64'.")

    try:
        if req.images_b64 is not None:
            if len(req.images_b64) == 0:
                raise HTTPException(status_code=400, detail="'images_b64' must be non-empty.")
            emb = _embed_batch(req.images_b64, size=req.size, crop_mode=req.crop_mode)
            return {
                "embeddings": emb.tolist(),
                "dim": int(emb.shape[1]),
                "count": int(emb.shape[0]),
                "size": req.size,
                "crop_mode": req.crop_mode,
                "model_id": MODEL_ID,
            }

        emb = _embed_batch([req.image_b64], size=req.size, crop_mode=req.crop_mode)
        return {
            "embedding": emb[0].tolist(),
            "dim": int(emb.shape[1]),
            "size": req.size,
            "crop_mode": req.crop_mode,
            "model_id": MODEL_ID,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
