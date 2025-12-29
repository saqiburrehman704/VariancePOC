import os, base64, io
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Dinov2Model

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---- HF cache ----
HF_BASE = "/runpod-volume/hf" if os.path.exists("/runpod-volume") else "/tmp/hf"
os.environ.setdefault("HF_HOME", HF_BASE)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_BASE, "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

MODEL_ID = os.getenv("MODEL_ID", "facebook/dinov2-large")

# Match torchvision Normalize(mean, std) used in Colab
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

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

    # Closest to Colab: fp32 weights + no autocast
    model = Dinov2Model.from_pretrained(MODEL_ID).to(device).eval()


def preprocess_square(img: Image.Image, size: int = 518) -> torch.Tensor:
    """
    Match Colab:
      transforms.Resize((518,518))  # stretch/squish
      transforms.ToTensor()         # [0,1], CHW
      transforms.Normalize(mean, std)
    """
    img = img.convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)  # match torchvision default behavior

    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC in [0,1]
    x = torch.from_numpy(arr).permute(2, 0, 1)        # CHW
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


def _decode_b64_to_pil(b64: str) -> Image.Image:
    # handles plain base64 (if you send data URLs, strip the prefix before calling)
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _embed_batch(images_b64: List[str], size: int) -> torch.Tensor:
    tensors = []
    for b64 in images_b64:
        img = _decode_b64_to_pil(b64)
        tensors.append(preprocess_square(img, size=size))

    x = torch.stack(tensors, dim=0).to(device, dtype=torch.float32)

    with torch.no_grad():
        out = model(pixel_values=x)
        emb = out.last_hidden_state[:, 0, :]   # CLS token
        emb = F.normalize(emb, p=2, dim=1)     # L2 normalize like Colab

    return emb.detach().cpu()


# -------- FastAPI app --------
app = FastAPI()


class EmbedRequest(BaseModel):
    image_b64: Optional[str] = None
    images_b64: Optional[List[str]] = None
    size: int = 518
    # keep this only to avoid breaking clients; Colab behavior is always "stretch"
    crop_mode: str = "stretch"


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.image_b64 and not req.images_b64:
        raise HTTPException(status_code=400, detail="Provide 'image_b64' or 'images_b64'.")

    # Colab behavior = always stretch; reject other modes to prevent mismatch
    if req.crop_mode != "stretch":
        raise HTTPException(status_code=400, detail="Only crop_mode='stretch' is supported to match Colab preprocessing.")

    init_model()

    try:
        if req.images_b64 is not None:
            if len(req.images_b64) == 0:
                raise HTTPException(status_code=400, detail="'images_b64' must be non-empty.")
            emb = _embed_batch(req.images_b64, size=req.size)
            return {
                "embeddings": emb.tolist(),
                "dim": int(emb.shape[1]),
                "count": int(emb.shape[0]),
                "size": req.size,
                "crop_mode": "stretch",
                "model_id": MODEL_ID,
            }

        emb = _embed_batch([req.image_b64], size=req.size)
        return {
            "embedding": emb[0].tolist(),
            "dim": int(emb.shape[1]),
            "size": req.size,
            "crop_mode": "stretch",
            "model_id": MODEL_ID,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
