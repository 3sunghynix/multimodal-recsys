"""CLIP 이미지·텍스트 인코더. 쿼리 이미지와 상품 title을 같은 벡터 공간으로 매핑한다.

OpenAI CLIP ViT-B/32는 image/text가 공유 공간에 투영되므로 쿼리 이미지(image encoder)와
상품 title(text encoder) 간 코사인 유사도 검색이 동작한다. 샘플 데이터처럼 실제 상품 이미지가
없는 환경에서도 "텍스트로 설명된 상품 ↔ 업로드 이미지" 매칭 데모가 가능하다.

Marqo-FashionSigLIP 같은 패션 특화 모델로 교체하려면 config.image_encoder_model만 바꾼다
(단, 임베딩 차원이 달라지면 embedding_dim과 items.image_embedding도 함께 조정).
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


class ClipImageEncoder:
    """지연 로딩 CLIP 래퍼. app.state에 하나만 보관해서 모든 요청이 재사용한다."""

    def __init__(self, model_name: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @property
    def dim(self) -> int:
        return int(self.model.config.projection_dim)

    def encode_image_bytes(self, data: bytes) -> np.ndarray:
        """이미지 바이트 → L2 정규화된 임베딩 (dim,)."""
        from PIL import Image, ImageOps

        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")   # 회전 보정 + RGB 통일
        return self._encode_image_pil(img)

    def _encode_image_pil(self, img: "Image") -> np.ndarray:
        import torch

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            v = self.model.get_image_features(**inputs)
        return _l2(v.cpu().numpy()[0])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """상품 title 등 짧은 텍스트 묶음 → (N, dim) L2 정규화 임베딩. seed/적재 스크립트용."""
        import torch

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            v = self.model.get_text_features(**inputs)
        return _l2_batch(v.cpu().numpy())


def _l2(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return (x / n).astype(np.float32) if n > 0 else x.astype(np.float32)


def _l2_batch(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


@lru_cache(maxsize=1)
def load_encoder(model_name: str) -> ClipImageEncoder:
    """프로세스당 한 번만 로드. seed_sample 같은 스크립트에서도 공유 가능."""
    return ClipImageEncoder(model_name)
