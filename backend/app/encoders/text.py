"""MiniLM 텍스트 인코더 + PCA 256 투영. 상품 text_embedding과 같은 파이프라인 (embeddings/v2):
all-MiniLM-L6-v2 → L2 → (v − mean) @ components.T → L2.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..services.search_core import project


class TextEncoder:
    def __init__(self, model_name: str, pca_path: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.pca_mean = self.pca_components = None
        if pca_path and Path(pca_path).exists():
            npz = np.load(pca_path)
            self.pca_mean = npz["mean"].astype(np.float32)
            self.pca_components = npz["components"].astype(np.float32)
        elif pca_path:
            print(f"[주의] PCA 파일 없음: {pca_path} — POST /search 텍스트 질의는 503으로 응답한다")

    @property
    def pca_ready(self) -> bool:
        return self.pca_mean is not None and self.pca_components is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        """텍스트 → (n, pca_dim) L2 정규화. PCA 미설치면 RuntimeError."""
        if not self.pca_ready:
            raise RuntimeError("PCA 변환 행렬이 로드되지 않았다 (pca_text_path 확인)")
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return project(v, self.pca_mean, self.pca_components)


@lru_cache(maxsize=1)
def load_text_encoder(model_name: str, pca_path: str | None = None) -> TextEncoder:
    return TextEncoder(model_name, pca_path)
