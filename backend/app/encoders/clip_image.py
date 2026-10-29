"""Marqo-FashionSigLIP 이미지 인코더 + PCA 256 투영.

상품 임베딩(items.image_embedding)은 드라이브 msrs_project/embeddings/v2 에서
Marqo-FashionSigLIP → L2 → PCA 256 → L2 로 생성됐다. 쿼리 이미지도 같은 파이프라인을 통과해야
cosine 비교가 의미를 갖는다.

- open_clip 로더로 hf-hub:Marqo/marqo-fashionSigLIP 모델·프리프로세서 로드
- 쿼리: PIL(RGB+EXIF 보정) → preprocess → encode_image → L2 → (q − pca_mean) @ pca_components.T → L2
- PCA 파일(pca_image.npz)이 없으면 encoder는 뜨지만 encode_image_bytes가 RuntimeError를 던진다.
  (/search/image 라우터가 이를 받아 503 응답으로 변환해 "PCA 미설치" 상태를 명시한다.)
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


class ClipImageEncoder:
    """지연 로딩 Marqo-FashionSigLIP 래퍼. app.state에 하나만 보관해서 모든 요청이 재사용한다."""

    def __init__(self, model_name: str, pca_path: str | None = None):
        import open_clip
        import torch

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        self.model = self.model.to(self.device).eval()

        self.pca_mean: np.ndarray | None = None
        self.pca_components: np.ndarray | None = None
        if pca_path:
            p = Path(pca_path)
            if p.exists():
                npz = np.load(p)
                self.pca_mean = npz["mean"].astype(np.float32)
                self.pca_components = npz["components"].astype(np.float32)
            else:
                print(f"[주의] PCA 파일 없음: {pca_path} — /search/image는 503으로 응답한다")

    @property
    def dim(self) -> int:
        if self.pca_components is not None:
            return int(self.pca_components.shape[0])
        # PCA 없으면 model의 raw 차원
        return int(self.model.visual.output_dim)

    @property
    def pca_ready(self) -> bool:
        return self.pca_mean is not None and self.pca_components is not None

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """텍스트 질의 → 상품 이미지 임베딩 공간 (n, pca_dim). 텍스트→이미지 검색용.

        이미지 평균을 빼지 않는다 (center=False). 텍스트 질의는 이미지 평균과 멀어서 빼면 모든 질의가
        한 방향으로 쏠린다 — 실측 정확도@20: 빼면 0.32, 안 빼면 0.92 (docs/project/검색.md).
        """
        import torch

        from ..services.search_core import project

        if not self.pca_ready:
            raise RuntimeError("PCA 변환 행렬이 로드되지 않았다 (pca_image_path 확인)")
        if not hasattr(self, "_tokenizer"):
            import open_clip
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.no_grad():
            v = self.model.encode_text(self._tokenizer(texts).to(self.device)).float().cpu().numpy()
        return project(v, self.pca_mean, self.pca_components, center=False)

    def encode_image_bytes(self, data: bytes) -> np.ndarray:
        """이미지 바이트 → L2 정규화된 임베딩 (pca_dim,). PCA 미설치면 RuntimeError."""
        from PIL import Image, ImageOps

        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        v = self._encode_image_pil(img)
        return self._apply_pca(v)

    def _encode_image_pil(self, img: "Image") -> np.ndarray:
        import torch

        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.model.encode_image(tensor).cpu().numpy()[0]
        return _l2(v)

    def _apply_pca(self, v: np.ndarray) -> np.ndarray:
        if not self.pca_ready:
            raise RuntimeError(
                f"PCA 변환 행렬이 로드되지 않았다 (pca_image_path 확인). "
                f"운영은 Marqo → L2 → PCA {self.dim} → L2 파이프라인이 필요하다"
            )
        q = (v - self.pca_mean) @ self.pca_components.T   # (dim,) @ (dim, pca_dim).T
        return _l2(q.astype(np.float32))


def _l2(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return (x / n).astype(np.float32) if n > 0 else x.astype(np.float32)


@lru_cache(maxsize=1)
def load_encoder(model_name: str, pca_path: str | None = None) -> ClipImageEncoder:
    """프로세스당 한 번만 로드. 같은 (model_name, pca_path) 조합은 재사용된다."""
    return ClipImageEncoder(model_name, pca_path)
