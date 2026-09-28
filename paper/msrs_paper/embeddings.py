"""상품 이미지 부분집합, 텍스트·이미지 임베딩, 섞기 대응표.

모든 임베딩 파일의 행 순서 = items.parquet의 idx 순서.
"""
import io
import re
import zipfile

import numpy as np
from PIL import Image

from .utils import write_json

IMAGE_RE = re.compile(r"images/\d{3}/(\d{10})\.jpg$")


def image_ids_in_zip(zip_path):
    """Kaggle 원본 zip 안의 이미지 목록 → article_id 집합."""
    with zipfile.ZipFile(zip_path) as zf:
        return {int(m.group(1)) for name in zf.namelist() if (m := IMAGE_RE.search(name))}


def write_image_subset(src_zip, article_ids, out_zip, max_side):
    """필요한 상품 이미지만 축소해서 새 zip으로 저장 (파일명: 10자리 article_id.jpg)."""
    with zipfile.ZipFile(src_zip) as src:
        names = {int(m.group(1)): n for n in src.namelist() if (m := IMAGE_RE.search(n))}
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as out:
            for aid in article_ids:
                img = Image.open(io.BytesIO(src.read(names[aid]))).convert("RGB")
                img.thumbnail((max_side, max_side))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=90)
                out.writestr(f"{aid:010d}.jpg", buf.getvalue())


def read_image(zf, article_id):
    return Image.open(io.BytesIO(zf.read(f"{article_id:010d}.jpg"))).convert("RGB")


def _image_dataset(zip_path, article_ids, transform):
    from torch.utils.data import Dataset

    class ZipImages(Dataset):
        def __init__(self):
            self.zf = None

        def __len__(self):
            return len(article_ids)

        def __getitem__(self, i):
            if self.zf is None:            # 워커마다 따로 연다
                self.zf = zipfile.ZipFile(zip_path)
            return transform(read_image(self.zf, int(article_ids[i])))

    return ZipImages()


def load_image_model(spec, device):
    import torch
    if spec["kind"] == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(spec["name"], pretrained=spec.get("pretrained"))
        model = model.to(device).eval()
        return (lambda x: model.encode_image(x)), preprocess
    if spec["kind"] == "torchvision_resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()          # 분류층 제거 → 2048차원 특징
        model = model.to(device).eval()
        return model, weights.transforms()
    raise ValueError(spec)


def extract_image(zip_path, article_ids, spec, batch_size, device, num_workers=2):
    import torch
    from torch.utils.data import DataLoader
    encode, preprocess = load_image_model(spec, device)
    loader = DataLoader(_image_dataset(zip_path, article_ids, preprocess), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        for x in loader:
            out.append(encode(x.to(device)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def extract_text(texts, model_name, batch_size, device):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=True).astype(np.float32)


def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def reduce(raw, dim, seed=42):
    """PCA로 차원을 맞추고 L2 정규화. PCA는 구매 정보를 쓰지 않으므로 전체 상품으로 학습해도 누수가 없다."""
    from sklearn.decomposition import PCA
    raw = l2_normalize(raw.astype(np.float32))
    if raw.shape[1] <= dim:
        return raw, {"raw_dim": int(raw.shape[1]), "pca": False}
    pca = PCA(n_components=dim, random_state=seed)
    emb = l2_normalize(pca.fit_transform(raw).astype(np.float32))
    return emb, {"raw_dim": int(raw.shape[1]), "pca": True,
                 "explained_variance": float(pca.explained_variance_ratio_.sum())}


def save_embedding(paths, name, emb, info):
    assert np.isfinite(emb).all(), f"{name}: NaN/inf 존재"
    np.save(paths.embeddings / f"{name}.npy", emb)
    write_json(paths.embeddings / f"{name}.json", {**info, "shape": list(emb.shape)})


def make_shuffles(product_types, seed):
    """섞기 대응표. 상품 i는 상품 perm[i]의 이미지 임베딩을 받는다.

    global: 전체 상품에서 무작위 / type: 같은 product_type_name 안에서만 무작위.
    """
    rng = np.random.default_rng(seed)
    n = len(product_types)
    perm_global = rng.permutation(n).astype(np.int32)
    perm_type = np.arange(n, dtype=np.int32)
    types = np.asarray(product_types)
    singles = 0
    for t in np.unique(types):
        idx = np.flatnonzero(types == t)
        if len(idx) == 1:
            singles += 1
        perm_type[idx] = idx[rng.permutation(len(idx))]
    info = {
        "seed": seed, "n": n,
        "global_fixed_share": float((perm_global == np.arange(n)).mean()),
        "type_fixed_share": float((perm_type == np.arange(n)).mean()),
        "type_single_item_types": int(singles),
        "type_preserved": bool((types[perm_type] == types).all()),
    }
    return perm_global, perm_type, info
