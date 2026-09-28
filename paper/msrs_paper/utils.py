import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .config import PAPER_DIR


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PAPER_DIR, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PAPER_DIR, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def environment_info():
    info = {"python": platform.python_version(), "git_commit": git_commit()}
    for pkg in ("torch", "numpy", "pandas", "scikit-learn", "open_clip_torch", "sentence-transformers", "timm"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        info["gpu"] = "cpu"
    return info


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def publish_dir(local_dir, final_dir):
    """로컬에서 다 만든 결과 폴더를 드라이브로 옮긴다. 완료 표시(run_info.json)는 마지막에 복사된다."""
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(local_dir).iterdir(), key=lambda p: p.name == "run_info.json")
    for f in files:
        shutil.copy2(f, final_dir / f.name)


def local_tmp_dir():
    return Path(tempfile.mkdtemp(prefix="msrs_"))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(paths):
    """공유 준비물(분할, 임베딩, 섞기 대응표)의 체크섬 목록."""
    files = sorted(list(paths.processed.rglob("*.parquet")) + list(paths.processed.rglob("*.json"))
                   + list(paths.embeddings.glob("*.npy")) + list(paths.embeddings.glob("*.csv")))
    manifest = {"created_at": now(), "git_commit": git_commit(),
                "files": {str(f.relative_to(paths.root)).replace("\\", "/"): sha256(f) for f in files}}
    write_json(paths.root / "manifest.json", manifest)
    return manifest


def verify_manifest(paths):
    """드라이브의 공유 준비물이 manifest.json과 같은지 확인. 다르면 실험을 시작하지 않는다."""
    manifest = read_json(paths.root / "manifest.json")
    bad = [rel for rel, digest in manifest["files"].items()
           if not (paths.root / rel).exists() or sha256(paths.root / rel) != digest]
    if bad:
        raise RuntimeError(f"manifest와 다른 파일 {len(bad)}개: {bad[:5]} — 준비물이 바뀌었거나 복사가 덜 됐습니다.")
    return manifest
