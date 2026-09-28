from pathlib import Path

import yaml

PAPER_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = PAPER_DIR / "configs"
SPLITS = ("main", "alt")


def load_yaml(name):
    with open(CONF_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all():
    return load_yaml("paper"), load_yaml("experiments"), load_yaml("analysis_plan")


def train_params(cfg, overrides=None):
    params = dict(cfg["train"]["default"])
    params.update(overrides or {})
    return params


class Paths:
    """공유 드라이브 폴더 구조. 모든 노트북이 이 경로만 쓴다."""

    def __init__(self, root):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.processed = self.root / "processed"
        self.embeddings = self.root / "embeddings"
        self.outputs = self.root / "outputs"
        self.results = self.root / "results"
        self.logs = self.root / "logs"

    def split_dir(self, split):
        assert split in SPLITS, split
        return self.processed / split

    def run_dir(self, group, split, name):
        return self.outputs / group / split / name

    def make_dirs(self):
        for p in (self.raw, self.processed, self.embeddings, self.outputs, self.results, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self
