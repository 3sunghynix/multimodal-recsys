"""Execute recovered preparation/evaluation cells on synthetic local data.

No model downloads, Colab connection, original H&M data, or training required.
Run from the repository root:
    python paper/experiment_results/kwangsuk/test_recovered_notebooks.py
"""
import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def cells(name):
    nb = json.loads((ROOT.parents[1] / "notebooks" / f"{name}.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def fixture(root):
    n_items, n_users = 5972, 180
    rng = np.random.default_rng(730)
    vectors = rng.normal(size=(n_items, 8))
    history_rows = []
    target_rows = []
    for user in range(n_users):
        kind = "T-shirt" if user < 90 else "Dress"
        noise = .02 + (user % 90) / 90
        for j in range(5):
            item = 5 * user + j
            vectors[item] = np.array([1., 0, 0, 0, 0, 0, 0, 0]) + noise * rng.normal(size=8)
            history_rows.append((user, item, datetime(2020, 8, 1 + j)))
        # Repeated purchase must not increase the five distinct products.
        history_rows.append((user, 5 * user, datetime(2020, 8, 10)))
        for j in range(6):
            item = 2000 + (0 if user < 90 else 1000) + j
            target_rows.append((user, item, False, False, "upper", "pattern"))
        # Future purchase must be excluded from consistency history.
        history_rows.append((user, 2000, datetime(2020, 9, 17)))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    types = np.array(["T-shirt"] * n_items, dtype=object)
    types[450:900] = "Dress"
    types[3000:3006] = "Dress"
    items = pd.DataFrame({
        "idx": np.arange(n_items), "article_id": np.arange(n_items) + 100000,
        "product_type_name": types, "pgroup": "upper", "appearance": "pattern",
        "desc_mentions_pattern": np.arange(n_items) % 2 == 1,
        "text": "Printed item. Cotton top."
    })
    history = pd.DataFrame(history_rows, columns=["user", "item_idx", "t_dat"])
    targets = pd.DataFrame(target_rows, columns=[
        "user", "item_idx", "is_repurchase", "is_new_item", "pgroup", "appearance"
    ])
    (root / "processed/main").mkdir(parents=True)
    (root / "embeddings").mkdir()
    items.to_parquet(root / "processed/items.parquet", index=False)
    history.to_parquet(root / "processed/main/history.parquet", index=False)
    targets.to_parquet(root / "processed/main/test_targets.parquet", index=False)
    (root / "processed/main/stats.json").write_text(json.dumps({"test_start": "2020-09-16"}))
    np.save(root / "embeddings/img_marqo.npy", vectors.astype(np.float32))
    for seed in [42, 43, 44]:
        for condition in ["C2", "C4"]:
            folder = root / f"outputs/A/main/{condition}_seed{seed}"
            folder.mkdir(parents=True)
            # Reverse customer order to exercise prediction-row alignment.
            users = np.arange(n_users)[::-1]
            candidates = np.full((n_users, 300), n_items - 1, dtype=np.int32)
            for row, user in enumerate(users):
                base = 2000 if user < 90 else 3000
                candidates[row, 0] = base
                if condition == "C2":
                    candidates[row, 1] = base + 1
            np.save(folder / "users.npy", users)
            np.save(folder / "candidates.npy", candidates)
    return items, vectors


def run(core_paper):
    sys.path.insert(0, str(core_paper))
    from msrs_paper import metrics
    # Load only the unmodified Paths class: the offline test does not need YAML.
    config_tree = ast.parse((core_paper / "msrs_paper/config.py").read_text())
    class_node = next(node for node in config_tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "Paths")
    namespace = {"Path": Path, "SPLITS": ("main", "alt")}
    exec(compile(ast.Module(body=[class_node], type_ignores=[]), "Paths", "exec"), namespace)
    Paths = namespace["Paths"]
    # In-memory synthetic table adapter: do not require a Parquet engine.
    # Real Parquet serialization and model-library integration are outside this test.
    tables = {}
    def write_table(frame, path, **kwargs):
        path = Path(path).resolve()
        tables[path] = frame.copy()
        path.write_text("Synthetic table adapter; not a real Parquet file.\n")
    def read_table(path, **kwargs):
        return tables[Path(path).resolve()].copy()
    with TemporaryDirectory(prefix="kwangsuk-notebook-check-") as temp, \
            patch.object(pd.DataFrame, "to_parquet", write_table), \
            patch.object(pd, "read_parquet", read_table):
        temp = Path(temp)
        root = temp / "data"
        items, vectors = fixture(root)
        base = {"DRIVE_ROOT": str(root), "metrics": metrics, "Paths": Paths,
                "CORE_DIR": core_paper.parent, "pd": pd, "np": np,
                "datetime": datetime, "timezone": timezone, "json": json, "Path": Path}
        with redirect_stdout(io.StringIO()):
            mention = cells("kwangsuk_text_mention")
            env = dict(base, RUN_OUTPUT_ROOT=temp / "mention")
            exec(mention[1], env)  # Original grouping source.
            exec(mention[2], env)  # Original evaluation source.
            assert np.isclose(env["estimate"], -1 / 3)
            assert len(env["bootstrap_estimates"]) == 1000

            mask = cells("kwangsuk_text_mask")
            exp_root = temp / "mask_inputs"
            env = dict(base, RUN_TRAINING=True, EXP_ROOT=exp_root, paths=Paths(root))
            exec(mask[2], env)  # Reconstructed input preparation only.
            versions = pd.read_parquet(exp_root / "text_versions.parquet")
            assert len(env["primary_items"]) == 5972
            assert versions.modified_text.eq("[MASK] item. Cotton top.").all()
            for version in ["original", "masked"]:
                folder = exp_root / "embeddings" / version
                folder.mkdir(parents=True)
                np.save(folder / "text_minilm.npy", vectors if version == "original" else vectors + .01)
                for seed in [42, 43, 44]:
                    for condition in ["C2", "C4"]:
                        src = root / f"outputs/A/main/{condition}_seed{seed}"
                        dst = exp_root / f"outputs/{version}/main/{condition}_seed{seed}"
                        dst.mkdir(parents=True)
                        users = np.load(src / "users.npy")
                        candidates = np.load(src / "candidates.npy")
                        if version == "masked" and condition == "C4":
                            candidates[:] = 5971
                        np.save(dst / "users.npy", users)
                        np.save(dst / "candidates.npy", candidates)
                        (dst / "run_info.json").write_text(json.dumps({
                            "seed": seed, "cond": condition, "split": "main", "group": version,
                            "params": {"synthetic": True}, "n_train_pairs": 1133553
                        }))
            env.update(RUN_OUTPUT_ROOT=temp / "mask_results")
            exec(mask[-1], env)
            assert np.isclose(env["estimate"], 1 / 6)
            assert env["n_purchases"] == 1080
            assert len(env["bootstrap_estimates"]) == 1000

            cohort = cells("kwangsuk_visual_consistency")
            env = dict(base, REBUILD_GROUPS=True, COHORT_ROOT=temp / "cohort_results/cohorts",
                       RUN_OUTPUT_ROOT=temp / "cohort_results")
            exec(cohort[2], env)
            assert env["customers"].n_type_history.eq(5).all()
            assert len(env["customers"]) == 180
            exec(cohort[-1], env)
            assert len(env["eligible_strata"]) == 2
            assert np.isclose(env["estimate"], 0)
            assert np.allclose(env["bootstrap_estimates"], 0)
            assert np.all(env["purchases_per_user"] == 6)
    print("PASS: three recovered analyses executed end-to-end on synthetic files.")
    print("      shuffled customer ordering; expected contrast signs; history deduplication;")
    print("      test-date exclusion; paired/stratified bootstrap; reconstructed masking inputs.")
    print("No Colab setup, model extraction or training was executed.")
    print("Synthetic tables used an in-memory adapter; Parquet serialization was not tested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-paper", type=Path, default=ROOT.parents[1])
    run(parser.parse_args().core_paper.resolve())
