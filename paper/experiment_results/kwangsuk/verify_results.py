"""Offline audit of the archived results; needs only NumPy and pandas.

Run from the repository root:
    python paper/experiment_results/kwangsuk/verify_results.py
This checks saved artifacts, not a replay of the original H&M predictions.
"""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=0, atol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def audit():
    provenance = read_json(ROOT / "PROVENANCE.json")
    for name, expected in provenance["artifact_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Original artifact changed: {name}")

    for hypothesis in ("H_TextMention", "H_TextMask", "H_VisualConsistency"):
        result = ROOT / hypothesis / "results"
        summary = read_json(result / "summary.json")
        samples = np.load(result / "bootstrap_estimates.npy", allow_pickle=False)
        assert samples.shape == (1000,) and np.isfinite(samples).all()
        close(np.quantile(samples, [.025, .975]),
              summary["ci95_customer_bootstrap"], f"{hypothesis} confidence interval")

    p = ROOT / "H_TextMention/results"
    s = read_json(p / "summary.json")
    table = pd.read_csv(p / "group_scores.csv")
    close(table["real_recall_at_300"] - table["shuffled_recall_at_300"],
          table["image_contribution"], "Mention contribution")
    counts = pd.read_csv(p / "purchase_counts.csv").set_index("pgroup")
    eligible = counts.index[(counts.not_mentioned >= 300) & (counts.mentioned >= 300)].tolist()
    assert eligible == s["eligible_categories"]
    weights = counts.loc[eligible].sum(axis=1)
    weights = weights / weights.sum()
    close(weights, [s["category_weights"][name] for name in eligible], "Mention weights")
    cells = table.pivot(index="category", columns="description_group", values="image_contribution")
    estimate = sum(weights[c] * (cells.loc[c, "not_mentioned"] - cells.loc[c, "mentioned"])
                   for c in eligible)
    close(estimate, s["estimate"], "Mention estimate")
    close(np.mean(list(s["seed_estimates"].values())), s["estimate"], "Mention seed average")

    p = ROOT / "H_TextMask/results"
    s = read_json(p / "summary.json")
    table = pd.read_csv(p / "condition_scores.csv").set_index("text_version")
    close(table.real_recall_at_300 - table.shuffled_recall_at_300,
          table.image_contribution, "Mask contribution")
    close(table.loc["original", "image_contribution"], s["original_image_contribution"], "Original gain")
    close(table.loc["masked", "image_contribution"], s["masked_image_contribution"], "Masked gain")
    close(table.loc["masked", "image_contribution"] - table.loc["original", "image_contribution"],
          s["difference_masked_minus_original"], "Mask estimate")
    seeds = pd.read_csv(p / "seed_scores.csv")
    close(seeds.masked_contribution - seeds.original_contribution, seeds.difference, "Mask seed differences")
    close(seeds.difference.mean(), s["difference_masked_minus_original"], "Mask seed average")
    settings = read_json(ROOT / "H_TextMask/settings/training_settings.json")
    infos = list((ROOT / "H_TextMask/training_metadata").rglob("run_info.json"))
    assert len(infos) == 12
    combinations = set()
    for path in infos:
        info = read_json(path)
        combinations.add((info["group"], info["cond"], info["seed"]))
        assert info["git_commit"] == settings["code_commit"]
        assert info["params"] == settings["cfg"]["train"]["default"]
        assert info["n_train_pairs"] == 1133553
        assert info["n_items"] == 53232 and info["input_dim"] == 512
        assert info["id_cfg"] is None and info["spec"] == settings["conditions"][info["cond"]]
    assert combinations == {(v, c, seed) for v in ("original", "masked")
                            for c in ("C2", "C4") for seed in (42, 43, 44)}

    p = ROOT / "H_VisualConsistency/results"
    s = read_json(p / "summary.json")
    table = pd.read_csv(p / "stratum_scores.csv")
    groups = pd.read_csv(p / "group_summary.csv").set_index("consistency_group")
    weights = pd.read_csv(p / "stratum_weights.csv").set_index(["product_type", "history_band"])["weight"]
    assert len(weights) == s["n_strata"]
    close(weights.sum(), 1.0, "Cohort weights sum")
    close(table.real_recall_at_300 - table.shuffled_recall_at_300,
          table.image_contribution, "Cohort cell contribution")
    assert int(table.customers.sum()) == s["n_evaluated_customers"]
    assert int(table.purchases.sum()) == s["n_evaluated_purchases"]
    counts = pd.read_csv(p / "evaluation_customer_counts.csv").set_index(["product_type", "history_band"])
    expected_counts = counts.loc[weights.index].sum(axis=1)
    close(expected_counts / expected_counts.sum(), weights, "Cohort weighting rule")
    for group in ("low", "high"):
        sub = table[table.consistency_group == group].set_index(["product_type", "history_band"])
        assert (sub.customers >= 30).all()
        for source, target in [("real_recall_at_300", "adjusted_real_recall"),
                               ("shuffled_recall_at_300", "adjusted_shuffled_recall"),
                               ("image_contribution", "adjusted_image_contribution")]:
            close((sub[source].reindex(weights.index) * weights).sum(), groups.loc[group, target],
                  f"Cohort {group} {source}")
    close(groups.loc["high", "adjusted_image_contribution"] - groups.loc["low", "adjusted_image_contribution"],
          s["difference"], "Cohort estimate")
    seeds = pd.read_csv(p / "seed_scores.csv")
    close(seeds.high_contribution - seeds.low_contribution, seeds.difference_high_minus_low,
          "Cohort seed differences")
    close(seeds.difference_high_minus_low.mean(), s["difference"], "Cohort seed average")

    notebook_dir = ROOT.parents[1] / "notebooks"
    for name in ("kwangsuk_text_mention", "kwangsuk_text_mask", "kwangsuk_visual_consistency"):
        nb = read_json(notebook_dir / f"{name}.ipynb")
        assert nb["nbformat"] == 4
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename=f"{name}:cell{i}")
                assert cell["outputs"] == [] and cell["execution_count"] is None
    print("PASS: 33 original artifact hashes; 3 saved CIs; tables/weights/seed contrasts;")
    print("      12 training metadata files; all recovered notebook Python cells.")
    print("Scope: offline consistency audit only; original H&M predictions were not replayed.")


if __name__ == "__main__":
    audit()
