"""Evaluate model scores against the public RoboJEPA rankability candidates."""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from rankability_metrics import evaluate


DATASET_REPO = "Ryukijano/robojepa-rankability"


def _load_data_files(split, data_dir):
    if data_dir is not None:
        root = Path(data_dir)
        if (root / "data").is_dir():
            root /= "data"
        candidates = root / f"candidates-{split}.parquet"
        anchors = sorted(root.glob(f"anchors-{split}-*.parquet"))
        if not candidates.is_file() or not anchors:
            raise FileNotFoundError(f"Expected candidates-{split}.parquet and anchors-{split}-*.parquet under {root}")
        return candidates, anchors

    candidates = Path(hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename=f"data/candidates-{split}.parquet"
    ))
    repo_files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
    anchor_names = sorted(
        name for name in repo_files
        if name.startswith(f"data/anchors-{split}-") and name.endswith(".parquet")
    )
    if not anchor_names:
        raise FileNotFoundError(f"No anchor parquet files found for split {split!r} in {DATASET_REPO}")
    anchors = [
        Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=name))
        for name in anchor_names
    ]
    return candidates, anchors


def _evaluate_scores(split, scores_path, data_dir):
    candidates_path, anchor_paths = _load_data_files(split, data_dir)
    anchors = pd.concat(
        [pd.read_parquet(path, columns=["anchor_id"]) for path in anchor_paths],
        ignore_index=True,
    )
    anchor_ids = set(anchors["anchor_id"].astype(str))
    candidates = pd.read_parquet(
        candidates_path,
        columns=["anchor_id", "cand_idx", "final_dist_m", "source"],
    )
    candidates["anchor_id"] = candidates["anchor_id"].astype(str)
    candidates["cand_idx"] = pd.to_numeric(candidates["cand_idx"], errors="raise").astype(int)
    candidates = candidates[candidates["anchor_id"].isin(anchor_ids)]
    if candidates.duplicated(["anchor_id", "cand_idx"]).any():
        raise ValueError("Candidate parquet contains duplicate (anchor_id, cand_idx) keys")

    scores = pd.read_parquet(scores_path)
    if "split" in scores.columns:
        scores = scores[scores["split"].astype(str) == split]
    required = {"anchor_id", "cand_idx", "score"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Score parquet is missing required columns: {', '.join(sorted(missing))}")
    scores = scores[["anchor_id", "cand_idx", "score"]].copy()
    scores["anchor_id"] = scores["anchor_id"].astype(str)
    scores["cand_idx"] = pd.to_numeric(scores["cand_idx"], errors="raise").astype(int)
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    if scores["score"].isna().any():
        raise ValueError("Scores must be finite numeric values")
    if not np.isfinite(scores["score"].to_numpy(dtype=float)).all():
        raise ValueError("Scores must be finite numeric values")
    if scores.duplicated(["anchor_id", "cand_idx"]).any():
        raise ValueError("Score parquet contains duplicate (anchor_id, cand_idx) keys")
    unknown = set(scores["anchor_id"]) - anchor_ids
    if unknown:
        raise ValueError(f"Score file contains {len(unknown)} anchor ids not present in the {split} anchors")
    candidate_keys = set(map(tuple, candidates[["anchor_id", "cand_idx"]].to_numpy()))
    score_keys = set(map(tuple, scores[["anchor_id", "cand_idx"]].to_numpy()))
    unexpected = score_keys - candidate_keys
    if unexpected:
        raise ValueError(f"Score file contains {len(unexpected)} candidate keys not present in the dataset")

    candidate_groups = {
        anchor_id: group.sort_values("cand_idx")
        for anchor_id, group in candidates.groupby("anchor_id", sort=True)
    }
    if any(len(group) != 64 for group in candidate_groups.values()):
        raise ValueError("Every anchor in the dataset must have exactly 64 candidates")
    scores_by_anchor = {anchor_id: group for anchor_id, group in scores.groupby("anchor_id", sort=True)}
    matrices = []
    dists = []
    sources = []
    incomplete = 0
    for anchor_id, candidate_group in candidate_groups.items():
        score_group = scores_by_anchor.get(anchor_id)
        if score_group is None or set(score_group["cand_idx"]) != set(candidate_group["cand_idx"]):
            incomplete += 1
            continue
        ordered = candidate_group.merge(
            score_group[["cand_idx", "score"]],
            on="cand_idx",
            how="inner",
            validate="one_to_one",
        )
        matrices.append(ordered["score"].to_numpy(dtype=float))
        dists.append(ordered["final_dist_m"].to_numpy(dtype=float))
        sources.append(ordered["source"].to_numpy())
    if len(matrices) < 2:
        raise ValueError(f"Need at least two complete anchors; found {len(matrices)}")

    result = evaluate(np.stack(matrices), np.stack(dists), np.stack(sources))
    return result, len(scores), incomplete


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["pretrain", "target"], required=True)
    parser.add_argument("--scores", type=Path, required=True, help="Parquet with anchor_id, cand_idx and score columns")
    parser.add_argument("--name", help="Optional model name for the output JSON")
    parser.add_argument("--out", type=Path, help="Optional path for a JSON results file")
    parser.add_argument("--data_dir", "--data-dir", type=Path, help="Local dataset/data directory; defaults to the HF dataset")
    args = parser.parse_args()

    metrics, n_score_rows, n_incomplete = _evaluate_scores(args.split, args.scores, args.data_dir)
    result = {
        "name": args.name or args.scores.stem,
        "split": args.split,
        "score_rows": n_score_rows,
        "incomplete_anchors_skipped": n_incomplete,
        "metrics": metrics,
    }
    rendered = json.dumps(_json_safe(result), indent=2, allow_nan=False)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
