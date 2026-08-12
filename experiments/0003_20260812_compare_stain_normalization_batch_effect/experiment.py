import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

METHODS = ("none", "macenko", "reinhard", "vahadane")


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Path to the config.yml file (unused; kept for run_slurm.sh's default RUN_COMMAND).")
    return parser.parse_args()


def _stratified_subsample(labels: np.ndarray, max_per_group: int, seed: int) -> np.ndarray:
    """Indices capping each unique value in `labels` to at most `max_per_group` rows.

    Plain global random subsampling would let small groups fall under the KNN CV fold
    count by chance (StratifiedKFold then raises); capping per-group instead guarantees
    every group keeps at least min(group_size, max_per_group) samples.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        keep.append(idx)
    idx = np.concatenate(keep)
    idx.sort()
    return idx


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.data_process.labels import load_batch_labels
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.trident_pipeline import coords_subdir_name, load_patch_features
    from lib.validate.batch_effect import compute_eta_squared, compute_knn_accuracy

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()
    variant_key = "default"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed = config.get("seed", 42)
    mag = config["mag"]
    patch_size = config["patch_size"]
    overlap = config["overlap"]
    n_patches_per_slide = config["n_patches_per_slide"]
    pathological_image_csv = config["pathological_image_csv"]
    compute_knn = config["compute_knn"]
    knn_n_neighbors = config["knn_n_neighbors"]
    knn_n_splits = config["knn_n_splits"]
    knn_max_per_group = config["knn_max_per_group"]

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    job_dir = dataset_dir / "trident_processed"
    sub_coords_dir = f"{coords_subdir_name(mag, patch_size, overlap)}_sub{n_patches_per_slide}"

    missing = [
        m for m in METHODS
        if not (job_dir / sub_coords_dir / f"features_uni_v1_{m}").is_dir()
    ]
    if missing:
        logger.error(
            f"Missing UNI features for method(s) {missing} under {job_dir / sub_coords_dir}. "
            "Run experiment 0002 (extract_uni_features_by_normalization) for all four "
            "methods first."
        )
        sys.exit(1)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:  {run_dir}")
    logger.info(f"job_dir:  {job_dir}")

    batch_labels = load_batch_labels(dataset_dir / pathological_image_csv)

    results = {}
    for method in METHODS:
        features_dir = job_dir / sub_coords_dir / f"features_uni_v1_{method}"
        X, slide_ids = load_patch_features(features_dir)
        logger.info(f"[{method}] loaded {X.shape[0]} patches ({X.shape[1]}-d) from {len(set(slide_ids))} slides")

        unknown = set(slide_ids) - set(batch_labels.index)
        if unknown:
            raise ValueError(f"[{method}] {len(unknown)} slide_id(s) have no EXP_ID label, e.g. {sorted(unknown)[:5]}")
        exp_ids = batch_labels.loc[slide_ids, "exp_id"].to_numpy()

        eta_slide = compute_eta_squared(X, slide_ids)
        eta_exp_id = compute_eta_squared(X, exp_ids)
        logger.info(f"[{method}] eta_sq_mean: slide={eta_slide['eta_sq_mean']:.4f}, exp_id={eta_exp_id['eta_sq_mean']:.4f}")

        method_result = {
            "n_patches": int(X.shape[0]),
            "n_slides": int(len(set(slide_ids))),
            "n_exp_ids": int(len(set(exp_ids))),
            "eta_squared_slide": eta_slide,
            "eta_squared_exp_id": eta_exp_id,
        }

        if compute_knn:
            slide_idx = _stratified_subsample(slide_ids, knn_max_per_group, seed)
            knn_slide = compute_knn_accuracy(
                X[slide_idx], slide_ids[slide_idx],
                n_neighbors=knn_n_neighbors, n_splits=knn_n_splits, random_state=seed,
            )
            exp_id_idx = _stratified_subsample(exp_ids, knn_max_per_group, seed)
            knn_exp_id = compute_knn_accuracy(
                X[exp_id_idx], exp_ids[exp_id_idx],
                n_neighbors=knn_n_neighbors, n_splits=knn_n_splits, random_state=seed,
            )
            logger.info(f"[{method}] knn_accuracy: slide={knn_slide:.4f} (n={len(slide_idx)}), exp_id={knn_exp_id:.4f} (n={len(exp_id_idx)})")
            method_result["knn_accuracy_slide"] = knn_slide
            method_result["knn_accuracy_exp_id"] = knn_exp_id

        results[method] = method_result

    # ── Comparison table ─────────────────────────────────────────────────────
    table_rows = []
    for method in METHODS:
        r = results[method]
        row = {
            "method": method,
            "eta_sq_mean_slide": r["eta_squared_slide"]["eta_sq_mean"],
            "eta_sq_mean_exp_id": r["eta_squared_exp_id"]["eta_sq_mean"],
        }
        if compute_knn:
            row["knn_accuracy_slide"] = r["knn_accuracy_slide"]
            row["knn_accuracy_exp_id"] = r["knn_accuracy_exp_id"]
        table_rows.append(row)
    comparison_df = pd.DataFrame(table_rows).set_index("method")
    comparison_df.to_csv(run_dir / "comparison_table.csv")
    logger.info("Comparison table:\n" + comparison_df.to_string())

    # ── Bar chart: eta_sq_mean by method, one bar group per grouping ───────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(METHODS))
    width = 0.35
    ax.bar(x - width / 2, comparison_df["eta_sq_mean_slide"], width, label="slide_id (スライド間差)")
    ax.bar(x + width / 2, comparison_df["eta_sq_mean_exp_id"], width, label="exp_id (施設間差)")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("eta_sq_mean (lower = less batch effect)")
    ax.set_title("Batch effect (eta-squared) by stain normalization method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "eta_squared_comparison.png", dpi=150)
    plt.close(fig)

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
