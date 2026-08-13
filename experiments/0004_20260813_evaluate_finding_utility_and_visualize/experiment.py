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


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.data_process.labels import load_finding_labels
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.trident_pipeline import coords_subdir_name, load_patch_features, pool_slide_mean_features
    from lib.validate.batch_effect import compute_logreg_probe, stratified_subsample_indices

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
    pathology_csv = config["pathology_csv"]
    logreg_n_splits = config["logreg_n_splits"]
    n_viz_patches_per_slide = config["n_viz_patches_per_slide"]
    viz_seed = config["viz_seed"]
    tsne_perplexity = config["tsne_perplexity"]

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

    finding_labels = load_finding_labels(
        dataset_dir / pathological_image_csv, dataset_dir / pathology_csv
    )

    # ── Shared visualization subsample: same row indices reused across all four
    # methods (they share identical coordinates by construction -- see 0002 --
    # so index i always refers to the same (slide, patch-location) in every method's
    # feature array), for an apples-to-apples "does the same patch set look more
    # mixed" comparison.
    viz_idx = None
    viz_slide_ids = None

    results = {}
    tsne_embeddings = {}
    for method in METHODS:
        features_dir = job_dir / sub_coords_dir / f"features_uni_v1_{method}"
        X, slide_ids = load_patch_features(features_dir)
        logger.info(f"[{method}] loaded {X.shape[0]} patches ({X.shape[1]}-d) from {len(set(slide_ids))} slides")

        # ── has_finding utility probe (slide-mean pooled) ───────────────────────
        X_mean, unique_slide_ids = pool_slide_mean_features(X, slide_ids)
        unknown = set(unique_slide_ids) - set(finding_labels.index)
        if unknown:
            raise ValueError(f"[{method}] {len(unknown)} slide_id(s) have no has_finding label, e.g. {sorted(unknown)[:5]}")
        y_has_finding = finding_labels.loc[unique_slide_ids, "has_finding"].to_numpy()

        finding_probe = compute_logreg_probe(X_mean, y_has_finding, n_splits=logreg_n_splits, random_state=seed)
        logger.info(f"[{method}] has_finding probe: roc_auc={finding_probe['roc_auc']:.4f}, balanced_accuracy={finding_probe['balanced_accuracy']:.4f}")

        # ── t-SNE visualization subsample (computed once, from the first method) ─
        if viz_idx is None:
            viz_idx = stratified_subsample_indices(slide_ids, n_viz_patches_per_slide, viz_seed)
            viz_slide_ids = slide_ids[viz_idx]
            logger.info(f"t-SNE visualization subsample: {len(viz_idx)} patches across {len(set(viz_slide_ids))} slides")

        from sklearn.manifold import TSNE

        embedding = TSNE(
            n_components=2, perplexity=tsne_perplexity, random_state=seed, init="pca"
        ).fit_transform(X[viz_idx])
        tsne_embeddings[method] = embedding
        logger.info(f"[{method}] t-SNE done: {embedding.shape}")

        results[method] = {
            "n_patches": int(X.shape[0]),
            "n_slides": int(len(unique_slide_ids)),
            "n_finding_positive_slides": int(y_has_finding.sum()),
            "has_finding_probe": finding_probe,
        }

    # ── Comparison table ─────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(
        [
            {
                "method": method,
                "roc_auc_has_finding": results[method]["has_finding_probe"]["roc_auc"],
                "balanced_accuracy_has_finding": results[method]["has_finding_probe"]["balanced_accuracy"],
            }
            for method in METHODS
        ]
    ).set_index("method")
    comparison_df.to_csv(run_dir / "comparison_table.csv")
    logger.info("Comparison table:\n" + comparison_df.to_string())

    # ── t-SNE scatter grid, colored by slide (no legend -- 998 slides is too many
    # to read off individually; the point is whether per-slide clumps merge) ───
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slide_codes = pd.factorize(viz_slide_ids)[0]
    cmap = plt.get_cmap("tab20")
    colors = cmap(slide_codes % 20)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, method in zip(axes.flat, METHODS):
        emb = tsne_embeddings[method]
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=4, alpha=0.6, linewidths=0)
        ax.set_title(method)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("t-SNE of UNI patch features, colored by slide (each slide = one color, repeating every 20)")
    fig.tight_layout()
    fig.savefig(run_dir / "tsne_by_slide.png", dpi=150)
    plt.close(fig)

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    np.savez(
        run_dir / "tsne_embeddings.npz",
        viz_slide_ids=viz_slide_ids,
        **{f"embedding_{method}": tsne_embeddings[method] for method in METHODS},
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
