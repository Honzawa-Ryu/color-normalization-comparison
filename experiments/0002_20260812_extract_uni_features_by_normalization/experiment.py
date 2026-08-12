import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml


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
    parser.add_argument(
        "--method",
        required=True,
        choices=("none", "macenko", "reinhard", "vahadane"),
        help="Stain normalization method applied before UNI feature extraction (GRID_ARGS dimension; see run_slurm.sh).",
    )
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.trident_pipeline import (
        auto_device,
        coords_subdir_name,
        extract_features_by_coords_lookup,
        get_stain_normalized_encoder,
    )

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    variant_key = args.method

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    mag = config["mag"]
    patch_size = config["patch_size"]
    overlap = config["overlap"]
    n_patches_per_slide = config["n_patches_per_slide"]
    batch_limit = config["batch_limit"]
    max_workers = config["max_workers"]

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, method=args.method)

    wsi_dir = dataset_dir / "raw_slide"
    # Same shared TRIDENT job_dir 0001 wrote coords into (see that experiment.py's
    # docstring comment for why it lives under data/ rather than outputs/).
    job_dir = dataset_dir / "trident_processed"
    rel_coords_dir = coords_subdir_name(mag, patch_size, overlap)
    sub_coords_dir = f"{rel_coords_dir}_sub{n_patches_per_slide}"

    if not (job_dir / sub_coords_dir / "patches").is_dir():
        logger.error(
            f"Subsampled coords not found at {job_dir / sub_coords_dir / 'patches'}. "
            "Run experiment 0001 (extract_tissue_coords) first."
        )
        sys.exit(1)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:        {run_dir}")
    logger.info(f"job_dir:        {job_dir}")
    logger.info(f"sub_coords_dir: {sub_coords_dir}")
    logger.info(f"method:         {args.method}")

    # ── method="none": reuse an already-computed, unnormalized features_uni_v1/ for
    # this coords_dir if one exists (e.g. copied in from a sibling project's plain
    # TRIDENT extraction over the same slides), instead of a full GPU forward pass.
    # See lib/trident_pipeline.extract_features_by_coords_lookup's docstring.
    existing_full_features_dir = job_dir / rel_coords_dir / "features_uni_v1"
    enc_name = f"uni_v1_{args.method}"
    reuse_existing = args.method == "none" and existing_full_features_dir.is_dir()
    if reuse_existing:
        logger.info(f"Reusing existing unnormalized features at {existing_full_features_dir} (coordinate lookup, no GPU inference).")
        dst_features_dir = job_dir / sub_coords_dir / f"features_{enc_name}"
        counts = extract_features_by_coords_lookup(
            existing_full_features_dir,
            job_dir / sub_coords_dir / "patches",
            dst_features_dir,
        )
        features_dir = str(dst_features_dir)
        n_slides = len(counts)
        logger.info(f"Reuse done: {n_slides} slides, {sum(counts.values())} patches -> {features_dir}")
    else:
        # ── UNI feature extraction (stain-normalization-wrapped) ────────────────
        device = auto_device()
        logger.info(f"device: {device}")

        from trident import Processor

        processor = Processor(
            job_dir=str(job_dir),
            wsi_source=str(wsi_dir),
            wsi_ext=[".svs"],
            skip_errors=True,
            max_workers=max_workers,
        )
        logger.info(f"Processor initialized for {len(processor.wsis)} slides.")

        encoder = get_stain_normalized_encoder(args.method, device)
        logger.info(f"Encoder ready: enc_name={encoder.enc_name}")

        features_dir = processor.run_patch_feature_extraction_job(
            coords_dir=sub_coords_dir,
            patch_encoder=encoder,
            device=device,
            saveas="h5",
            batch_limit=batch_limit,
        )
        n_slides = len(processor.wsis)
        logger.info(f"Feature extraction done: {features_dir}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "method": args.method,
        "job_dir": str(job_dir),
        "sub_coords_dir": sub_coords_dir,
        "features_dir": str(features_dir),
        "enc_name": enc_name,
        "n_slides": n_slides,
        "reused_existing_features": reuse_existing,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
