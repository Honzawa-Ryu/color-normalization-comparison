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
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.trident_pipeline import auto_device, coords_subdir_name, subsample_coords

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()
    variant_key = "default"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    mag = config["mag"]
    patch_size = config["patch_size"]
    overlap = config["overlap"]
    segmenter_name = config["segmenter"]
    seg_conf_thresh = config["seg_conf_thresh"]
    min_tissue_proportion = config["min_tissue_proportion"]
    max_workers = config["max_workers"]
    n_patches_per_slide = config["n_patches_per_slide"]
    subsample_seed = config["subsample_seed"]

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    wsi_dir = dataset_dir / "raw_slide"
    # Shared TRIDENT job_dir, written directly under project_root/data/ (NOT under
    # outputs/) -- unlike the get_run_dir()-managed outputs/ tree, this must persist
    # across separate job submissions (0001 now, 0002's 4 array tasks later) and is
    # resumable via TRIDENT's own wsi_states/.lock bookkeeping, so it's intentionally
    # kept outside the completed-guard/scratch-staging machinery that governs
    # per-experiment outputs/.
    job_dir = dataset_dir / "trident_processed"
    device = auto_device()

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:  {run_dir}")
    logger.info(f"wsi_dir:  {wsi_dir}")
    logger.info(f"job_dir:  {job_dir}")
    logger.info(f"device:   {device}, segmenter: {segmenter_name}")

    # ── Segmentation + coords (TRIDENT) ─────────────────────────────────────────
    from trident import Processor
    from trident.segmentation_models import segmentation_model_factory

    processor = Processor(
        job_dir=str(job_dir),
        wsi_source=str(wsi_dir),
        wsi_ext=[".svs"],
        skip_errors=True,
        max_workers=max_workers,
    )
    logger.info(f"Processor initialized for {len(processor.wsis)} slides.")

    seg_model = segmentation_model_factory(segmenter_name, confidence_thresh=seg_conf_thresh)
    processor.run_segmentation_job(
        segmentation_model=seg_model,
        seg_mag=seg_model.target_mag,  # segmentation runs at the segmenter's own mag, not the patch mag
        device=device,
    )
    logger.info("Segmentation done.")

    coords_dir_abs = processor.run_patching_job(
        target_magnification=mag,
        patch_size=patch_size,
        overlap=overlap,
        min_tissue_proportion=min_tissue_proportion,
    )
    logger.info(f"Coords done: {coords_dir_abs}")

    rel_coords_dir = coords_subdir_name(mag, patch_size, overlap)
    assert Path(coords_dir_abs) == job_dir / rel_coords_dir, (
        f"coords_subdir_name() naming drifted from TRIDENT's own run_patching_job() "
        f"output ({coords_dir_abs} != {job_dir / rel_coords_dir}); 0002 derives this "
        f"path independently, so this must always match."
    )
    src_patches_dir = job_dir / rel_coords_dir / "patches"

    # ── Subsample coords (shared across all 4 normalization variants in 0002) ──
    rel_sub_coords_dir = f"{rel_coords_dir}_sub{n_patches_per_slide}"
    dst_patches_dir = job_dir / rel_sub_coords_dir / "patches"

    logger.info(f"Subsampling to {n_patches_per_slide} patches/slide (seed={subsample_seed})...")
    counts = subsample_coords(
        src_patches_dir, dst_patches_dir, n_per_slide=n_patches_per_slide, seed=subsample_seed
    )
    n_slides = len(counts)
    n_total = sum(c["n_total"] for c in counts.values())
    n_sampled = sum(c["n_sampled"] for c in counts.values())
    logger.info(f"Subsampled {n_slides} slides: {n_total} -> {n_sampled} patches total.")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "job_dir": str(job_dir),
        "coords_dir": rel_coords_dir,
        "sub_coords_dir": rel_sub_coords_dir,
        "n_slides": n_slides,
        "n_patches_total": n_total,
        "n_patches_sampled": n_sampled,
        "mag": mag,
        "patch_size": patch_size,
        "overlap": overlap,
        "segmenter": segmenter_name,
        "n_patches_per_slide": n_patches_per_slide,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
