"""Helpers for driving TRIDENT (segmentation / coords / UNI feature extraction) with
stain normalization injected as a preprocessing step.

TRIDENT itself has no stain-normalization hook. Instead of re-implementing WSI patch
reading, we exploit the fact that `trident.patch_encoder_models.load.BasePatchEncoder`
stores its preprocessing pipeline as a plain, reassignable `eval_transforms` attribute
(a callable applied to each raw PIL patch crop inside
`WSIPatcherDataset.__getitem__`, i.e. once per patch, inside TRIDENT's own
DataLoader workers). Wrapping that attribute (see `lib/stain_normalization.py`) lets us
reuse TRIDENT's own `Processor.run_patch_feature_extraction_job` unmodified for all four
normalization variants, so every variant reads the exact same tissue coordinates and
only the pixels handed to UNI differ.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Union

import h5py
import numpy as np


def auto_device() -> str:
    """Return 'cuda:0' if a GPU is visible, else 'cpu'."""
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def coords_subdir_name(mag: float, patch_size: int, overlap: int) -> str:
    """TRIDENT's own naming convention for a (mag, patch_size, overlap) coords dir.

    Mirrors `Processor.run_patching_job`'s default `saveto` (`trident/Processor.py`'s
    internal `_fmt_mag`), so experiments that create the coords (0001) and experiments
    that later read them (0002) can derive the same path from shared config values
    instead of hardcoding/duplicating the string format or passing it out-of-band.
    """
    mag_str = str(int(mag)) if float(mag).is_integer() else f"{float(mag):g}"
    return f"{mag_str}x_{patch_size}px_{overlap}px_overlap"


def subsample_coords(
    src_patches_dir: Union[str, Path],
    dst_patches_dir: Union[str, Path],
    n_per_slide: int,
    seed: int = 42,
) -> dict:
    """Randomly subsample each slide's TRIDENT coords h5 down to `n_per_slide` patches.

    Reads every `<slide>_patches.h5` under `src_patches_dir` (TRIDENT's
    `<cdir>/patches/` layout) and writes a same-named, same-attrs h5 under
    `dst_patches_dir` whose `coords` dataset is a reproducible random subsample of at
    most `n_per_slide` rows. Slides with fewer than `n_per_slide` coords are copied
    unchanged (all patches kept).

    This exists so all four normalization variants (none/macenko/reinhard/vahadane)
    run UNI feature extraction over an identical, bounded-size patch set per slide
    (full-tissue extraction across 998 slides x 4 variants would be far more
    compute/disk than needed to estimate eta-squared).

    Slides are processed in sorted filename order against a single seeded RNG, so the
    same (src_patches_dir, n_per_slide, seed) always produces the same subsample.

    Returns
    -------
    dict
        Per-slide patch counts before/after subsampling, keyed by slide_id (the h5
        stem with the trailing "_patches" suffix stripped).
    """
    src_patches_dir = Path(src_patches_dir)
    dst_patches_dir = Path(dst_patches_dir)
    dst_patches_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    counts = {}

    src_files = sorted(glob.glob(str(src_patches_dir / "*_patches.h5")))
    for src_path in src_files:
        src_path = Path(src_path)
        slide_id = src_path.name.removesuffix("_patches.h5")

        with h5py.File(src_path, "r") as f:
            coords = f["coords"][:]
            attrs = dict(f["coords"].attrs)

        n_total = len(coords)
        if n_total > n_per_slide:
            idx = rng.choice(n_total, size=n_per_slide, replace=False)
            idx.sort()
            coords = coords[idx]

        dst_path = dst_patches_dir / src_path.name
        with h5py.File(dst_path, "w") as f:
            dset = f.create_dataset("coords", data=coords)
            for k, v in attrs.items():
                dset.attrs[k] = v

        counts[slide_id] = {"n_total": int(n_total), "n_sampled": int(len(coords))}

    return counts


def get_stain_normalized_encoder(method: str, device: str):
    """Build a TRIDENT UNI (`uni_v1`) patch encoder whose `eval_transforms` applies
    the given stain normalization method (see `lib/stain_normalization.py`) before the
    model's own resize/tensor/normalize preprocessing.

    `method="none"` leaves `eval_transforms` unmodified (TRIDENT's own baseline
    behavior). In both cases `encoder.enc_name` is set to `f"uni_v1_{method}"`: TRIDENT
    derives `run_patch_feature_extraction_job`'s default output dir (`features_<enc_name>/`)
    *and* its per-run log/config file names (`_logs_feats_<enc_name>.txt`,
    `_config_feats_<enc_name>.json`, written inside the *shared* coords_dir) from this
    attribute. Since all four normalization variants otherwise share the same coords_dir
    and the same underlying "uni_v1" model, leaving enc_name as the TRIDENT default would
    make concurrent per-method array tasks race on the same log/config files as well as
    collide on the same output directory.
    """
    from trident.patch_encoder_models.load import encoder_factory

    from lib.stain_normalization import make_patch_transform

    encoder = encoder_factory("uni_v1")
    encoder.eval_transforms = make_patch_transform(method, encoder.eval_transforms)
    encoder.enc_name = f"uni_v1_{method}"
    return encoder.to(device).eval()


def extract_features_by_coords_lookup(
    full_features_dir: Union[str, Path],
    sub_patches_dir: Union[str, Path],
    dst_features_dir: Union[str, Path],
) -> dict:
    """Build the "none" (no normalization) variant's subsampled feature files by
    selecting matching rows out of an *already-computed*, full-patch-set
    `features_uni_v1/` directory, instead of re-running UNI inference.

    Used when a plain (no-normalization) TRIDENT extraction already exists for the
    exact same slides at the exact same (mag, patch_size, overlap) -- e.g. copied in
    from a sibling project that ran the standard TRIDENT recipe over the same WSIs.
    Since "none" is by definition unnormalized TRIDENT output, it's identical to what
    `run_patch_feature_extraction_job(method="none")` would produce for the same
    coordinates, so this only needs an exact-coordinate row lookup (cheap, CPU-only) in
    place of a full GPU forward pass.

    Parameters
    ----------
    full_features_dir : the existing `<cdir>/features_uni_v1/` (all tissue patches,
        not subsampled).
    sub_patches_dir : this project's own subsampled `<cdir>_sub{N}/patches/` (written
        by `subsample_coords`), whose coordinates are a subset of the ones in
        `full_features_dir`.
    dst_features_dir : where to write the resulting `<slide>.h5` files, in the same
        (features, coords) layout `load_patch_features` expects.

    Raises
    ------
    FileNotFoundError
        If a slide present in `sub_patches_dir` has no matching file in
        `full_features_dir`.
    ValueError
        If a wanted coordinate isn't present in the existing features file (mag/
        patch_size/overlap mismatch, or the existing extraction covers a different
        slide set/region) -- silently falling back to a fresh GPU computation in this
        case would be surprising, so callers should let this propagate.
    """
    full_features_dir = Path(full_features_dir)
    sub_patches_dir = Path(sub_patches_dir)
    dst_features_dir = Path(dst_features_dir)
    dst_features_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for sub_h5 in sorted(sub_patches_dir.glob("*_patches.h5")):
        slide_id = sub_h5.name.removesuffix("_patches.h5")
        full_h5 = full_features_dir / f"{slide_id}.h5"
        if not full_h5.exists():
            raise FileNotFoundError(f"No existing features for slide {slide_id!r} at {full_h5}")

        with h5py.File(sub_h5, "r") as f:
            wanted_coords = f["coords"][:]

        with h5py.File(full_h5, "r") as f:
            full_coords = f["coords"][:]
            full_features = f["features"][:]

        coord_to_row = {tuple(c): i for i, c in enumerate(full_coords.tolist())}
        try:
            rows = [coord_to_row[tuple(c)] for c in wanted_coords.tolist()]
        except KeyError as e:
            raise ValueError(
                f"slide {slide_id!r}: coordinate {e.args[0]} from the subsampled "
                f"coords is missing from the existing features file {full_h5} "
                "(mag/patch_size/overlap mismatch, or a different extraction run?)."
            ) from e

        selected = full_features[np.array(rows, dtype=np.int64)]

        dst_h5 = dst_features_dir / f"{slide_id}.h5"
        with h5py.File(dst_h5, "w") as f:
            dset = f.create_dataset("features", data=selected)
            dset.attrs["encoder"] = "uni_v1_none"
            dset.attrs["name"] = slide_id
            f.create_dataset("coords", data=wanted_coords)

        counts[slide_id] = len(wanted_coords)

    return counts


def load_patch_features(features_dir: Union[str, Path]):
    """Load every `<slide>.h5` under a TRIDENT `features_<enc>/` directory into one
    stacked array, plus a parallel per-patch slide_id array.

    Returns
    -------
    X : np.ndarray, shape (n_patches_total, dim), float32
    slide_ids : np.ndarray, shape (n_patches_total,), str
        `slide_ids[i]` is the slide (h5 stem) patch `X[i]` came from.
    """
    features_dir = Path(features_dir)
    X_parts = []
    slide_id_parts = []

    for h5_path in sorted(features_dir.glob("*.h5")):
        slide_id = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]
        if len(feats) == 0:
            continue
        X_parts.append(feats)
        slide_id_parts.append(np.full(len(feats), slide_id))

    X = np.concatenate(X_parts, axis=0)
    slide_ids = np.concatenate(slide_id_parts, axis=0)
    return X, slide_ids


def pool_slide_mean_features(X: np.ndarray, slide_ids: np.ndarray):
    """Mean-pool patch-level features to one vector per slide.

    Used for slide-level labels (e.g. has_finding) that don't have a per-patch
    ground truth -- matches the "slide-mean UNI features" pooling convention used
    elsewhere in this project family (e.g. comparison-ad-toxpatho's finding-utility
    checks) rather than a patch-level probe.

    Returns
    -------
    X_mean : np.ndarray, shape (n_slides, dim)
    unique_slide_ids : np.ndarray, shape (n_slides,)
        Sorted unique slide ids; `X_mean[i]` is the mean of all patches whose
        `slide_ids` equals `unique_slide_ids[i]`.
    """
    unique_slide_ids = np.unique(slide_ids)
    X_mean = np.stack([X[slide_ids == sid].mean(axis=0) for sid in unique_slide_ids])
    return X_mean, unique_slide_ids
