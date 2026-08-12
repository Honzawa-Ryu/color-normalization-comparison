"""Stain normalization wrappers (Macenko/Reinhard/Vahadane) built on tiatoolbox.

Used to inject stain normalization into TRIDENT's own patch-encoder pipeline by
wrapping `patch_encoder.eval_transforms` (see lib/trident_pipeline.py) instead of
re-implementing WSI patch reading/batching ourselves. All four variants
(none/macenko/reinhard/vahadane) therefore share the exact same tissue coordinates
(computed once by TRIDENT) and only differ in the pixels handed to UNI.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from PIL import Image

METHODS = ("none", "macenko", "reinhard", "vahadane")

logger = logging.getLogger(__name__)


def build_normalizer(method: str):
    """Fit a stain normalizer on a fixed canonical H&E target image.

    Returns None for "none" (no normalization). Uses a vendored copy of tiatoolbox's
    canned target image (`lib/_vendor_stainnorm.stain_norm_target`, see that module's
    docstring for why the normalizer implementations are vendored rather than
    depending on the `tiatoolbox` package) as the fit target, rather than a patch
    picked from our own data, so the reference is a fixed, externally-defined choice
    instead of an arbitrary slide from the comparison itself.
    """
    if method == "none":
        return None
    if method not in METHODS:
        raise ValueError(
            f"Unknown stain normalization method: {method!r}. Must be one of {METHODS}."
        )

    from lib._vendor_stainnorm import get_normalizer, stain_norm_target

    normalizer = get_normalizer(method)
    normalizer.fit(stain_norm_target())
    return normalizer


def make_patch_transform(method: str, base_transform: Callable) -> Callable:
    """Wrap a patch encoder's `eval_transforms` with stain normalization.

    `base_transform` is the encoder's original eval_transforms (e.g.
    Resize/ToTensor/Normalize for UNI). This prepends stain normalization on the raw
    PIL patch before handing off to it, so downstream resizing/tensor conversion is
    unchanged. TRIDENT applies `patch_encoder.eval_transforms` inside
    `WSIPatcherDataset.__getitem__`, which runs inside the DataLoader workers it spawns
    for feature extraction, so normalization is parallelized across CPU workers for
    free (no separate multiprocessing setup needed on our side).

    If normalization fails on a patch (e.g. near-blank tissue-edge patches can make
    Macenko/Vahadane's stain-matrix estimation fail to converge -- the same failure
    mode hit in 00-utils/wsi_preprocess/normalize_stains.py), falls back to the
    unnormalized patch and logs a warning rather than crashing the whole run.
    """
    if method == "none":
        return base_transform

    normalizer = build_normalizer(method)

    def _transform(pil_img: Image.Image):
        arr = np.array(pil_img.convert("RGB"))
        try:
            normalized = normalizer.transform(arr)
        except Exception as e:  # e.g. numpy.linalg.LinAlgError on near-blank patches
            logger.warning("stain normalization (%s) failed on a patch, using raw pixels instead: %s", method, e)
            normalized = arr
        return base_transform(Image.fromarray(normalized))

    return _transform
