"""Vendored stain normalization core (Macenko/Reinhard/Vahadane), extracted from
tiatoolbox (https://github.com/TissueImageAnalytics/tiatoolbox,
`tiatoolbox/tools/stainnorm.py` + `tiatoolbox/tools/stainextract.py`, plus the
`rgb2od`/`od2rgb`/`get_luminosity_tissue_mask`/`contrast_enhancer` helpers it depends
on) and lightly adapted (only the H&E-relevant classes are kept; docstrings trimmed).

Vendored instead of depending on the `tiatoolbox` package directly because tiatoolbox
pins `timm>=1.0.3,<1.0.28`, which is incompatible with the `timm==0.9.16` TRIDENT
requires (see lib/trident_pipeline.py) -- this project only needs the small,
dependency-light stain-normalization core (numpy/opencv/scikit-image/scikit-learn),
not the rest of tiatoolbox (torch, dask, jupyterlab, sphinx, ...).

tiatoolbox is BSD-3-Clause, Copyright (c) 2026, Tissue Image Analytics (TIA) Centre.
Original code in turn credits StainTools (https://github.com/Peter554/StainTools) by
Peter Byfield for the underlying algorithms.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage import exposure
from sklearn.decomposition import DictionaryLearning

TARGET_IMAGE_PATH = Path(__file__).parent / "assets" / "stain_norm_target.png"


def stain_norm_target() -> np.ndarray:
    """Load the canonical H&E reference image used to fit Macenko/Reinhard/Vahadane.

    Vendored copy of tiatoolbox's bundled `tiatoolbox/data/target_image.png`, used as
    a fixed, externally-defined normalization target rather than an arbitrary patch
    from our own data.
    """
    img = cv2.imread(str(TARGET_IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"Could not read stain norm target image: {TARGET_IMAGE_PATH}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── utils/transforms.py ──────────────────────────────────────────────────────


def rgb2od(img: np.ndarray) -> np.ndarray:
    """Convert from RGB to optical density (OD) space."""
    mask = img == 0
    img = img.copy()
    img[mask] = 1
    return np.maximum(-1 * np.log(img / 255), 1e-6)


def od2rgb(od: np.ndarray) -> np.ndarray:
    """Convert from optical density (OD) to RGB."""
    od = np.maximum(od, 1e-6)
    return (255 * np.exp(-1 * od)).astype(np.uint8)


# ── utils/misc.py ────────────────────────────────────────────────────────────


def contrast_enhancer(img: np.ndarray, low_p: int = 2, high_p: int = 98) -> np.ndarray:
    """Enhance contrast of an image via percentile-based intensity rescaling."""
    if img.dtype != np.uint8:
        raise AssertionError("Image should be uint8.")
    img_out = img.copy()
    p_low, p_high = np.percentile(img_out, (low_p, high_p))
    if p_low >= p_high:
        p_low, p_high = np.min(img_out), np.max(img_out)
    if p_high > p_low:
        img_out = exposure.rescale_intensity(
            img_out, in_range=(p_low, p_high), out_range=(0.0, 255.0)
        )
    return img_out.astype(np.uint8)


def get_luminosity_tissue_mask(img: np.ndarray, threshold: float) -> np.ndarray:
    """Binary tissue mask based on luminosity (LAB L channel) of an RGB image."""
    img = img.astype("uint8")
    img = contrast_enhancer(img, low_p=2, high_p=98)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_lab = img_lab[:, :, 0] / 255.0
    tissue_mask = l_lab < threshold
    if tissue_mask.sum() == 0:
        raise ValueError("Empty tissue mask computed.")
    return tissue_mask


# ── tools/stainextract.py ────────────────────────────────────────────────────


def _vectors_in_correct_direction(e_vectors: np.ndarray) -> np.ndarray:
    if e_vectors[0, 0] < 0:
        e_vectors[:, 0] *= -1
    if e_vectors[0, 1] < 0:
        e_vectors[:, 1] *= -1
    return e_vectors


def _h_and_e_in_right_order(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    if v1[0] > v2[0]:
        return np.array([v1, v2])
    return np.array([v2, v1])


def _dl_output_for_h_and_e(dictionary: np.ndarray) -> np.ndarray:
    if dictionary[0, 0] < dictionary[1, 0]:
        return dictionary[[1, 0], :]
    return dictionary


class MacenkoExtractor:
    """Macenko, Marc, et al. "A method for normalizing histology slides for
    quantitative analysis." ISBI 2009.
    """

    def __init__(self, luminosity_threshold: float = 0.8, angular_percentile: float = 99) -> None:
        self._luminosity_threshold = luminosity_threshold
        self._angular_percentile = angular_percentile

    def get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        img = img.astype("uint8")
        tissue_mask = get_luminosity_tissue_mask(img, threshold=self._luminosity_threshold).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        _, eigen_vectors = np.linalg.eigh(np.cov(img_od, rowvar=False))
        eigen_vectors = eigen_vectors[:, [2, 1]]
        eigen_vectors = _vectors_in_correct_direction(e_vectors=eigen_vectors)

        proj = np.dot(img_od, eigen_vectors)
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        min_phi = np.percentile(phi, 100 - self._angular_percentile)
        max_phi = np.percentile(phi, self._angular_percentile)

        v1 = np.dot(eigen_vectors, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v2 = np.dot(eigen_vectors, np.array([np.cos(max_phi), np.sin(max_phi)]))
        he = _h_and_e_in_right_order(v1, v2)
        return he / np.linalg.norm(he, axis=1)[:, None]


class VahadaneExtractor:
    """Vahadane, Abhishek, et al. "Structure-preserving color normalization and
    sparse stain separation for histological images." IEEE TMI 2016.
    """

    def __init__(self, luminosity_threshold: float = 0.8, regularizer: float = 0.1) -> None:
        self._luminosity_threshold = luminosity_threshold
        self._regularizer = regularizer

    def get_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        img = img.astype("uint8")
        tissue_mask = get_luminosity_tissue_mask(img, threshold=self._luminosity_threshold).reshape((-1,))
        img_od = rgb2od(img).reshape((-1, 3))
        img_od = img_od[tissue_mask]

        dl = DictionaryLearning(
            n_components=2,
            alpha=self._regularizer,
            transform_alpha=self._regularizer,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_dict=True,
            verbose=False,
            max_iter=3,
            transform_max_iter=1000,
        )
        dictionary = dl.fit_transform(X=img_od.T).T
        dictionary = _dl_output_for_h_and_e(dictionary)
        return dictionary / np.linalg.norm(dictionary, axis=1)[:, None]


# ── tools/stainnorm.py ───────────────────────────────────────────────────────


class StainNormalizer:
    """Base class shared by Macenko/Vahadane normalizers (stain-matrix based)."""

    def __init__(self) -> None:
        self.extractor = None
        self.stain_matrix_target = None
        self.target_concentrations = None
        self.maxC_target = None

    @staticmethod
    def get_concentrations(img: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        od = rgb2od(img).reshape((-1, 3))
        x, _, _, _ = np.linalg.lstsq(stain_matrix.T, od.T, rcond=-1)
        return x.T

    def fit(self, target: np.ndarray) -> None:
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = self.get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))

    def transform(self, img: np.ndarray) -> np.ndarray:
        stain_matrix_source = self.extractor.get_stain_matrix(img)
        source_concentrations = self.get_concentrations(img, stain_matrix_source)
        max_c_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
        source_concentrations *= self.maxC_target / max_c_source
        trans = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        trans[trans > 255] = 255
        trans[trans < 0] = 0
        return trans.reshape(img.shape).astype(np.uint8)


class MacenkoNormalizer(StainNormalizer):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = MacenkoExtractor()


class VahadaneNormalizer(StainNormalizer):
    def __init__(self) -> None:
        super().__init__()
        self.extractor = VahadaneExtractor()


class ReinhardNormalizer:
    """Reinhard, Erik, et al. "Color transfer between images." IEEE CG&A 2001."""

    def __init__(self) -> None:
        self.target_means = None
        self.target_stds = None

    def fit(self, target: np.ndarray) -> None:
        self.target_means, self.target_stds = self.get_mean_std(target)

    def transform(self, img: np.ndarray) -> np.ndarray:
        chan1, chan2, chan3 = self.lab_split(img)
        means, stds = self.get_mean_std(img)
        norm1 = ((chan1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((chan2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((chan3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self.merge_back(norm1, norm2, norm3)

    @staticmethod
    def lab_split(img: np.ndarray):
        img = img.astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img_float = img.astype(np.float32)
        chan1, chan2, chan3 = cv2.split(img_float)
        chan1 /= np.asarray(2.55)
        chan2 -= np.asarray(128.0)
        chan3 -= np.asarray(128.0)
        return chan1, chan2, chan3

    @staticmethod
    def merge_back(chan1: np.ndarray, chan2: np.ndarray, chan3: np.ndarray) -> np.ndarray:
        chan1 = chan1 * 2.55
        chan2 = chan2 + 128.0
        chan3 = chan3 + 128.0
        img = np.clip(cv2.merge((chan1, chan2, chan3)), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    def get_mean_std(self, img: np.ndarray):
        img = img.astype("uint8")
        chan1, chan2, chan3 = self.lab_split(img)
        m1, sd1 = cv2.meanStdDev(np.asarray(chan1))
        m2, sd2 = cv2.meanStdDev(np.asarray(chan2))
        m3, sd3 = cv2.meanStdDev(np.asarray(chan3))
        means = float(m1[0][0]), float(m2[0][0]), float(m3[0][0])
        stds = float(sd1[0][0]), float(sd2[0][0]), float(sd3[0][0])
        return means, stds


def get_normalizer(method_name: str):
    """Return a fitted-ready normalizer instance for "reinhard"/"macenko"/"vahadane"."""
    method_name = method_name.lower()
    if method_name == "reinhard":
        return ReinhardNormalizer()
    if method_name == "macenko":
        return MacenkoNormalizer()
    if method_name == "vahadane":
        return VahadaneNormalizer()
    raise ValueError(f"Unsupported stain normalization method: {method_name!r}.")
