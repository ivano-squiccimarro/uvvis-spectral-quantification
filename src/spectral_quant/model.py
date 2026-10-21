"""Standalone PLS calibration and traceable augmentation of measured spectra."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from .preprocessing import SpectralConfig, correct_and_crop, derivative


def augmented_design(frame: pd.DataFrame, metadata: pd.DataFrame,
                     config: SpectralConfig, keep_calibration: set[str] | None = None,
                     keep_background: set[str] | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build sample-by-feature spectra from real calibration and backgrounds.

    Every descendant retains both parent IDs and its turbidity factor.
    Concentrations are in uM. The combinatorial count is not an experimental n.
    """
    grid, values, names = correct_and_crop(frame, config)
    del grid
    if metadata.sample_id.duplicated().any() or set(metadata.sample_id) != set(names):
        raise ValueError("Metadata and spectra must have a one-to-one sample mapping.")
    if not set(metadata.role).issubset({"calibration", "background"}):
        raise ValueError("Training roles must be calibration or background.")
    lookup = dict(zip(names, values))
    cal = metadata[metadata.role == "calibration"]
    bg = metadata[metadata.role == "background"]
    if keep_calibration is not None:
        cal = cal[cal.sample_id.isin(keep_calibration)]
    if keep_background is not None:
        bg = bg[bg.sample_id.isin(keep_background)]
    if cal.empty or bg.empty:
        raise ValueError("Calibration and background spectra are both required.")
    if not np.isfinite(cal.reference_uM.to_numpy(dtype=float)).all() or (cal.reference_uM < 0).any():
        raise ValueError("Calibration references must be finite non-negative uM.")
    rows, targets, lineage = [], [], []
    for d in cal.itertuples():
        for s in bg.itertuples():
            for k in config.turbidity_factors:
                rows.append(lookup[d.sample_id] + k * lookup[s.sample_id])
                targets.append(float(d.reference_uM))
                lineage.append({"calibration_id": d.sample_id, "background_id": s.sample_id,
                                "turbidity_factor": k, "reference_uM": float(d.reference_uM)})
    return derivative(np.asarray(rows), config), np.asarray(targets), pd.DataFrame(lineage)


class SpectralPLS:
    """PLS prediction in uM, preserving signed predictions for diagnostics."""

    def __init__(self, config: SpectralConfig | None = None) -> None:
        self.config = config or SpectralConfig()

    def fit(self, frame: pd.DataFrame, metadata: pd.DataFrame) -> "SpectralPLS":
        """Fit the fixed calibration using the measured parent spectra."""
        self.grid_, _, _ = correct_and_crop(frame, self.config)
        X, y, self.lineage_ = augmented_design(frame, metadata, self.config)
        if self.config.n_components > min(X.shape[1], X.shape[0] - 1):
            raise ValueError("Too many PLS components for the design.")
        self.estimator_ = PLSRegression(n_components=self.config.n_components, scale=False).fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """Predict uM on an identical grid; do not silently interpolate or clip."""
        if not hasattr(self, "estimator_"):
            raise ValueError("Fit or load a model before prediction.")
        grid, values, names = correct_and_crop(frame, self.config)
        if not np.array_equal(grid, self.grid_):
            raise ValueError("Prediction wavelengths do not match calibration.")
        return pd.Series(self.estimator_.predict(derivative(values, self.config)).ravel(),
                         index=names, name="predicted_uM")

    def save(self, path: str | Path) -> None:
        """Persist a trusted local model, including preprocessing and grid."""
        if not hasattr(self, "estimator_"):
            raise ValueError("Cannot save an unfitted model.")
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "SpectralPLS":
        """Load only a trusted joblib file; deserialization can execute code."""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("The file does not contain a SpectralPLS model.")
        return model
