"""Validation with splits defined before combinatorial augmentation."""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .model import augmented_design
from .preprocessing import SpectralConfig


def metrics(y: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """RMSE, MAE and signed bias in uM; n counts augmented spectra, not batches."""
    y, predicted = np.asarray(y), np.asarray(predicted)
    if y.shape != predicted.shape or y.size == 0:
        raise ValueError("Reference and prediction must be non-empty and aligned.")
    error = predicted - y
    return {"n_augmented": int(y.size), "rmse_uM": float(np.sqrt(mean_squared_error(y, predicted))),
            "mae_uM": float(np.abs(error).mean()), "bias_uM": float(error.mean())}


def evaluate(frame: pd.DataFrame, metadata: pd.DataFrame,
             config: SpectralConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed models using level and joint-parent holdouts.

    LOCO excludes all calibration spectra at one concentration, while reusing
    the background library: this reproduces the source producer's diagnostic.
    Joint holdout additionally excludes a background fold. Neither establishes
    external laboratory or independent batch validation. Components are fixed.
    """
    calibration = metadata[metadata.role == "calibration"]
    backgrounds = list(metadata.loc[metadata.role == "background", "sample_id"])
    levels = sorted(calibration.reference_uM.unique())
    if len(levels) < 3 or len(backgrounds) < 2:
        raise ValueError("Validation requires at least three levels and two backgrounds.")
    all_cal = set(calibration.sample_id)
    all_bg = set(backgrounds)
    folds = [set(a.tolist()) for a in np.array_split(np.asarray(backgrounds), min(3, len(backgrounds)))]
    rows, predictions = [], []
    for scheme in ("leave_concentration_out", "joint_concentration_background_out"):
        bg_folds = [all_bg] if scheme == "leave_concentration_out" else folds
        for level in levels:
            held_cal = set(calibration.loc[calibration.reference_uM == level, "sample_id"])
            for fold, held_bg in enumerate(bg_folds):
                train_bg = all_bg if scheme == "leave_concentration_out" else all_bg - held_bg
                Xtr, ytr, train_lineage = augmented_design(frame, metadata, config, all_cal - held_cal, train_bg)
                Xte, yte, test_lineage = augmented_design(frame, metadata, config, held_cal, held_bg)
                if set(train_lineage.calibration_id) & set(test_lineage.calibration_id):
                    raise RuntimeError("Calibration-parent leakage.")
                if scheme.startswith("joint") and set(train_lineage.background_id) & set(test_lineage.background_id):
                    raise RuntimeError("Background-parent leakage.")
                # Baseline selects one derivative wavelength on training data only.
                centered = Xtr - Xtr.mean(axis=0)
                denom = np.sqrt((centered**2).sum(axis=0))
                score = np.abs(centered.T @ (ytr - ytr.mean())) / np.maximum(denom, 1e-30)
                chosen = int(np.argmax(score))
                pls = PLSRegression(n_components=config.n_components, scale=False).fit(Xtr, ytr)
                linear = LinearRegression().fit(Xtr[:, [chosen]], ytr)
                candidates = {"pls": pls.predict(Xte).ravel(),
                              "single_derivative_wavelength": linear.predict(Xte[:, [chosen]]).ravel(),
                              "training_mean": np.repeat(ytr.mean(), len(yte))}
                for model_name, pred in candidates.items():
                    rows.append({"scheme": scheme, "model": model_name, "held_out_uM": level,
                                 "background_fold": fold, **metrics(yte, pred)})
                    block = test_lineage.copy()
                    block["scheme"], block["model"] = scheme, model_name
                    block["background_fold"] = fold
                    block["predicted_uM"] = pred
                    predictions.append(block)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)
