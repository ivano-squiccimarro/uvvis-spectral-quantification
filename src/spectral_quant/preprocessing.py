"""Deterministic spectral processing with explicit wavelength units (nm)."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for the existing talazoparib chromophore model.

    The two latent components and spectral window are fixed, not selected
    against this repository's evaluation results. Absorbance is in AU.
    """

    wavelength_min_nm: float = 270.0
    wavelength_max_nm: float = 380.0
    baseline_min_nm: float = 780.0
    baseline_max_nm: float = 800.0
    window_length: int = 9
    polyorder: int = 2
    derivative: int = 2
    n_components: int = 2
    turbidity_factors: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)

    def __post_init__(self) -> None:
        if self.wavelength_min_nm >= self.wavelength_max_nm:
            raise ValueError("Spectral window bounds must increase.")
        if self.window_length % 2 != 1 or self.window_length <= self.polyorder:
            raise ValueError("Savitzky-Golay window must be odd and exceed polyorder.")
        if not 0 <= self.derivative <= self.polyorder or self.n_components < 1:
            raise ValueError("Invalid derivative order or component count.")
        if not self.turbidity_factors or not all(np.isfinite(k) and k >= 0 for k in self.turbidity_factors):
            raise ValueError("Turbidity factors must be finite and non-negative.")

    def to_dict(self) -> dict:
        """Return serializable configuration with units in field names."""
        return asdict(self)


def canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide spectrum frame and sort its wavelength axis.

    Input has wavelength_nm in nm and one column per measured spectrum in AU.
    Missing values, duplicate wavelengths and duplicate sample IDs are errors.
    Descending exports are accepted but no interpolation is performed.
    """
    if "wavelength_nm" not in frame or frame.columns.duplicated().any():
        raise ValueError("Expected unique columns including wavelength_nm.")
    if len(frame.columns) < 2 or frame.empty:
        raise ValueError("At least one measured spectrum is required.")
    out = frame.apply(pd.to_numeric, errors="raise").sort_values("wavelength_nm").reset_index(drop=True)
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Non-finite wavelength or absorbance.")
    if out.wavelength_nm.duplicated().any():
        raise ValueError("Duplicate wavelengths are not allowed.")
    return out


def correct_and_crop(frame: pd.DataFrame, config: SpectralConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ascending grid (nm), sample-by-wavelength AU, and sample IDs.

    Subtract the median absorbance over 780-800 nm before cropping. As in the
    source producer, use the full-spectrum minimum if fewer than three tail
    points are available. This fallback is explicit in the method description.
    """
    frame = canonical_frame(frame)
    grid = frame.wavelength_nm.to_numpy()
    names = list(frame.columns[1:])
    values = frame[names].to_numpy().T
    tail = (grid >= config.baseline_min_nm) & (grid <= config.baseline_max_nm)
    offset = np.median(values[:, tail], axis=1) if tail.sum() >= 3 else values.min(axis=1)
    keep = (grid >= config.wavelength_min_nm) & (grid <= config.wavelength_max_nm)
    grid, values = grid[keep], (values - offset[:, None])[:, keep]
    if len(grid) < config.window_length:
        raise ValueError("Too few wavelengths for the configured smoothing window.")
    if not np.allclose(np.diff(grid), 1.0, rtol=0, atol=1e-8):
        raise ValueError("This calibrated preprocessing requires a 1-nm grid.")
    if not np.isclose(grid[0], config.wavelength_min_nm) or not np.isclose(grid[-1], config.wavelength_max_nm):
        raise ValueError("The full configured spectral window must be present.")
    return grid, values, names


def derivative(values: np.ndarray, config: SpectralConfig) -> np.ndarray:
    """Apply Savitzky-Golay differentiation on the wavelength axis (1-nm step)."""
    return savgol_filter(values, config.window_length, config.polyorder,
                         deriv=config.derivative, axis=1, delta=1.0)
