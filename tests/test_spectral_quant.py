"""Unit and integration checks; generated fixtures are not experimental data."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from spectral_quant import SpectralConfig, SpectralPLS
from spectral_quant.cli import CASE_FILES, load_case, validate_inputs
from spectral_quant.model import augmented_design
from spectral_quant.preprocessing import canonical_frame, correct_and_crop
from spectral_quant.validation import evaluate, metrics


def fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a deterministic mathematical fixture for software tests only."""
    wavelength = np.arange(200.0, 801.0)
    drug = np.exp(-((wavelength - 310.0) / 15.0)**2)
    frame = pd.DataFrame({"wavelength_nm": wavelength})
    rows = []
    for i, c in enumerate((0.0, 1.0, 3.0, 6.0)):
        name = f"cal_{i}"
        frame[name] = c * drug + 0.1
        rows.append({"sample_id": name, "role": "calibration", "reference_uM": c})
    for i in range(3):
        name = f"bg_{i}"
        frame[name] = (i + 1) * (300.0 / wavelength)**2
        rows.append({"sample_id": name, "role": "background", "reference_uM": np.nan})
    return frame, pd.DataFrame(rows)


def write_case(folder: Path, observation_text: str | None = None,
               observation_meta: pd.DataFrame | None = None) -> Path:
    """Write the fixture as a hashed case directory; optional overrides are tampered inputs."""
    frame, metadata = fixture()
    frame.to_csv(folder / "training_spectra.csv", index=False)
    metadata.to_csv(folder / "training_metadata.csv", index=False)
    observations = frame[["wavelength_nm", "cal_1", "cal_2"]].rename(
        columns={"cal_1": "obs_1", "cal_2": "obs_2"})
    (folder / "observation_spectra.csv").write_text(
        observation_text or observations.to_csv(index=False), encoding="utf-8")
    if observation_meta is None:
        observation_meta = pd.DataFrame({"sample_id": ["obs_1", "obs_2"], "role": ["blank", "supernatant"]})
    observation_meta.to_csv(folder / "observation_metadata.csv", index=False)
    files = [{"path": name, "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}
             for name in CASE_FILES]
    manifest = {"case": "fixture", "concentration_unit": "uM", "absorbance_unit": "AU",
                "wavelength_unit": "nm", "files": files}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


class CaseLoadingTests(unittest.TestCase):
    def test_valid_case_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observation_meta = load_case(write_case(Path(tmp)))[4]
            self.assertEqual(list(observation_meta.sample_id), ["obs_1", "obs_2"])

    def test_missing_observation_metadata_fails(self) -> None:
        meta = pd.DataFrame({"sample_id": ["obs_1"], "role": ["blank"]})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            load_case(write_case(Path(tmp), observation_meta=meta))

    def test_extra_observation_metadata_fails(self) -> None:
        meta = pd.DataFrame({"sample_id": ["obs_1", "obs_2", "obs_3"], "role": ["blank"] * 3})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            load_case(write_case(Path(tmp), observation_meta=meta))

    def test_duplicate_spectrum_header_fails(self) -> None:
        frame = fixture()[0]
        text = "wavelength_nm,obs_1,obs_2,obs_1\n" + "".join(
            f"{w},{a},{b},{a}\n" for w, a, b in zip(frame.wavelength_nm, frame.cal_1, frame.cal_2))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            load_case(write_case(Path(tmp), observation_text=text))

    def test_manifest_without_entry_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp))
            manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"] = manifest["files"][1:]
            (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_inputs(case)

    def test_manifest_unit_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp))
            manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
            manifest["concentration_unit"] = "nM"
            (case / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_inputs(case)


class PreprocessingTests(unittest.TestCase):
    def test_descending_axis_is_equivalent(self) -> None:
        frame, _ = fixture()
        a = correct_and_crop(frame, SpectralConfig())
        b = correct_and_crop(frame.iloc[::-1], SpectralConfig())
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_duplicate_wavelength_fails(self) -> None:
        frame, _ = fixture()
        with self.assertRaises(ValueError):
            canonical_frame(pd.concat([frame, frame.iloc[:1]]))

    def test_missing_value_fails(self) -> None:
        frame, _ = fixture()
        frame.loc[20, "cal_0"] = np.nan
        with self.assertRaises(ValueError):
            canonical_frame(frame)

    def test_infinite_value_fails(self) -> None:
        frame, _ = fixture()
        frame.loc[20, "cal_0"] = np.inf
        with self.assertRaises(ValueError):
            canonical_frame(frame)

    def test_nonunit_grid_fails(self) -> None:
        frame, _ = fixture()
        with self.assertRaises(ValueError):
            correct_and_crop(frame.iloc[::2], SpectralConfig())

    def test_missing_window_fails(self) -> None:
        frame, _ = fixture()
        with self.assertRaises(ValueError):
            correct_and_crop(frame[frame.wavelength_nm > 275], SpectralConfig())

    def test_constant_offset_is_removed(self) -> None:
        frame, _ = fixture()
        shifted = frame.copy()
        shifted.iloc[:, 1:] += 3.0
        np.testing.assert_allclose(correct_and_crop(frame, SpectralConfig())[1],
                                   correct_and_crop(shifted, SpectralConfig())[1], atol=1e-14)

    def test_explicit_minimum_fallback(self) -> None:
        frame, _ = fixture()
        frame = frame[frame.wavelength_nm <= 700]
        _, values, _ = correct_and_crop(frame, SpectralConfig())
        self.assertTrue(np.isfinite(values).all())

    def test_invalid_config_fails(self) -> None:
        for kwargs in ({"window_length": 8}, {"derivative": 3}, {"n_components": 0},
                       {"turbidity_factors": (-1.0,)}, {"wavelength_min_nm": 400}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SpectralConfig(**kwargs)


class ModelTests(unittest.TestCase):
    def test_augmentation_count_and_parent_ids(self) -> None:
        frame, metadata = fixture()
        X, y, lineage = augmented_design(frame, metadata, SpectralConfig())
        self.assertEqual(X.shape, (4 * 3 * 5, 111))
        self.assertEqual(len(y), len(lineage))
        self.assertEqual(lineage.calibration_id.nunique(), 4)
        self.assertEqual(lineage.background_id.nunique(), 3)

    def test_duplicate_metadata_fails(self) -> None:
        frame, metadata = fixture()
        with self.assertRaises(ValueError):
            augmented_design(frame, pd.concat([metadata, metadata.iloc[:1]]), SpectralConfig())

    def test_missing_metadata_fails(self) -> None:
        frame, metadata = fixture()
        with self.assertRaises(ValueError):
            augmented_design(frame, metadata.iloc[:-1], SpectralConfig())

    def test_unknown_training_role_fails(self) -> None:
        frame, metadata = fixture()
        metadata.loc[0, "role"] = "unrecognized"
        with self.assertRaises(ValueError):
            augmented_design(frame, metadata, SpectralConfig())

    def test_negative_reference_fails(self) -> None:
        frame, metadata = fixture()
        metadata.loc[0, "reference_uM"] = -1.0
        with self.assertRaises(ValueError):
            augmented_design(frame, metadata, SpectralConfig())

    def test_empty_parent_split_fails(self) -> None:
        frame, metadata = fixture()
        with self.assertRaises(ValueError):
            augmented_design(frame, metadata, SpectralConfig(), keep_background=set())

    def test_unfitted_prediction_fails(self) -> None:
        with self.assertRaises(ValueError):
            SpectralPLS().predict(fixture()[0])

    def test_serialization_roundtrip(self) -> None:
        frame, metadata = fixture()
        model = SpectralPLS().fit(frame, metadata)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            model.save(path)
            np.testing.assert_allclose(model.predict(frame), SpectralPLS.load(path).predict(frame), atol=1e-12)

    def test_signed_prediction_is_not_clipped(self) -> None:
        frame, metadata = fixture()
        model = SpectralPLS().fit(frame, metadata)
        negative = frame[["wavelength_nm"]].copy()
        negative["negative_fixture"] = -frame.cal_1
        self.assertLess(model.predict(negative).iloc[0], 0)

    def test_validation_pooled_from_predictions(self) -> None:
        frame, metadata = fixture()
        scores, pred = evaluate(frame, metadata, SpectralConfig())
        self.assertEqual(set(scores.scheme), {"leave_concentration_out", "joint_concentration_background_out"})
        self.assertEqual(set(scores.model), {"pls", "single_derivative_wavelength", "training_mean"})
        self.assertTrue(np.isfinite(scores.rmse_uM).all())
        self.assertEqual(len(pred[pred.model == "pls"]), 2 * 4 * 3 * 5)

    def test_endpoint_folds_are_extrapolation(self) -> None:
        frame, metadata = fixture()
        scores, pred = evaluate(frame, metadata, SpectralConfig())
        region = scores.groupby("held_out_uM").region.unique().map(tuple).to_dict()
        self.assertEqual(region, {0.0: ("extrapolation",), 1.0: ("interpolation",),
                                  3.0: ("interpolation",), 6.0: ("extrapolation",)})
        self.assertFalse(pred.region.isna().any())

    def test_metrics_bias_and_count(self) -> None:
        result = metrics(np.array([0.0, 1.0]), np.array([-1.0, 0.0]))
        self.assertEqual(result["rmse_uM"], 1)
        self.assertEqual(result["bias_uM"], -1)
        self.assertEqual(result["n_augmented"], 2)


class RealDataTests(unittest.TestCase):
    def test_real_case_hashes_and_producer_parity(self) -> None:
        project = Path(__file__).resolve().parents[1]
        case = project / "data" / "talazoparib_nanode"
        manifest = validate_inputs(case)
        self.assertLess(manifest["producer_parity_max_abs_uM"], 1e-8)
        model = SpectralPLS().fit(pd.read_csv(case / "training_spectra.csv"),
                                  pd.read_csv(case / "training_metadata.csv"))
        observations = pd.read_csv(case / "observation_spectra.csv")
        parity = pd.read_csv(project / "reports" / "producer_parity.csv")
        actual = model.predict(observations).reindex(parity.sample_id).to_numpy()
        np.testing.assert_allclose(actual, parity.source_producer_uM, rtol=1e-9, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
