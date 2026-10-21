"""Command-line reproduction of the private talazoparib/nanoDE case study."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from .model import SpectralPLS
from .preprocessing import SpectralConfig, correct_and_crop
from .validation import evaluate, metrics


def validate_inputs(case: Path) -> dict:
    """Check all supplied input hashes before executing any model."""
    manifest = json.loads((case / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = case / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {path.name}")
    return manifest


def run(case: Path, output: Path) -> dict:
    """Train and evaluate measured spectra; save diagnostics without clipping."""
    if output.resolve() == case.resolve() or case.resolve() in output.resolve().parents:
        raise ValueError("Outputs must not overwrite the input dataset directory.")
    manifest = validate_inputs(case)
    frame = pd.read_csv(case / "training_spectra.csv")
    metadata = pd.read_csv(case / "training_metadata.csv")
    observations = pd.read_csv(case / "observation_spectra.csv")
    observation_meta = pd.read_csv(case / "observation_metadata.csv")
    config = SpectralConfig()
    model = SpectralPLS(config).fit(frame, metadata)
    result = observation_meta.merge(model.predict(observations).rename_axis("sample_id").reset_index(),
                                    on="sample_id", validate="one_to_one")
    scores, predictions = evaluate(frame, metadata, config)
    summary = []
    for (scheme, name), block in predictions.groupby(["scheme", "model"]):
        summary.append({"scheme": scheme, "model": name,
                        **metrics(block.reference_uM.to_numpy(), block.predicted_uM.to_numpy())})
    blanks = result[result.role == "blank"]
    report = {"configuration": config.to_dict(), "validation": summary,
              "blank": {"n_measured_spectra": len(blanks),
                        "mean_uM": float(blanks.predicted_uM.mean()),
                        "sd_uM": float(blanks.predicted_uM.std(ddof=1))},
              "counts": {"calibration_spectra": int((metadata.role == "calibration").sum()),
                         "background_spectra": int((metadata.role == "background").sum()),
                         "augmented_training_spectra": len(model.lineage_),
                         "observation_spectra": len(result)},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "scipy": scipy.__version__,
                           "scikit_learn": sklearn.__version__},
              "source_manifest": manifest["case"],
              "interpretation": "Augmented holdout diagnostics, not independent experimental batches. Nonblank supernatant input concentrations are not ground-truth residual concentrations."}
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "validation_by_fold.csv", index=False)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    result.to_csv(output / "supernatant_predictions.csv", index=False)
    model.lineage_.to_csv(output / "augmentation_lineage.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    model.save(output / "model.joblib")
    if not np.allclose(SpectralPLS.load(output / "model.joblib").predict(observations),
                       model.predict(observations), rtol=0, atol=1e-12):
        raise RuntimeError("Serialized-model predictions changed.")
    make_figures(frame, metadata, config, predictions, result, output)
    return report


def make_figures(frame: pd.DataFrame, metadata: pd.DataFrame, config: SpectralConfig,
                 predictions: pd.DataFrame, result: pd.DataFrame, output: Path) -> None:
    """Write diagnostic PDF/SVG plots with explicit units and sampling labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    grid, values, names = correct_and_crop(frame, config)
    lookup = dict(zip(names, values))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for role, ax in zip(("calibration", "background"), axes):
        cmap = plt.get_cmap("viridis")
        maximum = metadata.loc[metadata.role == "calibration", "reference_uM"].max()
        for item in metadata[metadata.role == role].itertuples():
            color = cmap(item.reference_uM / maximum) if role == "calibration" else "#666666"
            ax.plot(grid, lookup[item.sample_id], linewidth=0.6, color=color)
        if role == "calibration":
            scalar = plt.cm.ScalarMappable(norm=plt.Normalize(0, maximum), cmap=cmap)
            fig.colorbar(scalar, ax=ax, label="Calibration reference (uM)")
        ax.set(xlabel="Wavelength (nm)", ylabel="Baseline-corrected absorbance (AU)", title=f"Measured {role} spectra")
    save_figure(fig, output / "measured_spectra")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for scheme, ax in zip(predictions.scheme.unique(), axes):
        b = predictions[(predictions.scheme == scheme) & (predictions.model == "pls")]
        # Points are means over augmented descendants; bars show their SD, not CI.
        g = b.groupby("reference_uM").predicted_uM.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt="o", color="#0072B2", capsize=3)
        bounds = [min(0, float(g["mean"].min())), max(float(g.index.max()), float(g["mean"].max()))]
        ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Held-out reference (uM)", ylabel="Prediction (uM)",
               title="Concentration holdout" if scheme.startswith("leave") else "Concentration + background holdout")
    fig.suptitle("PLS augmented holdouts: mean ± SD across descendants")
    save_figure(fig, output / "held_out_predictions")
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    b = result[result.role == "blank"]
    ax.scatter(np.arange(1, len(b) + 1), b.predicted_uM, color="#D55E00")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Blank spectrum index", ylabel="Signed prediction (uM)", title="Measured matrix blanks (not clipped)")
    save_figure(fig, output / "matrix_blanks")


def save_figure(fig: object, path: Path) -> None:
    """Save vector and raster previews on opaque backgrounds."""
    import matplotlib.pyplot as plt
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), facecolor="white", transparent=False, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the configured case from any working directory with explicit paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/talazoparib_nanode"))
    parser.add_argument("--output", type=Path, default=Path("outputs/talazoparib_nanode"))
    args = parser.parse_args()
    report = run(args.data, args.output)
    print(json.dumps({"counts": report["counts"], "blank": report["blank"],
                      "validation": report["validation"]}, indent=2))


if __name__ == "__main__":
    main()
