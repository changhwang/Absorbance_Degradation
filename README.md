# Absorbance Degradation Analysis

This repository is focused on UV-Vis absorbance degradation analysis, especially for time-resolved PDCBT film datasets. The main workflow preprocesses spectra, applies MCR-ALS deconvolution, estimates degradation kinetics, and generates aggregate thickness-dependent summary plots.

Legacy optimization and initial-point generation utilities were moved out of the tracked repository so this codebase stays focused on UV-Vis deconvolution only.

## Repository layout

- `main.py`: primary UV-Vis workflow using concentration-based first-order kinetics.
- `main_ver2.py`: alternate UV-Vis workflow using normalized signal decay.
- `spectral_evolution_analysis.py`: overlay-style spectral evolution analysis across sampled timepoints.
- `progressive_mcr_analysis.py`: sliding-window MCR-ALS analysis and spectral export.
- `uvvis_shared.py`: shared UV-Vis loading, preprocessing, kinetics export, and plotting helpers.
- `thickness_normalisation_visualisation.py`: standalone entry point for kinetics summary visualization.
- `data/`: raw and generated experiment data.

## UV-Vis workflow

1. Load UV-Vis CSV files from a target directory.
2. Remove extra columns such as `Index` and `Unnamed:*`.
3. Restrict the wavelength range to 300-800 nm.
4. Baseline-correct the absorbance traces and clean outliers.
5. Use MCR-ALS to separate reference and degraded spectral components.
6. Fit an early-time first-order degradation trend.
7. Save per-sample plots and aggregate `kinetics_data.csv`.
8. Generate thickness dependence, residual, log-log, pairplot, and 3D summary plots.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default workflow:

```bash
python main.py
```

Run the alternate kinetics workflow:

```bash
python main_ver2.py
```

Run the spectral evolution overlay workflow:

```bash
python spectral_evolution_analysis.py
```

## Notes

- The scripts currently default to datasets under `data/uvvis/...`.
- Generated plots are written next to the source data unless otherwise noted.
- `main.py` and `main_ver2.py` share preprocessing and plotting helpers through `uvvis_shared.py`.
