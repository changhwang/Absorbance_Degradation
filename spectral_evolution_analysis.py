from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymcr.constraints import ConstraintNonneg
from pymcr.mcr import McrAR

from uvvis_shared import load_uvvis_csv, preprocess_data


DEFAULT_DATA_DIRECTORY = Path("data") / "uvvis" / "spectral_deconvolution_overtime"
DEFAULT_SKIP_HOURS = 2


def plot_spectral_evolution(df: pd.DataFrame, output_base: str | Path, skip_hours: int = 1) -> None:
    """Run MCR-ALS against each sampled timepoint and save overlay and summary outputs."""
    try:
        times = df.columns[1:].astype(float) / 3600
        wavelengths = df["Wavelength"]
        initial_spectrum = df.iloc[:, 1].to_numpy()

        deconvoluted_spectra = []
        time_points = []

        print(f"\nProcessing {Path(output_base).stem}...")
        total_times = len(times)
        processed_times = 0

        for index, time_point in enumerate(times):
            if time_point % skip_hours >= 0.1:
                continue

            processed_times += 1
            print(f"\rProgress: {processed_times}/{total_times} time points processed", end="")
            current_spectrum = df.iloc[:, index + 1].to_numpy()

            mcr = McrAR(
                c_regr="OLS",
                st_regr="NNLS",
                c_constraints=[ConstraintNonneg()],
                max_iter=500,
                tol_increase=2.0,
                tol_n_increase=20,
                tol_err_change=1e-8,
            )

            try:
                data_matrix = np.column_stack([initial_spectrum, current_spectrum])
                st_init = np.column_stack([initial_spectrum, current_spectrum])
                mcr.fit(data_matrix, ST=st_init)

                if hasattr(mcr, "ST_"):
                    deconvoluted_spectra.append(mcr.ST_)
                    time_points.append(time_point)
                    print(f"\nSuccessfully processed time point {time_point:.1f}h")
            except Exception as error:
                print(f"\nError in MCR-ALS at time {time_point}: {error}")

        if not deconvoluted_spectra:
            return

        output_base = Path(output_base)
        time_indices = np.argsort(time_points)
        sorted_times = np.array(time_points)[time_indices]
        sorted_spectra = np.array(deconvoluted_spectra)[time_indices]
        midpoint = len(sorted_times) // 2
        legend_positions = {0, midpoint, len(sorted_times) - 1}

        plt.figure(figsize=(12, 8))
        cmap = plt.cm.viridis
        for index, time_point in enumerate(sorted_times):
            color = cmap(index / len(sorted_times))
            label = f"t = {time_point:.1f}h (Reference)" if index in legend_positions else None
            plt.plot(wavelengths, sorted_spectra[index][:, 0], color=color, linestyle="-", alpha=0.7, label=label)

        for index, time_point in enumerate(sorted_times):
            color = cmap(index / len(sorted_times))
            label = f"t = {time_point:.1f}h (Degraded)" if index in legend_positions else None
            plt.plot(wavelengths, sorted_spectra[index][:, 1], color=color, linestyle="--", alpha=0.7, label=label)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorbance")
        plt.title(f"Spectral Evolution: {output_base.stem}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.savefig(output_base.with_name(f"{output_base.stem}_deconvolution_overlay.png"), bbox_inches="tight", dpi=300)
        plt.close()

        summary_df = pd.DataFrame(
            {
                "Time (h)": sorted_times,
                "Max Reference": [spectrum[:, 0].max() for spectrum in sorted_spectra],
                "Max Degraded": [spectrum[:, 1].max() for spectrum in sorted_spectra],
            }
        )
        summary_df.to_csv(output_base.with_name(f"{output_base.stem}_summary.csv"), index=False)
        print(f"\nSuccessfully saved plot and summary for {output_base.stem}")
    except Exception as error:
        print(f"\nError in spectral evolution analysis: {error}")
        plt.close()


def run_directory_analysis(directory: str | Path = DEFAULT_DATA_DIRECTORY, skip_hours: int = DEFAULT_SKIP_HOURS) -> None:
    directory_path = Path(directory)
    results_dir = directory_path / "spectral_evolution_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(directory_path.glob("*.csv")):
        try:
            df = preprocess_data(load_uvvis_csv(file_path))
            plot_spectral_evolution(df, results_dir / file_path.stem, skip_hours=skip_hours)
            print(f"Completed processing {file_path.name}")
        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")

    print("\nAnalysis completed!")


if __name__ == "__main__":
    run_directory_analysis()
