from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pymcr.constraints import ConstraintNonneg
from pymcr.mcr import McrAR

from uvvis_shared import (
    find_peak_data,
    get_initial_guess,
    load_uvvis_csv,
    plot_thickness_dependence,
    preprocess_data,
    save_kinetics_data,
    visualize_kinetics_data,
)


DEFAULT_DATA_DIRECTORY = Path("data") / "uvvis" / "PDCBT_1day_data_success"
DEFAULT_SKIP = 40
TARGET_TIME_SECONDS = 24 * 3600


def _legend_interval(column_count: int) -> int:
    return max(1, column_count // 5)


def _normalize_signal(signal: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(signal, dtype=float)
    max_value = np.max(array) if array.size else 0.0
    return array if max_value == 0 else array / max_value


def process_uvvis_data(df: pd.DataFrame, name: str | Path, directory: str | Path, skip: int = 1) -> None:
    """Alternative UV-Vis workflow using normalized spectra and signal decay."""
    output_path = Path(name).with_suffix(".png")
    directory_path = Path(directory)

    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)

    wavelengths = df["Wavelength"]
    time_columns = df.columns[1:]
    end_time_hours = float(time_columns[-1]) / 3600

    plt.subplot(3, 1, 1)
    legend_elements = []
    interval = _legend_interval(len(time_columns))

    for column_index in range(1, len(df.columns), skip):
        time_seconds = float(df.columns[column_index])
        time_hours = time_seconds / 3600
        time_fraction = time_hours / end_time_hours if end_time_hours else 0
        color = (time_fraction * 0.5 + 0.5, time_fraction * 0.7, time_fraction)

        plt.plot(wavelengths, df[df.columns[column_index]], color=color, linewidth=1)
        if (
            column_index == 1
            or (column_index - 1) % interval == 0
            or column_index >= len(df.columns) - skip
        ):
            legend_elements.append(Line2D([0], [0], color=color, label=f"t = {time_hours:.1f} h"))

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.title("Raw UV-VIS Spectra Over Time")
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    mcr = McrAR(
        c_regr="OLS",
        st_regr="NNLS",
        c_constraints=[ConstraintNonneg()],
        max_iter=100,
        tol_increase=1.2,
        tol_n_increase=15,
    )

    try:
        initial_guess = get_initial_guess(df)
        spectra_matrix = df.iloc[:, 1:].to_numpy().T
        mcr.fit(spectra_matrix, ST=initial_guess)

        if len(mcr.C_opt_) == 0 or len(mcr.ST_) == 0:
            raise ValueError("MCR-ALS failed to produce valid results.")

        time_points = time_columns.astype(float)
        closest_idx = int(np.argmin(np.abs(time_points - TARGET_TIME_SECONDS)))
        closest_time_hours = time_points[closest_idx] / 3600

        plt.subplot(3, 1, 2)
        total_signal = _normalize_signal(mcr.D_opt_[closest_idx])
        best_fit_signal = _normalize_signal(mcr.D_opt_[closest_idx])
        reference_signal = _normalize_signal(mcr.C_opt_[closest_idx, 0] * mcr.ST_[0])
        degraded_signal = _normalize_signal(mcr.C_opt_[closest_idx, 1] * mcr.ST_[1])
        initial_spectrum = _normalize_signal(df.iloc[:, 1])

        plt.plot(wavelengths, total_signal, "k-", label="Total Signal")
        plt.plot(wavelengths, best_fit_signal, "y--", label="Best Fit Signal")
        plt.plot(wavelengths, reference_signal, "r--", label="Reference Signal")
        plt.plot(wavelengths, degraded_signal, "b--", label="Degraded Signal")
        plt.plot(wavelengths, initial_spectrum, "g:", label="Initial Spectrum")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Absorbance")
        plt.title(f"Normalized Spectral Deconvolution Results at t = {closest_time_hours:.1f} h")
        plt.legend()

        signals = mcr.D_opt_.max(axis=1)
        valid_mask = signals > 0
        valid_times = time_points[valid_mask]
        valid_signals = signals[valid_mask]
        log_signals = np.log(valid_signals)

        min_fitting_points = 5
        best_fit = {
            "rmse": float("inf"),
            "slope": None,
            "intercept": None,
            "uncertainty": float("inf"),
            "n_points": 0,
            "x_data": None,
            "y_data": None,
        }

        for n_points in range(min_fitting_points, len(valid_times)):
            x_data = valid_times[:n_points]
            y_data = log_signals[:n_points]
            non_nan_mask = ~np.isnan(y_data)
            x_fit = x_data[non_nan_mask]
            y_fit = y_data[non_nan_mask]

            if len(x_fit) < min_fitting_points:
                continue

            fit = np.polyfit(x_fit, y_fit, 1, full=True)
            slope, intercept = fit[0]
            residuals = fit[1]
            if len(residuals) == 0:
                continue

            rmse = np.sqrt(residuals[0] / len(x_fit))
            x_variance = np.var(x_fit)
            uncertainty = rmse / np.sqrt(x_variance * (len(x_fit) - 2))
            if uncertainty < best_fit["uncertainty"]:
                best_fit.update(
                    {
                        "rmse": rmse,
                        "slope": slope,
                        "intercept": intercept,
                        "uncertainty": uncertainty,
                        "n_points": n_points,
                        "x_data": x_fit,
                        "y_data": y_fit,
                    }
                )

        plt.subplot(3, 1, 3)
        if best_fit["slope"] is not None:
            x_data_hours = best_fit["x_data"] / 3600
            plt.scatter(x_data_hours, best_fit["y_data"], c="blue", s=10)

            x_fit_hours = np.linspace(x_data_hours[0], x_data_hours[-1], 100)
            y_fit = best_fit["slope"] * (x_fit_hours * 3600) + best_fit["intercept"]
            plt.plot(x_fit_hours, y_fit, "k-")

            stats_text = f"k = {abs(best_fit['slope']):.2e} s^-1\n"
            stats_text += f"Uncertainty: {best_fit['uncertainty']:.2e}\n"
            stats_text += f"RMSE: {best_fit['rmse']:.2e}"
            plt.text(
                0.05,
                0.95,
                stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            equation_text = r"$\bf{First\ Order\ Kinetics:}$" + "\n"
            equation_text += r"$\frac{dS}{dt} = -kS$" + "\n"
            equation_text += r"$ln(S) = -kt + b$" + "\n"
            equation_text += f"$y = {best_fit['slope']:.2e}x {best_fit['intercept']:+.2e}$"
            plt.text(
                0.95,
                0.95,
                equation_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=10,
            )

            peak_wavelength, peak_absorbance = find_peak_data(wavelengths, df.iloc[:, 1])
            save_kinetics_data(directory_path, output_path.stem, peak_wavelength, peak_absorbance, best_fit["slope"])
        else:
            plt.text(
                0.5,
                0.5,
                "No valid degradation trend found",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

        plt.xlabel("Time (h)")
        plt.ylabel("ln(Signal)")
        plt.title("Degradation Kinetics")
        plt.savefig(output_path)
        plt.close()
    except Exception as error:
        print(f"Error in MCR-ALS analysis for {output_path.name}: {error}")
        plt.close()


def run_directory_analysis(directory: str | Path = DEFAULT_DATA_DIRECTORY, skip: int = DEFAULT_SKIP) -> None:
    directory_path = Path(directory)
    kinetics_csv = directory_path / "kinetics_data.csv"
    if kinetics_csv.exists():
        kinetics_csv.unlink()

    for file_path in sorted(directory_path.glob("*.csv")):
        if file_path.name == kinetics_csv.name:
            continue
        try:
            df = preprocess_data(load_uvvis_csv(file_path))
            print(f"Processing {file_path.name}...")
            process_uvvis_data(df, directory_path / f"GRAPH_{file_path.stem}", directory_path, skip=skip)
            print(f"Completed processing {file_path.name}")
        except Exception as error:
            print(f"Error processing {file_path.name}: {error}")

    plot_thickness_dependence(directory_path)
    visualize_kinetics_data(kinetics_csv)


if __name__ == "__main__":
    run_directory_analysis()
