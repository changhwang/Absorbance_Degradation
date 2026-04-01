from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter


DEFAULT_WAVELENGTH_MIN = 300
DEFAULT_WAVELENGTH_MAX = 800
MIN_SIGNAL = 1e-10
KINETICS_FILENAME = "kinetics_data.csv"
PAIRPLOT_COLUMNS = [
    "Peak_Absorbance",
    "Rate_Constant_h-1",
    "Simple_Normalized_Rate",
    "Beer_Lambert_Normalized_Rate",
]


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clean_uvvis_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")].copy()
    cleaned = cleaned.drop(columns=["Index"], errors="ignore")
    return cleaned


def load_uvvis_csv(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    df = pd.read_csv(path, comment="#")
    df = clean_uvvis_dataframe(df)
    if "Wavelength" in df.columns:
        return df

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = [next(handle) for _ in range(11)]
    time_points = header[10].strip().split(",")[2:]
    df = pd.read_csv(path, skiprows=11, names=["Index", "Wavelength", *time_points])
    return clean_uvvis_dataframe(df)


def preprocess_data(
    df: pd.DataFrame,
    wavelength_min: float = DEFAULT_WAVELENGTH_MIN,
    wavelength_max: float = DEFAULT_WAVELENGTH_MAX,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    processed = clean_uvvis_dataframe(df)
    if "Wavelength" not in processed.columns:
        raise KeyError("Expected a 'Wavelength' column in the UV-Vis dataset.")

    processed = processed.copy()
    processed["Wavelength"] = pd.to_numeric(processed["Wavelength"], errors="coerce")
    signal_columns = [column for column in processed.columns if column != "Wavelength"]
    if not signal_columns:
        raise ValueError("Expected at least one absorbance column besides 'Wavelength'.")

    processed[signal_columns] = processed[signal_columns].apply(pd.to_numeric, errors="coerce")
    processed = processed.dropna(subset=["Wavelength"])
    processed = processed.loc[
        processed["Wavelength"].between(wavelength_min, wavelength_max)
    ].copy()

    signal_frame = processed[signal_columns]
    signal_frame = signal_frame.subtract(signal_frame.min(axis=0), axis=1)

    z_scores = stats.zscore(signal_frame, nan_policy="omit")
    if np.ndim(z_scores) == 1:
        z_scores = np.expand_dims(z_scores, axis=1)
    outlier_mask = np.abs(np.nan_to_num(z_scores, nan=0.0)) > zscore_threshold
    signal_frame = signal_frame.mask(outlier_mask)
    signal_frame = signal_frame.interpolate(method="linear", axis=0).bfill().ffill()
    signal_frame = signal_frame.clip(lower=0)

    processed.loc[:, signal_columns] = signal_frame
    return processed


def _resolve_window_length(length: int, preferred: int, polyorder: int) -> int:
    if length < 3:
        return 0

    window_length = min(preferred, length if length % 2 == 1 else length - 1)
    minimum_allowed = polyorder + 2
    if minimum_allowed % 2 == 0:
        minimum_allowed += 1
    if window_length < minimum_allowed:
        return 0 if minimum_allowed > length else minimum_allowed
    return window_length


def _smooth_signal(signal: np.ndarray, preferred_window: int = 15, polyorder: int = 3) -> np.ndarray:
    window_length = _resolve_window_length(len(signal), preferred_window, polyorder)
    if window_length == 0:
        return signal
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def get_initial_guess(df: pd.DataFrame) -> np.ndarray:
    initial_spectrum = df.iloc[:, 1].to_numpy(dtype=float)
    final_spectrum = df.iloc[:, -1].to_numpy(dtype=float)

    smoothed_initial = _smooth_signal(initial_spectrum)
    smoothed_final = _smooth_signal(final_spectrum)
    difference = np.maximum(smoothed_initial - smoothed_final, 0)

    initial_scale = np.max(smoothed_initial) or 1.0
    difference_scale = np.max(difference) or 1.0

    normalized_initial = np.maximum(smoothed_initial / initial_scale, MIN_SIGNAL)
    normalized_difference = np.maximum(difference / difference_scale, MIN_SIGNAL)
    return np.vstack([normalized_initial, normalized_difference])


def find_peak_data(wavelengths, first_spectrum) -> tuple[float, float]:
    peak_idx = int(np.argmax(first_spectrum))
    return float(wavelengths.iloc[peak_idx] if hasattr(wavelengths, "iloc") else wavelengths[peak_idx]), float(
        first_spectrum.iloc[peak_idx] if hasattr(first_spectrum, "iloc") else first_spectrum[peak_idx]
    )


def save_kinetics_data(
    directory: str | Path,
    filename: str,
    peak_wavelength: float,
    peak_absorbance: float,
    slope: float,
) -> Path:
    directory_path = ensure_directory(directory)
    csv_path = directory_path / KINETICS_FILENAME
    row = pd.DataFrame(
        {
            "Sample": [filename],
            "Peak_Wavelength_nm": [peak_wavelength],
            "Peak_Absorbance": [peak_absorbance],
            "Rate_Constant_h-1": [abs(slope)],
        }
    )

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["Sample"] != filename]
        data = pd.concat([existing, row], ignore_index=True)
    else:
        data = row

    data.to_csv(csv_path, index=False)
    return csv_path


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = np.divide(
        numerator.to_numpy(dtype=float),
        denominator.to_numpy(dtype=float),
        out=np.full(len(numerator), np.nan, dtype=float),
        where=np.abs(denominator.to_numpy(dtype=float)) > 0,
    )
    return pd.Series(values, index=numerator.index)


def _plot_linear_fit(ax, x: pd.Series, y: pd.Series, title: str, ylabel: str) -> None:
    ax.scatter(x, y)
    ax.set_xlabel("Peak Absorbance (a.u.)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return

    coefficients = np.polyfit(valid["x"], valid["y"], 1)
    trend = np.poly1d(coefficients)
    ax.plot(
        valid["x"],
        trend(valid["x"]),
        "r--",
        label=f"Linear fit\nSlope: {coefficients[0]:.2e}\nIntercept: {coefficients[1]:.2e}",
    )
    ax.legend()


def plot_thickness_dependence(directory: str | Path) -> Path | None:
    directory_path = Path(directory)
    csv_path = directory_path / KINETICS_FILENAME
    if not csv_path.exists():
        return None

    data = pd.read_csv(csv_path)
    if data.empty:
        return None

    data["Simple_Normalized_Rate"] = _safe_divide(
        data["Rate_Constant_h-1"], data["Peak_Absorbance"]
    )
    beer_lambert_term = 1 - np.exp(-data["Peak_Absorbance"])
    data["Beer_Lambert_Normalized_Rate"] = _safe_divide(
        data["Rate_Constant_h-1"], beer_lambert_term
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _plot_linear_fit(
        axes[0],
        data["Peak_Absorbance"],
        data["Rate_Constant_h-1"],
        "Raw Degradation Rate vs Film Thickness",
        "Rate Constant (h^-1)",
    )
    _plot_linear_fit(
        axes[1],
        data["Peak_Absorbance"],
        data["Simple_Normalized_Rate"],
        "Simple Normalized Rate vs Film Thickness",
        "Rate Constant / Absorbance (h^-1/abs)",
    )
    _plot_linear_fit(
        axes[2],
        data["Peak_Absorbance"],
        data["Beer_Lambert_Normalized_Rate"],
        "Beer-Lambert Normalized Rate vs Film Thickness",
        "Rate Constant / (1-exp(-abs)) (h^-1)",
    )

    figure_path = directory_path / "thickness_dependence.png"
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    data.to_csv(csv_path, index=False)
    return figure_path


def _save_pairplot(data: pd.DataFrame, output_path: Path) -> None:
    plot_data = data.dropna(subset=PAIRPLOT_COLUMNS)
    if plot_data.empty:
        return
    pairplot = sns.pairplot(plot_data[PAIRPLOT_COLUMNS])
    pairplot.figure.savefig(output_path)
    plt.close(pairplot.figure)


def _plot_loglog(ax, x: pd.Series, y: pd.Series, title: str, ylabel: str) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    valid = valid[(valid["x"] > 0) & (valid["y"] > 0)]
    if valid.empty:
        ax.set_title(f"{title}\n(No positive data)")
        ax.set_xlabel("Log(Peak Absorbance)")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return

    ax.loglog(valid["x"], valid["y"], "o")
    ax.set_xlabel("Log(Peak Absorbance)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)


def _plot_residuals(ax, x: pd.Series, y: pd.Series, title: str) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        ax.set_title(f"{title}\n(Not enough data)")
        ax.set_xlabel("Peak Absorbance")
        ax.set_ylabel("Residuals")
        return

    coefficients = np.polyfit(valid["x"], valid["y"], 1)
    fit = np.poly1d(coefficients)
    residuals = valid["y"] - fit(valid["x"])
    ax.scatter(valid["x"], residuals)
    ax.axhline(y=0, color="r", linestyle="--")
    ax.set_xlabel("Peak Absorbance")
    ax.set_ylabel("Residuals")
    ax.set_title(title)


def visualize_kinetics_data(csv_path: str | Path) -> Path | None:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return None

    data = pd.read_csv(csv_file)
    if data.empty:
        return None

    missing_columns = [column for column in PAIRPLOT_COLUMNS if column not in data.columns]
    if missing_columns:
        return None

    directory = csv_file.parent
    _save_pairplot(data, directory / "pairplot.png")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _plot_loglog(
        axes[0], data["Peak_Absorbance"], data["Rate_Constant_h-1"], "Log-Log Plot: Raw Rate", "Log(Rate Constant)"
    )
    _plot_loglog(
        axes[1],
        data["Peak_Absorbance"],
        data["Simple_Normalized_Rate"],
        "Log-Log Plot: Simple Normalized",
        "Log(Simple Normalized Rate)",
    )
    _plot_loglog(
        axes[2],
        data["Peak_Absorbance"],
        data["Beer_Lambert_Normalized_Rate"],
        "Log-Log Plot: Beer-Lambert Normalized",
        "Log(Beer-Lambert Normalized Rate)",
    )
    fig.tight_layout()
    fig.savefig(directory / "loglog_plots.png")
    plt.close(fig)

    unique_absorbance = data["Peak_Absorbance"].nunique(dropna=True)
    if unique_absorbance >= 2:
        bin_count = min(4, unique_absorbance)
        labels = ["Very Thin", "Thin", "Thick", "Very Thick"][:bin_count]
        data["Thickness_Bin"] = pd.qcut(
            data["Peak_Absorbance"],
            q=bin_count,
            labels=labels,
            duplicates="drop",
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        sns.boxplot(x="Thickness_Bin", y="Rate_Constant_h-1", data=data, ax=axes[0])
        sns.boxplot(x="Thickness_Bin", y="Simple_Normalized_Rate", data=data, ax=axes[1])
        sns.boxplot(x="Thickness_Bin", y="Beer_Lambert_Normalized_Rate", data=data, ax=axes[2])
        axes[0].set_title("Raw Rate by Thickness")
        axes[1].set_title("Simple Normalized Rate by Thickness")
        axes[2].set_title("Beer-Lambert Normalized Rate by Thickness")
        for axis in axes:
            axis.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(directory / "boxplots.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    _plot_residuals(axes[0], data["Peak_Absorbance"], data["Rate_Constant_h-1"], "Raw Rate Residuals")
    _plot_residuals(
        axes[1], data["Peak_Absorbance"], data["Simple_Normalized_Rate"], "Simple Normalized Residuals"
    )
    _plot_residuals(
        axes[2],
        data["Peak_Absorbance"],
        data["Beer_Lambert_Normalized_Rate"],
        "Beer-Lambert Normalized Residuals",
    )
    fig.tight_layout()
    fig.savefig(directory / "residual_plots.png")
    plt.close(fig)

    fig = plt.figure(figsize=(20, 6))
    axes = [
        fig.add_subplot(131, projection="3d"),
        fig.add_subplot(132, projection="3d"),
        fig.add_subplot(133, projection="3d"),
    ]
    scatter1 = axes[0].scatter(
        data["Peak_Absorbance"],
        data["Peak_Wavelength_nm"],
        data["Rate_Constant_h-1"],
        c=data["Rate_Constant_h-1"],
        cmap="viridis",
    )
    axes[0].set_xlabel("Peak Absorbance")
    axes[0].set_ylabel("Peak Wavelength (nm)")
    axes[0].set_zlabel("Rate Constant (h^-1)")
    axes[0].set_title("Raw Rate 3D Plot")
    fig.colorbar(scatter1, ax=axes[0])

    scatter2 = axes[1].scatter(
        data["Peak_Absorbance"],
        data["Peak_Wavelength_nm"],
        data["Simple_Normalized_Rate"],
        c=data["Simple_Normalized_Rate"],
        cmap="viridis",
    )
    axes[1].set_xlabel("Peak Absorbance")
    axes[1].set_ylabel("Peak Wavelength (nm)")
    axes[1].set_zlabel("Simple Normalized Rate")
    axes[1].set_title("Simple Normalized 3D Plot")
    fig.colorbar(scatter2, ax=axes[1])

    scatter3 = axes[2].scatter(
        data["Peak_Absorbance"],
        data["Peak_Wavelength_nm"],
        data["Beer_Lambert_Normalized_Rate"],
        c=data["Beer_Lambert_Normalized_Rate"],
        cmap="viridis",
    )
    axes[2].set_xlabel("Peak Absorbance")
    axes[2].set_ylabel("Peak Wavelength (nm)")
    axes[2].set_zlabel("Beer-Lambert Normalized Rate")
    axes[2].set_title("Beer-Lambert Normalized 3D Plot")
    fig.colorbar(scatter3, ax=axes[2])

    fig.tight_layout()
    output_path = directory / "3d_plots.png"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
