from pathlib import Path

from uvvis_shared import visualize_kinetics_data


DEFAULT_CSV_PATH = Path("data") / "uvvis" / "PDCBT_all_data" / "kinetics_data.csv"


if __name__ == "__main__":
    visualize_kinetics_data(DEFAULT_CSV_PATH)
