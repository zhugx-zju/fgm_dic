from pathlib import Path

import pandas as pd


RAW_COLUMN_NAMES = [
    "Type",
    "X",
    "Y",
    "Z",
    "DX",
    "DY",
    "DZ",
    "Disp_Combined",
    "EXX",
    "EYY",
    "EXY",
    "E1",
    "E2",
    "grid_row",
    "grid_col",
]

COLUMN_NAMES = RAW_COLUMN_NAMES[1:]


def read_dic_csv(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file_path,
        header=None,
        skiprows=1,
        usecols=range(len(RAW_COLUMN_NAMES)),
        names=RAW_COLUMN_NAMES,
        encoding="gbk",
        engine="python",
        index_col=False,
        on_bad_lines="skip",
    )
    extracted = df[COLUMN_NAMES].reset_index(drop=True)
    return extracted


def find_stage_files(input_dir: Path, pattern: str = "stage_*.comdat_comdata.csv") -> list[Path]:
    return sorted(input_dir.glob(pattern))


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
