from pathlib import Path
import csv

import numpy as np
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
    """Read 2D (11 columns) or legacy 3D (15 columns) DIC exports strictly.

    The translated header can omit the final grid-column label; data width is
    therefore determined from the first record, never from the header alone.
    """
    with Path(file_path).open(encoding='gb18030', newline='') as stream:
        reader = csv.reader(stream)
        next(reader)
        first = next(reader)
    names_2d = ['Type', 'X', 'Y', 'DX', 'DY', 'Disp_Combined',
                'EXX', 'EYY', 'EXY', 'grid_row', 'grid_col']
    if len(first) == len(names_2d):
        names = names_2d
    elif len(first) == len(RAW_COLUMN_NAMES):
        names = RAW_COLUMN_NAMES
    else:
        raise ValueError(f'{file_path}: unsupported DIC record width {len(first)}')
    df = pd.read_csv(
        file_path,
        header=None,
        skiprows=1,
        names=names,
        encoding="gb18030",
        engine="python",
        index_col=False,
        on_bad_lines="error",
    )
    extracted = df[names[1:]].apply(pd.to_numeric, errors='raise')
    return extracted.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def remove_duplicate_grid_points(df: pd.DataFrame) -> pd.DataFrame:
    """Reject conflicting repeated points before keeping one row per grid ID."""
    keys = ['grid_row', 'grid_col']
    repeated = df[df.duplicated(keys, keep=False)]
    if not repeated.empty:
        columns = [c for c in ('X', 'Y', 'DX', 'DY') if c in df]
        variation = repeated.groupby(keys)[columns].nunique(dropna=False)
        if np.any(variation.to_numpy() > 1):
            raise ValueError('Repeated DIC grid IDs contain inconsistent coordinates/displacements')
    return df.drop_duplicates(keys).copy()


def find_stage_files(input_dir: Path, pattern: str = "stage_*.comdat_comdata.csv") -> list[Path]:
    return sorted(input_dir.glob(pattern))


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
