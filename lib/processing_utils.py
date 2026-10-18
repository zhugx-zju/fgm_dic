from pathlib import Path

import pandas as pd

from lib.dic_fields import clean_data, restore_reference_coordinates
from lib.io_utils import read_dic_csv


def _normalize_corrected_cache(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "DX" not in normalized.columns and "DX_corr" in normalized.columns:
        normalized["DX"] = normalized["DX_corr"]
    if "DY" not in normalized.columns and "DY_corr" in normalized.columns:
        normalized["DY"] = normalized["DY_corr"]

    if "DX_raw" not in normalized.columns and "DX" in normalized.columns:
        normalized["DX_raw"] = normalized["DX"]
    if "DY_raw" not in normalized.columns and "DY" in normalized.columns:
        normalized["DY_raw"] = normalized["DY"]

    if "X_ref" not in normalized.columns and {"X", "DX_raw"}.issubset(normalized.columns):
        normalized["X_ref"] = normalized["X"] - normalized["DX_raw"]
    if "Y_ref" not in normalized.columns and {"Y", "DY_raw"}.issubset(normalized.columns):
        normalized["Y_ref"] = normalized["Y"] - normalized["DY_raw"]

    return normalized


def load_raw_processed(raw_csv_path: Path, required_fields: list[str] | None = None) -> pd.DataFrame:
    df = read_dic_csv(raw_csv_path)
    df = clean_data(df, required_fields=required_fields)
    return restore_reference_coordinates(df)


def load_exported_corrected_csv(
    corrected_csv_file: Path,
    raw_csv_path: Path,
    required_fields: list[str] | None = None,
) -> pd.DataFrame:
    # Legacy support for the older fixed-point-corrected CSV cache path.
    if not corrected_csv_file.exists():
        raise FileNotFoundError(f"Corrected CSV not found: {corrected_csv_file}")

    corrected_df = pd.read_csv(corrected_csv_file, encoding="utf-8-sig")
    corrected_df = _normalize_corrected_cache(corrected_df)
    if required_fields is None or set(required_fields).issubset(corrected_df.columns):
        return corrected_df

    raw_df = load_raw_processed(raw_csv_path, required_fields=required_fields)
    merged = raw_df.copy()

    for column in corrected_df.columns:
        if column in merged.columns or column in {"DX", "DY", "DX_raw", "DY_raw", "DX_corr", "DY_corr", "X_ref", "Y_ref"}:
            merged[column] = corrected_df[column].to_numpy()

    missing_fields = [] if required_fields is None else [field for field in required_fields if field not in merged.columns]
    if missing_fields:
        raise ValueError(f"Corrected CSV is missing required fields even after merge: {missing_fields}")
    return merged
