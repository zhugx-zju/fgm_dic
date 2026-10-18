from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io_utils import remove_duplicate_grid_points


@dataclass(frozen=True)
class StructuredGrid:
    row_values: np.ndarray
    col_values: np.ndarray
    fields: dict[str, np.ndarray]


def reconstruct_structured_grid(
    df: pd.DataFrame,
    fields: tuple[str, ...] = ("X", "Y", "DX", "DY", "X_ref", "Y_ref"),
) -> dict[str, np.ndarray]:
    missing = [field for field in ("grid_row", "grid_col", *fields) if field not in df.columns]
    if missing:
        raise ValueError(f"Cannot reconstruct structured grid. Missing columns: {missing}")

    clean_df = df.dropna(subset=["grid_row", "grid_col"]).copy()
    clean_df = remove_duplicate_grid_points(clean_df)
    clean_df["grid_row"] = clean_df["grid_row"].astype(int)
    clean_df["grid_col"] = clean_df["grid_col"].astype(int)

    row_values = np.sort(clean_df["grid_row"].unique())
    col_values = np.sort(clean_df["grid_col"].unique())

    result: dict[str, np.ndarray] = {
        "grid_row": row_values,
        "grid_col": col_values,
    }
    expected_shape = (len(row_values), len(col_values))

    for field in fields:
        pivot = clean_df.pivot_table(index="grid_row", columns="grid_col", values=field, aggfunc="first")
        pivot = pivot.reindex(index=row_values, columns=col_values)
        values = pivot.to_numpy(dtype=float)
        if values.shape != expected_shape:
            raise ValueError(f"Unexpected grid shape for field {field}: {values.shape} != {expected_shape}")
        result[field] = values

    return result
