import pandas as pd


DEFAULT_REQUIRED_FIELDS = ["X", "Y", "DX", "DY"]


def clean_data(df: pd.DataFrame, required_fields: list[str] | None = None) -> pd.DataFrame:
    fields = required_fields or DEFAULT_REQUIRED_FIELDS
    return df.dropna(subset=fields).copy()


def restore_reference_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    restored = df.copy()
    restored["X_ref"] = restored["X"] - restored["DX"]
    restored["Y_ref"] = restored["Y"] - restored["DY"]
    return restored
