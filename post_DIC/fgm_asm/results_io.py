"""
Centralized result-path and result-file handling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .config_types import coerce_forward_config


FORWARD_DATA_FILENAME = "forward_problem_data.pkl"
INVERSE_RESULTS_FILENAME = "inverse_results.pkl"
LCURVE_ANALYSIS_FILENAME = "lcurve_analysis.pkl"
RESULTS_FOLDER_GLOB = "Geo_*_Mesh_*_Alpha_*_Beta_*_Gamma_*"
LEGACY_FORWARD_DIR = "forward_results"
LEGACY_INVERSE_DIR = "inverse_results"
LEGACY_LCURVE_DIR = "inverse_results_lbfgs_lcurve"
CONFIG_SNAPSHOT_FILENAME = "config.py"


def _normalize_python_value(value: Any) -> Any:
    """Convert runtime values to plain Python literals suitable for repr()."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            _normalize_python_value(key): _normalize_python_value(val)
            for key, val in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_normalize_python_value(item) for item in value)
    if isinstance(value, list):
        return [_normalize_python_value(item) for item in value]
    if isinstance(value, set):
        return {_normalize_python_value(item) for item in value}
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def write_python_config_snapshot(output_dir: Path | str,
                                 sections: list[tuple[str, dict[str, Any]]],
                                 filename: str = CONFIG_SNAPSHOT_FILENAME) -> Path:
    """Write a small Python config snapshot for a saved inverse-result folder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Auto-generated configuration snapshot for this inverse run.",
        "# Saved together with the inverse results for later inspection.",
        "",
    ]

    for title, entries in sections:
        if not entries:
            continue
        lines.append(f"# {title}")
        for key, value in entries.items():
            lines.append(f"{key} = {repr(_normalize_python_value(value))}")
        lines.append("")

    file_path = output_dir / filename
    file_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return file_path


def find_latest_results_root(search_root: Path | str = ".") -> Path:
    """Find the latest top-level results folder."""
    search_root = Path(search_root)
    matching_folders = list(search_root.glob(RESULTS_FOLDER_GLOB))
    if matching_folders:
        return max(matching_folders, key=lambda path: path.stat().st_mtime)
    return search_root / LEGACY_INVERSE_DIR


def find_forward_data_path(search_root: Path | str = ".") -> tuple[Path, Path]:
    """Find the latest forward problem data file."""
    search_root = Path(search_root)
    matching_folders = list(search_root.glob(RESULTS_FOLDER_GLOB))
    if matching_folders:
        forward_data_folder = max(matching_folders, key=lambda path: path.stat().st_mtime)
        forward_data_path = forward_data_folder / FORWARD_DATA_FILENAME
    else:
        forward_data_folder = search_root / LEGACY_FORWARD_DIR
        forward_data_path = forward_data_folder / FORWARD_DATA_FILENAME

    if not forward_data_path.exists():
        raise FileNotFoundError(
            f"Forward data not found at {forward_data_path}\n"
            "Please run forward_job.py first!"
        )

    return forward_data_path, forward_data_folder


def load_forward_data(data_path: Path | str) -> dict[str, Any]:
    """Load forward problem data from a pickle file and normalize the config object."""
    data_path = Path(data_path)
    with open(data_path, "rb") as file:
        forward_data = pickle.load(file)

    if "config" in forward_data:
        forward_data["config"] = coerce_forward_config(forward_data["config"])

    return forward_data


def save_forward_data(forward_data: dict[str, Any], output_dir: Path | str,
                      filename: str = FORWARD_DATA_FILENAME) -> Path:
    """Save forward problem data to a pickle file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with open(file_path, "wb") as file:
        pickle.dump(forward_data, file)
    return file_path


def save_inverse_results(results: dict[str, Any], errors: dict[str, Any] | None, E_true,
                         output_dir: Path | str,
                         filename: str = INVERSE_RESULTS_FILENAME,
                         extra_data: dict[str, Any] | None = None) -> Path:
    """Save inverse-problem results to a pickle file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "results": results,
        "errors": errors,
        "E_true": E_true,
        "E_reconstructed": results["E_final"],
    }
    if extra_data:
        save_data.update(extra_data)

    file_path = output_dir / filename
    with open(file_path, "wb") as file:
        pickle.dump(save_data, file)
    return file_path


def save_lcurve_analysis(lcurve_results: dict[str, Any], output_dir: Path | str,
                         filename: str = LCURVE_ANALYSIS_FILENAME,
                         extra_data: dict[str, Any] | None = None) -> Path:
    """Save L-curve analysis data to a pickle file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {"lcurve_results": lcurve_results}
    if extra_data:
        save_data.update(extra_data)

    file_path = output_dir / filename
    with open(file_path, "wb") as file:
        pickle.dump(save_data, file)
    return file_path

