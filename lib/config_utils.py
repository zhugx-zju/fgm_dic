from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    results: Path


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    project_root = root or Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    return ProjectPaths(
        root=project_root,
        data=data_dir,
        results=project_root / "results",
    )


def ensure_output_dirs(paths: ProjectPaths) -> None:
    for path in (
        paths.data,
        paths.results,
    ):
        path.mkdir(parents=True, exist_ok=True)
