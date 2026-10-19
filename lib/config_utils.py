from dataclasses import dataclass
from pathlib import Path
import importlib
import importlib.util


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


def load_python_configuration(name: str):
    """Load an explicit Python module or file without changing the working directory."""
    if not name.endswith('.py'):
        return importlib.import_module(name)
    path = Path(name).resolve()
    spec = importlib.util.spec_from_file_location('dic_case_user_config', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load Python configuration: {path}')
    configuration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration)
    return configuration


def prepared_input(path: Path | None, config: str) -> Path:
    """Configuration locates a case; scientific inputs still come from frozen step 1."""
    return path.resolve() if path is not None else Path(load_python_configuration(config).CASE_OUTPUT).resolve()/'01_prepared'
