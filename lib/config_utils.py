"""Explicit configuration and frozen preparation location."""

from pathlib import Path
import importlib
import importlib.util


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
