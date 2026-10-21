"""Explicit configuration and frozen preparation location."""

from pathlib import Path
import hashlib
import importlib
import importlib.util
import json


def configuration_fingerprint(configuration) -> str:
    """Hash requested values before DIC-derived geometry is resolved/frozen."""
    def serialize(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError(f'Unsupported configuration value: {type(value).__name__}')

    values = {key: value for key, value in vars(configuration).items() if key.isupper()}
    encoded = json.dumps(values, default=serialize, sort_keys=True, allow_nan=False,
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_python_configuration(name: str):
    """Read the saved user file afresh, including repeated runs in one IDE."""
    if not name.endswith('.py'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not spec.origin.endswith('.py'):
            raise ValueError(f'Cannot locate Python configuration: {name}')
        name = spec.origin
    path = Path(name).resolve()
    spec = importlib.util.spec_from_file_location('dic_case_user_config', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot load Python configuration: {path}')
    configuration = importlib.util.module_from_spec(spec)
    # Avoid both the module cache and same-second bytecode caching: the source
    # just saved by the user is the requested configuration we fingerprint.
    exec(compile(path.read_bytes(), str(path), 'exec'), vars(configuration))
    return configuration

def prepared_input(path: Path | None, config: str) -> Path:
    """Configuration locates a case; scientific inputs still come from frozen step 1."""
    if path is not None:
        return path.resolve()
    cfg = load_python_configuration(config)
    source = Path(cfg.CASE_OUTPUT).resolve()/'01_prepared'
    record = source/'run_metadata.json'
    if not record.exists():
        raise FileNotFoundError(f'未找到准备数据：{source}。请先运行第1步，或用 --input 指定已有准备目录。')
    metadata = json.loads(record.read_text(encoding='utf-8'))
    fingerprint = metadata.get('requested_configuration_sha256')
    if hasattr(cfg, 'DATA_MODE') and fingerprint != configuration_fingerprint(cfg):
        raise ValueError('当前配置与第1步冻结配置不一致。请更换RUN_NAME后重新执行第1步；'
                         '若要使用原配置的已有结果，请用 --input 显式指定原01_prepared目录。')
    frozen = metadata['configuration']
    mode = frozen.get('DATA_MODE', 'synthetic' if metadata.get('synthetic_observations') else 'dic')
    print(f"冻结输入：data={mode}, mesh={frozen.get('MESH_SOURCE')}, "
          f"regularization={frozen['REGULARIZATION_MODE']}\n保存位置：{source}")
    return source
