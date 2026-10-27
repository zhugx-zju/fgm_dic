"""Export an allowlisted source bundle without Git history or private experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def export(output: Path):
    root = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError(output)
    files = [root/name for name in ('config.py', 'README.md', 'requirements.txt')]
    for folder, pattern in [('fgm_asm','*.py'), ('lib','*.py'), ('script','*.py'), ('tools','*.py'), ('tests','*.py'),
                            ('data/example','*'), ('docs/public','*.md')]:
        files.extend(p for p in (root/folder).rglob(pattern) if p.is_file() and '__pycache__' not in p.parts)
    files = sorted(set(files))
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(root).as_posix())
        bundle.writestr('source_manifest.json', json.dumps(hashes, indent=2)+'\n')
    print(f'Exported {len(files)} files: {output.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'dist'/'fgm_dic_source.zip')
    export(parser.parse_args().output)
