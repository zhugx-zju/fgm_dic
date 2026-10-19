"""Use maintained rot90 orientation, reuse the saved inverse and inspect all frames."""

import argparse
import importlib
from pathlib import Path

from archive.legacy_workflows.lib.dic_rot90_followup import run_rot90_followup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='archive.legacy_workflows.dic_rot90_config')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = run_rot90_followup(importlib.import_module(args.config), args.source, args.output)
    print('Output:', args.output.resolve())
    print('Comparison:', summary['comparison'])
    print('Frame diagnostics:', summary['frame_diagnostics'])


if __name__ == '__main__':
    main()
