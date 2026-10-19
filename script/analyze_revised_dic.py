"""Audit revised DIC and run explicitly conditional forward/inverse comparisons."""

import argparse
import importlib
from pathlib import Path

from lib.dic_revision_study import run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dic_revision_config')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    summary = run_study(importlib.import_module(args.config), args.output, smoke=args.smoke)
    print('Conditional result:', args.output.resolve())
    print('Coordinates verified:', summary['coordinates_verified'])
    print('Solver:', summary['solver'])


if __name__ == '__main__':
    main()
