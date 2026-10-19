"""Run the unified synthetic test: python -m archive.legacy_workflows.script.run_synthetic."""

import argparse
from datetime import datetime
import importlib
from pathlib import Path

from . import REPO_ROOT
from lib.research_workflows import run_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='synthetic_config', help='Python configuration module')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)
    output = args.output or REPO_ROOT / 'results' / ('synthetic_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    summary = run_synthetic(cfg, output)
    print(f"Results: {output.resolve()}")
    print(f"Converged: {summary['solver']['converged']}; {summary['solver']['message']}")
    print(f"Synthetic modulus MAE: {summary['modulus_mae_percent']:.3f}%")
    if not summary['solver']['converged']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
