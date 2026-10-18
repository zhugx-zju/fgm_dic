"""Run platform inspection, calibrated forward comparison, or DIC inversion."""

import argparse
from datetime import datetime
import importlib
from pathlib import Path

from . import REPO_ROOT
from lib.research_workflows import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='experiment_config', help='Python configuration module')
    parser.add_argument('--mode', choices=('inspect', 'model', 'forward', 'inverse'), default='inspect')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)
    output = args.output or REPO_ROOT / 'results' / (
        f'experiment_{args.mode}_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    summary = run_experiment(cfg, output, args.mode)
    print(f"Status: {summary['status']}; results: {output.resolve()}")
    if args.mode == 'inverse':
        print(f"Converged: {summary['solver']['converged']}; {summary['solver']['message']}")
        if not summary['solver']['converged']:
            raise SystemExit(2)
    if args.mode == 'inspect':
        for name, record in summary['platforms'].items():
            print(f"{name}: {record['frame_count']} frames, {record['common_points']} points, "
                  f"export span={record['reference_span']}")


if __name__ == '__main__':
    main()
