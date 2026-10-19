"""Step 1: prepare one experiment's DIC, registration, force and geometry."""

import argparse
import importlib
import importlib.util
from pathlib import Path

from lib.dic_case_workflow import prepare_dic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dic_case_config', help='Python module or .py path')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.config.endswith('.py'):
        spec = importlib.util.spec_from_file_location('dic_case_user_config', Path(args.config).resolve())
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
    else:
        cfg = importlib.import_module(args.config)
    output = args.output or Path(cfg.CASE_OUTPUT)/'01_prepared'
    result = prepare_dic(cfg, output)
    print('Preparation:', result['status'], output.resolve())
    if result['status'] == 'needs_registration':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
