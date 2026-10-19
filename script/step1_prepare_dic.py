"""Step 1: prepare one experiment's DIC, registration, force and geometry."""

import argparse
from pathlib import Path

from lib.config_utils import load_python_configuration
from lib.dic_case_workflow import prepare_dic
from lib.dic_case_reporting import publish_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dic_case_config', help='Python module or .py path')
    parser.add_argument('--output', type=Path, help='Override 01_prepared directory; later stages then need --input')
    args = parser.parse_args()
    cfg = load_python_configuration(args.config)
    output = args.output or Path(cfg.CASE_OUTPUT)/'01_prepared'
    result = prepare_dic(cfg, output)
    print('Preparation:', result['status'], output.resolve())
    publish_stage(output.parent, '01_prepared', output, result)
    if result['status'] == 'needs_registration':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
