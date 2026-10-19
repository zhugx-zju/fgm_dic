"""Step 3: free 2D DIC inversion and held-out prediction; no calibrated forward fit."""

import argparse
from pathlib import Path

from lib.config_utils import prepared_input
from lib.dic_case_workflow import inverse_dic
from lib.dic_case_reporting import publish_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='01_prepared directory; defaults to CASE_OUTPUT in --config')
    parser.add_argument('--config', default='dic_case_config', help='Locates CASE_OUTPUT only; inputs stay frozen')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = prepared_input(args.input, args.config)
    output = args.output or source.parent/'03_inverse'
    result = inverse_dic(source, output)
    print('Inverse:', result['status'], output.resolve())
    publish_stage(source.parent, '03_inverse', output, result)
    if result['status'] != 'computed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
