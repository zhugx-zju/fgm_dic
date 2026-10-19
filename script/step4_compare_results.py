"""Step 4: compare saved DIC, forward and inverse results; no recomputation."""

import argparse
from pathlib import Path

from lib.config_utils import prepared_input
from lib.dic_case_workflow import compare_dic_results
from lib.dic_case_reporting import publish_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='01_prepared directory; defaults to CASE_OUTPUT in --config')
    parser.add_argument('--config', default='dic_case_config', help='Locates CASE_OUTPUT only; inputs stay frozen')
    parser.add_argument('--forward', type=Path)
    parser.add_argument('--inverse', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    source = prepared_input(args.input, args.config)
    root = source.parent
    output = args.output or root/'04_comparison'
    result = compare_dic_results(source, args.forward or root/'02_forward',
                                  args.inverse or root/'03_inverse', output, plots=not args.no_plots)
    print('Comparison:', result['status'], output.resolve())
    publish_stage(root, '04_comparison', output, result)


if __name__ == '__main__':
    main()
