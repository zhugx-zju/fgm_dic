"""Step 4: compare saved DIC, forward and inverse results; no recomputation."""

import argparse
from pathlib import Path

from lib.dic_case_workflow import compare_dic_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='01_prepared directory')
    parser.add_argument('--forward', type=Path)
    parser.add_argument('--inverse', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    root = args.input.parent
    output = args.output or root/'04_comparison'
    result = compare_dic_results(args.input, args.forward or root/'02_forward',
                                  args.inverse or root/'03_inverse', output, plots=not args.no_plots)
    print('Comparison:', result['status'], output.resolve())


if __name__ == '__main__':
    main()
