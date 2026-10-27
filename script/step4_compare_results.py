"""Step 4: compare saved DIC, forward and inverse results; no recomputation."""

import argparse
from pathlib import Path

from lib.config_utils import prepared_input
from lib.dic_case_workflow import compare_dic_results
from lib.dic_case_reporting import publish_stage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='01_prepared directory; defaults to CASE_OUTPUT in --config')
    parser.add_argument('--config', default='config', help='User configuration; verified against frozen step-1 inputs')
    parser.add_argument('--forward', type=Path)
    parser.add_argument('--inverse', type=Path)
    parser.add_argument('--output', type=Path)
    drawing = parser.add_mutually_exclusive_group()
    drawing.add_argument('--no-plots', action='store_true')
    drawing.add_argument('--plot-only', action='store_true', help='Refresh plots from the saved comparison; no numerical changes')
    args = parser.parse_args(argv)
    source = prepared_input(args.input, args.config)
    root = source.parent
    output = args.output or root/'04_comparison'
    result = compare_dic_results(source, args.forward or root/'02_forward',
                                  args.inverse or root/'03_inverse', output, plots=not args.no_plots,
                                  plot_only=args.plot_only)
    print('Comparison:', result['status'], output.resolve())
    publish_stage(root, '04_comparison', output, result)
    print('Cloud plots:', (output/'打开查看结果.html').resolve())


if __name__ == '__main__':
    main()
