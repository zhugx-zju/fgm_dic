"""Step 2: independent calibrated forward problem, using frozen step-1 inputs."""

import argparse
from pathlib import Path

from lib.config_utils import prepared_input
from lib.dic_case_workflow import forward_calibrated, plot_saved_stage
from lib.dic_case_reporting import publish_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='01_prepared directory; defaults to CASE_OUTPUT in --config')
    parser.add_argument('--config', default='dic_case_config', help='User configuration; verified against frozen step-1 inputs')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot-only', action='store_true', help='Redraw saved fields only; no FE solve')
    args = parser.parse_args()
    source = prepared_input(args.input, args.config)
    output = args.output or source.parent/'02_forward'
    result = plot_saved_stage(output, source) if args.plot_only else forward_calibrated(source, output)
    print('Forward:', result['status'], output.resolve())
    publish_stage(source.parent, '02_forward', output, result)
    print('Cloud plots:', (output/'云图.html').resolve())


if __name__ == '__main__':
    main()
