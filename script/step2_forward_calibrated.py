"""Step 2: independent calibrated forward problem, using frozen step-1 inputs."""

import argparse
from pathlib import Path

from lib.dic_case_workflow import forward_calibrated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='01_prepared directory')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input.parent/'02_forward'
    result = forward_calibrated(args.input, output)
    print('Forward:', result['status'], output.resolve())


if __name__ == '__main__':
    main()
