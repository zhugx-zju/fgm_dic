"""Step 3: free 2D DIC inversion and held-out prediction; no calibrated forward fit."""

import argparse
from pathlib import Path

from lib.dic_case_workflow import inverse_dic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='01_prepared directory')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input.parent/'03_inverse'
    result = inverse_dic(args.input, output)
    print('Inverse:', result['status'], output.resolve())
    if result['status'] != 'computed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
