"""Step 3: free 2D DIC inversion and held-out prediction; no calibrated forward fit."""

import argparse
import json
from pathlib import Path

from lib.config_utils import prepared_input
from lib.dic_case_workflow import inverse_dic, plot_saved_stage, resume_l_curve
from lib.dic_case_reporting import publish_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='01_prepared directory; defaults to CASE_OUTPUT in --config')
    parser.add_argument('--config', default='dic_case_config', help='Locates CASE_OUTPUT only; inputs stay frozen')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot-only', action='store_true', help='Redraw saved fields only; no inversion')
    parser.add_argument('--resume', action='store_true', help='Recover unresolved scan line searches within the original remaining budget')
    args = parser.parse_args()
    if args.resume and args.plot_only:
        parser.error('--resume and --plot-only are mutually exclusive')
    source = prepared_input(args.input, args.config)
    output = args.output or source.parent/'03_inverse'
    record_path = output/'run_metadata.json' if args.plot_only else source/'run_metadata.json'
    saved = json.loads(record_path.read_text(encoding='utf-8'))['configuration']
    print(f"Saved solver settings: FTOL={saved['FTOL']:g}, GTOL={saved['GTOL']:g}; "
          f"regularization={saved['REGULARIZATION_MODE']}; gamma selection=L-curve")
    if args.resume:
        result = resume_l_curve(source, output)
    else:
        result = plot_saved_stage(output, source) if args.plot_only else inverse_dic(source, output)
    print('Inverse:', result['status'], output.resolve())
    publish_stage(source.parent, '03_inverse', output, result)
    if (output/'云图.html').exists():
        print('Cloud plots:', (output/'云图.html').resolve())
    if result['status'] != 'computed':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
