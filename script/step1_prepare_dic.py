"""Step 1: prepare DIC inputs or synthetic observations selected in the user config."""

import argparse
from pathlib import Path

from lib.config_utils import load_python_configuration
from lib.dic_case_workflow import plot_saved_stage, prepare_dic
from lib.dic_case_reporting import publish_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='dic_case_config', help='Python module or .py path')
    parser.add_argument('--output', type=Path, help='Override 01_prepared directory; later stages then need --input')
    parser.add_argument('--plot-only', action='store_true', help='Redraw saved fields only; do not reread raw CSV')
    args = parser.parse_args()
    cfg = load_python_configuration(args.config)
    print(f"当前选择：data={getattr(cfg, 'DATA_MODE', 'legacy configuration')}, "
          f"mesh={getattr(cfg, 'MESH_SOURCE', 'configured_grid')}, regularization={cfg.REGULARIZATION_MODE}")
    output = args.output or Path(cfg.CASE_OUTPUT)/'01_prepared'
    result = plot_saved_stage(output) if args.plot_only else prepare_dic(cfg, output)
    print('Preparation:', result['status'], output.resolve())
    publish_stage(output.parent, '01_prepared', output, result)
    print('Cloud plots:', (output/'云图.html').resolve())
    if result['status'] == 'needs_registration':
        raise SystemExit(3)


if __name__ == '__main__':
    main()
