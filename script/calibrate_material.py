"""独立材料标定：修改根目录 config.py 后执行本入口。"""
import argparse
from lib.config_utils import load_python_configuration
from lib.calibration_workflow import calibrate_material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config')
    cfg = load_python_configuration(parser.parse_args().config)
    print(calibrate_material(cfg).to_string(index=False))
    print('Calibration:', cfg.CALIBRATION_OUTPUT)


if __name__ == '__main__':
    main()
