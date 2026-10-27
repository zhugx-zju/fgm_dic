"""只读取00_calibration的冻结曲线和拟合结果并重绘，不重新标定。"""
import argparse
from pathlib import Path
from lib.config_utils import load_python_configuration
from lib.calibration_workflow import plot_calibration_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config')
    parser.add_argument('--input', type=Path, help='已保存的00_calibration目录')
    args = parser.parse_args()
    output = args.input or load_python_configuration(args.config).CALIBRATION_OUTPUT
    plot_calibration_results(output)
    print('Calibration figures:', output)


if __name__ == '__main__':
    main()
