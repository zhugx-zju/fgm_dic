"""只读取00_frame_selection的冻结时序和选帧结果重绘，不重新选帧。"""
import argparse
from pathlib import Path
from lib.config_utils import load_python_configuration
from lib.frame_selection_workflow import plot_frame_selection_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config')
    parser.add_argument('--input', type=Path, help='已保存的00_frame_selection目录')
    args = parser.parse_args()
    output = args.input or load_python_configuration(args.config).FRAME_SELECTION_OUTPUT
    plot_frame_selection_results(output)
    print('Frame selection figures:', output)


if __name__ == '__main__':
    main()
