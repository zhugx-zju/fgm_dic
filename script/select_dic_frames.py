"""选取DIC全场导出帧，核对同期载荷及参考力；参数均在config.py。"""
import argparse
from lib.config_utils import load_python_configuration
from lib.frame_selection_workflow import select_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config')
    cfg = load_python_configuration(parser.parse_args().config)
    for record in select_frames(cfg):
        print(f'run{record["run"]}: {record["stage_first"]}–{record["stage_last"]}, '
              f'{record["status"]}, force coverage={record["force_coverage_count"]}/{record["frame_count"]}')
    print('Export requests:', cfg.FRAME_SELECTION_OUTPUT/'export_requests.csv')


if __name__ == '__main__':
    main()
