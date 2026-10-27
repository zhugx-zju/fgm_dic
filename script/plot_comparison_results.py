"""重绘已保存的结果；不重新执行数值计算。"""
import sys
from script.step4_compare_results import main


if __name__ == "__main__":
    main(["--plot-only", *sys.argv[1:]])
