# 历史研究与辅助验证

当前实验的四个步骤入口位于项目根目录 `script/`，配置是 `dic_case_config.py`；方法及输出见 [当前流程](../../docs/dic_stepwise_workflow.md)，结果从根目录 [打开结果.html](../../打开结果.html) 查看。

本目录保留早期方法及复现入口，历史几何、网格、泊松比和配准假设不作为当前实验配置。

| 目录或文件 | 研究内容 |
| --- | --- |
| `script/run_synthetic.py` | 70 mm 合成真值、自由二维模量识别与留出预测；使用根目录 `synthetic_config.py` |
| `script/run_experiment.py` | 早期实验 inspect/model/forward/inverse；使用根目录 `experiment_config.py` |
| `script/analyze_*`、`check_revised_dic_sensitivity.py`、`report_revised_dic.py` | 早期 DIC 方位、噪声、增量及敏感性分析 |
| `script/benchmark_*`、`compare_regularization.py`、`check_regularization_gradient.py` | 正则方法比较与梯度核验 |
| `script/forward_job.py`、`inverse_*`、`plot_*` | 原 110 mm 算例的求解与后处理 |
| `config.py` | 原 110 mm 合成配置，仍用于算法测试 |
| `dic_revision_config.py`、`dic_rot90_config.py`、`lib/` | 早期 DIC 探索配置、分析及原散点报告实现 |
| `docs/`、`plan/` | 对应历史研究报告和图件 |
| `local_work/`（仅本机） | 原有未提交的额外入口、绘图模块、说明与测试 |
| `local_snapshot/`（仅本机） | 原有未提交工作快照、差异及文件哈希记录 |

从项目根目录以包名运行历史入口，例如：

```powershell
python -m archive.legacy_workflows.script.run_synthetic --output results/synthetic_replication_02
python -m archive.legacy_workflows.script.run_experiment --mode inspect --output results/inspect_replication_02
python -m archive.legacy_workflows.script.analyze_revised_dic --output results/revised_replication_02
```

原始实验、材料标定和已有计算输出继续位于 `post_tensile_test/` 与 `results/`。算法测试保留在根目录 `tests/`。
