# 目录与主次

沿用 ASM-Python 的组织方式：根目录 `config.py` 管理运行参数，`script/` 提供独立入口，`fgm_asm/` 提供共用求解内核。实验输入输出增加明确规范，避免通过最新目录或历史结果推断输入。

```text
config.py                   唯一日常配置
script/
  calibrate_material.py     独立材料标定
  select_dic_frames.py      DIC时序、选帧、同期力与参考力核对
  step1_prepare_dic.py      冻结观测、载荷、坐标、权重与配置
  step2_forward_calibrated.py
  step3_inverse_dic.py      L-curve扫描与最终反演
  step4_compare_results.py  读取保存场，比较并生成结果页
  plot_*.py                 六个逐步结果入口，只重绘冻结数据
fgm_asm/                    原Q4有限元、观测、伴随及正则内核
lib/                        数据读取、标定、选帧、四步编排与结果保存
data/example/               可再分发的合成输入及来源说明
data/local/                 私人输入清单，Git忽略
results/<RUN_NAME>/         每次运行的完整证据，Git忽略
research/                   可选专题、旧配置与旧文档
tests/                      主流程关键输入规则的少量检查
tools/                      示例重建与纯源码导出；不属于日常计算步骤
docs/public/                当前使用说明
```

| 内容 | 定位 | 日常运行方式 |
|---|---|---|
| 自由二维H1 | 默认主方法，所有y行模量独立 | `REGULARIZATION_MODE='gradient'` |
| Huber-TV、精确TV | 用户确认保留的主流程可选方法 | 在同一配置切换，并重新设γ范围及求解预算 |
| 独立材料标定、标定正演、模量与位移比较、回代 | 主线证据链 | 六个入口分别执行 |
| 一维/E(x)、已知界面、顺序多工况、联合参考诊断 | 辅助研究 | `research/`，不从主线导入 |
| 裁剪、γ敏感性、均匀性专题、多套绘图报告 | 辅助研究 | `research/`，不改变默认实验范围或γ |
| 合成恢复 | 软件核验 | 不替代物理实验验证 |

每个计算入口都有单独的结果脚本，具体对应关系见运行文档。计算时也会生成初版图；之后可单独运行结果脚本反复查看，不需要再次计算。

1026的不同记录使用独立参考图，默认逐次计算。顺序传递初值的最终场只优化末工况目标，不能作为同时联合反演的标准解路径。

原 `post_tensile_test/`、`3D_print/`、历史数值结果和用户原有修改保留原路径。新的输入可以遵循 `data/<case>/` 规范；本地实验预设显式引用原目录。旧专题模块迁至 `research/` 后，命令前缀由 `python -m lib.<专题>` 改为 `python -m research.<专题>`。旧结果中的原命令作为历史元数据保留。

原根配置逐字保留在 `research/previous_dic_case_config.py.txt`，不会被主流程导入。旧工程文档在 `research/history/docs/`；其“当前配置”描述属于历史。

源码导出使用白名单，只包含日常运行代码、合成示例、当前文档和必要检查；不包含 `research/`、原始实验、已有结果、打印文件或Git历史。当前实验仓库与其历史仍包含旧资料，公开分享使用该导出包。
