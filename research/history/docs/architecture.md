# 保留的模块与数据流

`script`四个入口调用`lib.dic_case_workflow`，配置读取及冻结身份在`lib.config_utils`，原始数据在`lib.experiment_data`／`lib.io_utils`，绘图与入口在`lib.dic_preprocessing`／`lib.dic_case_reporting`／`lib.research_reporting`。`lib.synthetic_recovery`保留两类受控仿真恢复。

`fgm_asm`保留原共用Q4有限元、材料、观测、载荷、伴随、正则、L-curve与精确TV。所有内核文件保持字节一致。grf与tgv被原实现导入，保留以保护接口，不另设日常运行入口。流程代码仅新增未配对输入的前置检查。

旧的网格加密、扫描恢复、独立TV研究与报告模块不由四步入口依赖，归入archive/legacy_20261026.zip。原始配置、说明和方案一并按原路径保存。恢复前先核对docs/slimming_manifest.json，避免覆盖之后的新文件。

历史参考包只保存代表图、CSV、摘要和核验记录，不含全部优化场、原始DIC与完整迭代日志；完整证据仍在原1016工程，来源及文件SHA256见参考包provenance.json。不可将摘要包当作独立可重算的输入集。
