# 输入规范

所有路径由项目根目录 `config.py` 的 `PROJECT_ROOT` 推导，不依赖机器盘符或当前工作目录。用户配置、原始输入和生成结果分别存放；数值模块不猜测“最新实验”。新建私人输入放 `data/local/` 或配置指定的外部目录，自带 `data/example/` 全部为合成数据。

## 材料标定

`CALIBRATION_MANIFEST` 为UTF-8 CSV：

```csv
specimen_id,infill_percent,width_mm,thickness_mm,gauge_mm,file,block
40-1,40,10.02,3.18,50,40/40_1.xlsb,1
40-2,40,10.01,3.20,50,40/40_2.csv,1
```

`file` 相对清单所在目录解析，也支持绝对路径。尺寸取每根实测值；`block` 明确指定xlsb中的1基数据块，CSV只有1块。归一化曲线CSV字段为 `time_s,force_N,displacement_mm,extension_mm`；xlsb沿用“时间、力、位移、变形”四列一组。缺失或异常输入应修正，不静默选择另一条曲线。

同一 `CALIBRATION_SETTINGS` 对所有试样生效。默认逐根峰前5%–80%、步长2.5%、最小窗口跨度10%、至少60点、R²≥0.995、半段斜率相对差≤5%、应变非递减步比例≥90%。每根起始 `CALIBRATION_BASELINE_POINTS` 点的力中位数仅作常数修正。随后联合选择每根窗口，使各根相对组均值偏差≤3%、组均值随填充率不减，并最小化Σ(1−R²)。

**这两个性质是显式选段先验。** 输出同一候选池下的无约束逐根最高R²结果；不将约束后的离散作为重复试样天然一致性的证据。选择 `individual_highest_R2` 可关闭联合约束。标定来源只使用本次1023试样，不自动比较其他批次。标定值仅用于标定正演、事后比较；不进入反演初值、目标或γ选择。

若复用已有标定，在 `config.py` 将 `CALIBRATION_CSV` 指向已有 `group_summary.csv`，并匹配 `CALIBRATION_METHOD`，可跳过标定入口。该表要求 `infill_percent,batch,n,mean_E_MPa,sd_E_MPa,fit_method`；每个选用填充率恰有一行。

## DIC选帧与力配对

每次采集分别配置一个 `TIMING_RUNS` 条目：

- 试验机文件：xlsb（时间、力、位移、变形）或CSV `time_s,force_N,displacement_mm,extension_mm`。
- DIC两点文件：原软件两段 `-DY` 导出，或CSV `stage,point1_DY_mm,point2_DY_mm`。必须确认0基、连续采集和真实fps，不能把行号直接当秒。
- `events`：本次中间停留的进入/退出事件，各给出试验机与DIC的拟合区间和转折搜索区间。局部两直线允许DIC跳变；由两点及均值的全部事件偏移取中位数。不得继承另一试样的时差。经硬件同步时可显式选择 `synchronization='hardware'` 并附时间偏移与来源说明。
- `tail_interval/tail_search`：本次最大加载后的DIC停留入口拟合范围；位移信号须按加载为正准备。默认入口后6秒起取15帧，要求末帧不晚于入口后9秒，并检查测点/区间扰动下至少4秒的稳定等待。
- `reference_force_N/reference_note`：与DIC参考图对应的参考力及依据。未核实则留空，不能自动拿首条示值或零力替代。

输出 `export_requests.csv` 明确列出应导出的文件名。`selected_dic_windows.csv` 同时记录连续窗口、估计时差、时差敏感性范围、载荷覆盖、窗口力变化和参考力。仅在所有事件偏移范围内均有同期力、记录间隔合格、窗口力足够稳定且接近最大载荷、参考力有说明时，输出 `ready` 和净力。其他情况保留为 `export_candidate_only`，不外推或填充力。

事件分散范围是方法敏感性，不是硬件同步精度；两点平均位移含刚体运动，不能当作应变或全场稳定证明。导出后仍须检查全场相关质量、共同有效点和时域变化。

`USE_FRAME_SELECTION=True` 时，第1步读取本次选帧表；未完成配对会拒绝进入计算。若用户明确接受假定载荷，可设置 `False`，在 `PLATFORMS` 填写 `stages,force_N,force_note`，并设 `FORCE_ASSUMED=True`。假设会进入冻结元数据，不能称为已核实载荷。

## 全场CSV、坐标和几何

推荐目录 `data/<case>/dic/run1/stage_00130.comdat_comdata.csv`。窗口包含首尾帧，所有文件必须齐全。各次分别保留 `stage_00000` 原始参考图；该图不是停留窗口的一部分。默认读11列2D导出（含Type、X、Y、DX、DY、位移幅值、三应变分量、grid_row、grid_col），兼容15列3D导出；以记录列数判断格式，不能调换列序。

`DIC_COORDINATE_STATE='deformed'` 时用 `[X-DX,Y-DY]` 恢复参考坐标；`reference` 时直接用 `[X,Y]`。坐标与位移均为mm，以共同grid ID求逐帧均值；保留时间标准差，不假设各帧独立而除√n。冲突重复点、参考坐标变化、缺失帧与域外点均有显式检查。

承载宽度与DIC ROI宽度分别配置。牵引为 `净力/(真实承载宽×厚度)`，ROI端力按宽度换算；内部截取侧边的自由边界只是近似。`REGISTRATION` 只允许保长旋转/反射和平移，禁止将点云跨度缩放到制造尺寸。分区位置只服务标定正演和比较，不限制自由二维模量参数。

## 现有1026输入

`CASE_PROFILE='experiment'` 提供sample_2全段三次采集预设；`SAMPLE_RUN` 选1/2/3，厚度约3.2mm，按用户已接受的名义末载减记录首载假设运行。原三组窗口591–605、625–639、689–703均缺同期平台力；选帧入口会如实输出候选状态，不把这些假设改称已验证配对。预设不继承40%–70%裁剪或缩小γ上限研究。

均匀60%试样的历史窗口945–959、465–479、235–249分别属于不同参考图；第1/2次同样缺平台同期力，第3次有载荷覆盖但参考图与起载对应仍未核实。其既有报告保留原处，不自动套用为sample_2的参数。
