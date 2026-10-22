# 核心结果比较图集：3张粗网格三载荷正文图

三张图统一采用121×19粗网格、H1等权重、主ν=0.26。180N、280N、380N分别用自身DIC独立选参和识别E，各载荷结果在同一张图中并列比较。

| 图号与PNG文件名 | 比较内容 | 说明 |
| --- | --- | --- |
| 01_三平台二维模量对比.png | 独立标定与三平台各自识别的二维E | 同一粗网格、共同模量颜色范围，展示完整空间结构。 |
| 02_三平台E_x与独立标定.png | 三平台E(x)、独立标定与横向空间离散 | 横向空间标准差不是标定不确定度。 |
| 03_三载荷位移对比.png | 三载荷DIC、独立标定正演、本平台识别模量正演回代 | 三列为180N、280N、380N；上半轴向、下半横向。每个分量在全部载荷和三种来源之间共用色标。 |

输出位于`results/dic_full/independent_platforms_20261022/PPT精选/`，目录直接提供3张PNG；3张矢量图位于`SVG/`。`index.html`为精选预览，`图片说明.md`说明图意。完整汇总页与原图清单均指向当前三图精选。此前五图精选保存在`history/ppt_5figures_before_three_load_merge/`，原始计算与原图均保留。

三载荷可以共同展示，但名义平台与净拉力须区分：180N、280N的配对净拉力分别为119.7761787、219.7354507N；380N采用假定净力319.9N，尚无同步实测力。三组识别均使用本平台DIC，其正演回代不是未使用观测的独立验证。

## 位移图定义与复现

位移图从各自冻结的`01_prepared/fields.npz`、`02_forward/nu_0.26/fields.npz`、`03_inverse/nu_0.26/fields.npz`读取全部23920个测点与已配准预测。没有重解有限元或使用修改后的内核。各平台的DIC、标定正演、识别模量正演同时减去本平台DIC均值，仅改变显示基点；未按载荷、位移幅值归一化，点对点残差不变。

三个轴向图组共用对称范围±0.17285136298mm；三个横向图组共用±0.02816281478mm。行标签统一为`DIC Data`、`Displacement from Calibrated E`、`Displacement from Identified E`。各平台的模量来源均为自身识别。

从项目根目录复现新增联合位移图：

```python
from pathlib import Path
from lib.dic_three_load_figures import plot_three_load_displacements
root = Path.cwd()
plot_three_load_displacements(
    root, root / 'results/dic_full/independent_platforms_20261022/coarse_three_platforms')
```

源图为`coarse_three_platforms/three_load_displacement_comparison.png`及同名SVG，240dpi PNG尺寸4080×2160。`three_load_displacement_metadata.json`保存12个输入文件的SHA256、网格、权重、ν、实际使用力和色标。前两张沿用已有三平台粗网格图。

## 核验与范围

新增图已实际生成并检查渲染，修正了组件组之间的标题重叠。独立数值检查确认三平台场均为有限值，居中前后逐点残差一致至浮点精度，共同色标覆盖全部18幅位移场，12个冻结输入哈希均未变化；SVG含全部平台及来源标签。精选的3张PNG与3张SVG均与源图逐文件哈希一致，10个精选页面本地链接均存在。

`selection.json`记录源路径、精选路径和SHA256；`verification.json`记录文件数和复制及链接核验。此项仅改变图集组织和位移图布局，不改变反演的求解停止状态、材料结论或物理配准与边界近似的验证状态。
