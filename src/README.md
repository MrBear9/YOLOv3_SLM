# `src` 脚本使用说明

本目录包含当前项目的教师网络/光学学生网络评估、硬件相位导出、单图推理与论文绘图脚本。

以下命令默认在项目根目录 `E:\pythonProject\YOLOv3_SLM` 中执行，并先进入项目使用的 Conda 环境：

```powershell
conda activate deeplearn
```

PowerShell 多行命令使用反引号 `` ` `` 续行；Linux 服务器可改用反斜杠 `\`，也可以将命令写在一行。

## 快速选择

| 需求 | 脚本 |
|---|---|
| 将处理后的 CCD 光学特征图直接送入 Light 检测头 | `detect_ccd_light_dataset.py` |
| 评估光学学生网络 + Light 检测头 | `evaluate_slm_light_dataset.py` |
| 评估 Teacher V2 + Light 检测头 | `evaluate_teacher_v2_dataset.py` |
| 评估旧版 compact 教师模型 | `evaluate_teacher_compact.py` |
| 对单张图像预测 Teacher V2 动态相位并检测 | `predict_teacher_v2_single_image.py` |
| 批量导出 Teacher V2 硬件相位、输入和检测结果 | `export_teacher_v2_dataset_hardware.py` |
| 批量导出静态 SLM + Light 的硬件输入、固定相位和检测结果 | `export_slm_light_dataset_hardware.py` |
| 导出光学学生网络的静态 SLM 相位 | `export_slm_phase.py` |
| 将已导出的 SLM 相位绘制成论文图 | `visualize_slm_phase.py` |
| 从 TensorBoard 日志导出真实 PR/AP 图 | `export_tensorboard_pr_ap.py` |
| 从评估报告生成四联性能图 | `plot_evaluation_report_figure.py` |
| 生成不同数据集/模型的 mAP 对比柱状图 | `plot_validation_performance_bars.py` |

## 评估阈值说明

数据集评估脚本中的三个阈值用途不同：

- `--conf-threshold`：设置固定工作点，用于计数混淆矩阵、TP、FP、FN 和审查信息。它通常不会直接提高由整条置信度排序曲线计算的 mAP。
- `--nms-threshold`：非极大值抑制的 IoU 阈值，决定重叠预测框的保留方式，可能影响最终 mAP。
- `--iou-threshold`：在 `evaluate_teacher_v2_dataset.py` 和 `evaluate_slm_light_dataset.py` 中仅控制混淆矩阵和审查清单的匹配阈值。AP50 固定为 IoU=0.5，多阈值 AP 固定为 0.50:0.05:0.95，不随此参数改变。

若不传这些参数，脚本使用对应训练配置中的默认值。为了公平比较不同 checkpoint，建议固定数据集、split 和三个阈值。

### 离线 mAP@0.5:0.95

### 官方 COCO 评估（默认开启）

教师 V2 和 SLM Light 两个离线脚本现在默认调用官方 `pycocotools.COCOeval`。在运行评估的环境中安装：

```powershell
python -m pip install pycocotools
```

随后运行原来的评估命令即可。缺少依赖会在模型运行前报错；若只需要旧指标，可显式传入 `--skip-coco`。无需修改训练或重新训练。

官方结果位于 `evaluation_report.json` 的 `metrics.coco`，同时输出独立的 `coco_evaluation_report.json`。主要字段：`AP`（官方 mAP50:95）、`AP50`、`AP75`、`AP_small`、`AP_medium`、`AP_large`、`AR1`、`AR10`、`AR100`、分尺寸 AR 和 `per_class`。某分组无有效真实目标时报告为 `null`，对应官方控制台的 -1。

输出目录还包含 `coco_annotations.json`、`coco_predictions.json`，可直接交由 pycocotools 复核。框按数据集真实 letterbox 的缩放与填充逆变换回原图像素坐标，面积按原图框宽×高计算；预测框不额外裁剪。官方阈值、101 点采样、面积分组与 `maxDets=[1,10,100]` 使用原生默认值。

当前标签源是 YOLO 普通框，导出均设 `iscrowd=0`，不能恢复源数据未提供的 crowd 或 ignore 标注。若数据包含需要忽略的人群区域，应先补全原始标注。解码仍采用项目的置信度阈值、NMS 和最大候选数量，具体设置记录在 `protocol` 中；官方评估器不负责重新生成被提前过滤的框。`--iou-threshold` 只影响混淆矩阵与审查匹配，不改变官方 AP 阈值。

以下 `metrics.map50_95` 等字段继续保留为历史自定义指标；论文使用官方协议结果时，应读取 `metrics.coco.AP`，不要混用两个字段。此次不扩展旧 compact 入口。

### 保留的自定义多阈值指标

`evaluate_teacher_v2_dataset.py` 和 `evaluate_slm_light_dataset.py` 默认同时输出 mAP50、mAP50:95 和 mAP75，不需要增加命令行开关，也不需要重新训练。多阈值计算复用已收集的预测框，不对每个 IoU 阈值重复运行模型。旧 compact 评估入口本次未扩展。

结果写入 `--output` 目录中的 `evaluation_report.json`：

| JSON 字段（位于 `metrics`） | 含义 |
| --- | --- |
| `map50` | 保留原有全点 PR 面积积分的 AP50，方便比较历史记录 |
| `map50_95` | 10 个 IoU 阈值 0.50、0.55、…、0.95 下的 101 点插值 AP 均值 |
| `map50_101` | 与新多阈值指标同积分口径的 AP50 |
| `map75` | 101 点插值 AP75 |
| `map_by_iou` | 各 IoU 阈值的宏平均 AP |
| `per_class.<类别>.ap50_95` | 对应类别的多阈值 AP；没有真实目标的类别为 null |
| `multi_iou_protocol` | 阈值、积分方式、解码设置和有效类别数 |

新指标按图像、类别和置信度排序，在每个 IoU 阈值独立完成一对一匹配。没有真实目标的类别不参与宏平均。采用当前解码的置信度、NMS 和最大检测数设置，统计全部目标面积，不处理 COCO crowd/ignore 标注，因此不是完整 COCO 官方评估协议；与其他论文比较时需对齐这些设置。旧 `map50` 与 `map50_101` 因积分方式不同可能有小幅差异。

示例（在项目根目录、具备项目依赖的 Python 环境中运行；检查点必须与当前光学配置和网络结构匹配）：

```powershell
python src/evaluate_slm_light_dataset.py --checkpoint output/SLM_Tv2_dmd640_contextdw_d20d10_p13_v2/detector_best.pth --data data/military/data.yaml --split val --output output/slm_eval_multi_iou
python src/evaluate_teacher_v2_dataset.py --checkpoint output/Tv2_dmd640_contextdw_d20d10_p13_v2/teacher_detector_best.pth --data data/military/data.yaml --split val --output output/teacher_eval_multi_iou
```

训练日志与最佳检查点选择仍使用原有 mAP50，本次未增加训练期开销。若后续需要按 mAP50:95 选模型，应对多个候选检查点在同一验证集离线重评，或者单独加入周期性训练验证；只评估 mAP50 最佳检查点，不保证找到 mAP50:95 最佳检查点。测试集不用于选择模型。

## `detect_ccd_light_dataset.py`

### 作用

批量读取 `tools/capture_ccd_video_frames.py` 已完成 ROI 和透视校正的 CCD 光学特征图，只加载 checkpoint 中的 Light 检测头进行识别。该脚本不运行 Teacher 或 SLM 仿真，也不会再次执行场景图 letterbox 或逐图 min-max 归一化。

默认读取：

```text
output/Tv2_dmd640_scratch/hardware_export_100/ccd
```

并自动创建同级输出目录：

```text
output/Tv2_dmd640_scratch/hardware_export_100/ccd_detection
```

### 用法

当前实验可直接运行：

```powershell
python src/detect_ccd_light_dataset.py --device cuda
```

完整写法：

```powershell
python src/detect_ccd_light_dataset.py `
  --images output/Tv2_dmd640_scratch/hardware_export_100/ccd `
  --checkpoint output/Tv2_dmd640_scratch/teacher_detector_best.pth `
  --data data/military/data.yaml `
  --output output/Tv2_dmd640_scratch/hardware_export_100/ccd_detection `
  --device cuda `
  --batch-size 4
```

只处理前两张进行测试：

```powershell
python src/detect_ccd_light_dataset.py --device cuda --limit 2
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--images` | 否 | `output/Tv2_dmd640_scratch/hardware_export_100/ccd` | 处理后的 CCD 光学特征图目录。 |
| `--checkpoint` | 否 | `output/Tv2_dmd640_scratch/teacher_detector_best.pth` | 含 `detector_state_dict` 的 Light checkpoint。教师或学生联合 checkpoint 均可，但检测头结构和类别必须匹配。 |
| `--data` | 否 | `data/military/data.yaml` | 用于恢复类别数和类别名称的数据集 YAML。 |
| `--output` | 否 | `<images同级>/ccd_detection` | 检测结果目录。 |
| `--device` | 否 | 项目配置值 | `cuda`、`cuda:0` 或 `cpu`。 |
| `--batch-size` | 否 | `4` | 推理批大小。显存不足时减小。 |
| `--conf-threshold` | 否 | 配置中的 `CONF_THRESH` | 输出预测框的置信度阈值。 |
| `--nms-threshold` | 否 | 配置中的 `NMS_THRESH` | NMS IoU 阈值。 |
| `--max-det` | 否 | 配置中的 `MAX_DET` | 每张图最多保留的预测框数。 |
| `--limit` | 否 | 全部 | 只处理排序后的前 N 张。 |

### 主要输出

- `0001.png`、`0002.png`……：在 CCD 强度图上绘制类别、置信度和预测框；即使没有预测框也会保存并标注检测数为 0。
- `detection_results.json`：记录 checkpoint、预处理、阈值以及每张图的类别、置信度和框坐标。

CCD 图若不是当前配置的 `640×640`，脚本会直接双线性缩放到检测头尺寸，并在 JSON 中将 `resized_for_detector` 标为 `true`。这里不使用 letterbox，因为 CCD ROI 已代表完整的光学特征平面。

## 1. `evaluate_slm_light_dataset.py`

### 作用

在带标签的 `val` 或 `test` 数据集上评估“光学学生网络 + Light 检测头”。脚本会计算 mAP50、分类别指标和混淆矩阵，并生成可直接传给 `dataset/review_delete_images.py --list` 的审查清单。

还可以保存前若干个样本的光场可视化，包括：

- DMD 输入强度；
- 最终原始光学强度 `|U|^2`；
- 复光场 Real/Imag 的二维联合映射；
- 一份公共的映射图例。

光场结果只保存 PNG 和 JSON，不保存 NPY。

### 用法

```powershell
python src/evaluate_slm_light_dataset.py `
  --checkpoint output/SLM_Tv2_dmd640_scratch/detector_best.pth `
  --data data/military/data.yaml `
  --output output/SLM_Tv2_dmd640_scratch/eval_results_val `
  --device cuda `
  --batch-size 2 `
  --split val `
  --optical-field-samples 2
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--checkpoint` | 是 | 无 | 同时含 `student_state_dict` 和 `detector_state_dict` 的配对 checkpoint。 |
| `--data` | 否 | `data/military/data.yaml` | 数据集 YAML。 |
| `--split` | 否 | `test` | 评估划分，只能为 `val` 或 `test`。 |
| `--output` | 否 | `output/slm_light_dataset_eval` | 结果目录。 |
| `--batch-size` | 否 | 学生配置值 | 批大小覆盖值。显存不足时调小。 |
| `--device` | 否 | 学生配置值 | `cuda`、`cuda:0` 或 `cpu`。 |
| `--conf-threshold` | 否 | 配置值 | 计数混淆矩阵使用的置信度阈值。 |
| `--nms-threshold` | 否 | 配置值 | 解码预测时的 NMS IoU 阈值。 |
| `--iou-threshold` | 否 | 配置值 | AP 和混淆矩阵的框匹配 IoU 阈值。 |
| `--optical-field-samples` | 否 | `0` | 保存前 N 张图的光场；`0` 表示关闭。 |

### 主要输出

- `evaluation_report.json`
- `confusion_matrix_counts_with_background.png/.csv`
- `confusion_matrix_normalized_foreground_percent.png/.csv`
- `review_unrecognized_images.txt`
- `review_map_error_images.txt`
- `review_candidates.jsonl`
- `optical_fields/`（仅当 `--optical-field-samples` 大于 0）

一般只想检查完全未识别的明显异常图时，优先使用：

```powershell
python dataset/review_delete_images.py `
  --images data/military/val/images `
  --list output/SLM_Tv2_dmd640_scratch/eval_results_val/review_unrecognized_images.txt
```

`review_map_error_images.txt` 会包含几乎所有存在任意定位、类别或置信度误差的图，因此通常数量很大，更适合全面误差分析，不适合直接作为删除候选。

## 2. `evaluate_teacher_v2_dataset.py`

### 作用

在带标签数据集上评估“Teacher V2 + Light 检测头”，输出格式与学生网络评估脚本保持一致，审查清单同样可以传给 `dataset/review_delete_images.py`。

### 用法

```powershell
python src/evaluate_teacher_v2_dataset.py `
  --checkpoint output/Tv2_dmd640_scratch/teacher_detector_best.pth `
  --data data/military/data.yaml `
  --output output/Tv2_dmd640_scratch/eval_results_val `
  --device cuda `
  --batch-size 2 `
  --split val
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--checkpoint` | 是 | 无 | 同时含 `teacher_state_dict` 和 `detector_state_dict` 的 checkpoint。 |
| `--data` | 否 | `data/military/data.yaml` | 数据集 YAML。 |
| `--split` | 否 | `test` | 评估划分，只能为 `val` 或 `test`。 |
| `--output` | 否 | `output/teacher_v2_dataset_eval` | 结果目录。 |
| `--batch-size` | 否 | 训练配置值 | 批大小覆盖值。 |
| `--device` | 否 | 训练配置值 | `cuda`、`cuda:0` 或 `cpu`。 |
| `--conf-threshold` | 否 | 配置值 | 混淆矩阵和固定工作点统计的置信度阈值。 |
| `--nms-threshold` | 否 | 配置值 | 所有指标解码使用的 NMS IoU 阈值。 |
| `--iou-threshold` | 否 | 配置值 | AP 和混淆矩阵的框匹配 IoU 阈值。 |

### 主要输出

与 `evaluate_slm_light_dataset.py` 的评估结果相同，但不生成 `optical_fields/`。

## 3. `evaluate_teacher_compact.py`

### 作用

评估教师网络与 compact 检测头的 checkpoint。该脚本从 `models.teacher_train_compact.Config` 读取数据集、设备及其他设置，只提供少量命令行覆盖参数，适合复现对应的 compact 配置。

### 用法

```powershell
python src/evaluate_teacher_compact.py `
  --checkpoint output/Tv1_compactv2/teacher_detector_best.pth `
  --batch-size 2 `
  --conf-threshold 0.35 `
  --output output/Tv1_compactv2/confusion_matrix_eval.png
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--checkpoint` | 否 | `output/Tv1_compactv2/teacher_detector_best.pth` | 含教师和检测头状态的 checkpoint。 |
| `--batch-size` | 否 | 配置值 | 验证批大小。 |
| `--conf-threshold` | 否 | 配置值 | precision、recall 和混淆矩阵的固定工作点。 |
| `--output` | 否 | `output/Tv1_compactv2/confusion_matrix_eval.png` | 混淆矩阵 PNG 路径。 |

该脚本没有 `--data`、`--split` 和 `--device`；如需自由选择这些项目，应使用前两个数据集评估脚本。

## 4. `predict_teacher_v2_single_image.py`

### 作用

对单张输入图像执行 Teacher V2 动态相位预测和 Light 检测，并导出可加载到硬件 SLM 的灰度相位图。Teacher V2 相位依赖当前输入图像，不能把某一张图生成的相位当作全数据集通用静态相位。

可通过 `--captured-feature` 传入相机采集并完成 ROI/透视校正的光学特征图，直接交给 Light 检测头测试真实硬件链路。

### 用法

```powershell
python src/predict_teacher_v2_single_image.py `
  --image data/military/val/images/example.jpg `
  --checkpoint output/Tv2_dmd640_scratch/teacher_detector_best.pth `
  --output output/Tv2_dmd640_scratch/single_example `
  --device cuda
```

带 CCD 特征图和标定 LUT：

```powershell
python src/predict_teacher_v2_single_image.py `
  --image data/military/val/images/example.jpg `
  --captured-feature output/Tv2_dmd640_scratch/hardware_export_100/ccd/0001.png `
  --checkpoint output/Tv2_dmd640_scratch/teacher_detector_best.pth `
  --output output/Tv2_dmd640_scratch/single_hardware_0001 `
  --device cuda `
  --lut calibration/slm_lut.npy
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--image` | 是 | 无 | 输入场景图像，即送入 DMD/SLM 系统的图像。 |
| `--checkpoint` | 是 | 无 | Teacher V2 + Light checkpoint。 |
| `--output` | 否 | `output/single_teacher_v2` | 输出目录。 |
| `--device` | 否 | 训练配置值 | `cuda`、`cuda:0` 或 `cpu`。 |
| `--conf-threshold` | 否 | 配置值 | 检测置信度阈值。 |
| `--nms-threshold` | 否 | 配置值 | NMS IoU 阈值。 |
| `--captured-feature` | 否 | 无 | 相位加载后由相机采集的光学特征图。脚本会转换为灰度并缩放到检测头输入尺寸。 |
| `--lut` | 否 | 无 | 灰度到相位的标定 LUT，支持 `.npy`、`.csv`、`.txt`。 |
| `--gray-inverted` | 否 | 关闭 | LUT 转换后反转 SLM 灰度驱动。 |
| `--phase-levels` | 否 | `256` | SLM 灰度级数，范围 2–256。 |
| `--export-phase-offset-rad` | 否 | `pi` | 相位包裹到 `[0, 2pi)` 前添加的硬件相位偏移，单位 rad。 |

### 主要输出

- `input_letterboxed.png`
- 每层居中相位、硬件相位和 SLM 灰度图
- `detection_simulation.png/.json`
- `detection_hardware.png/.json`（传入 `--captured-feature` 时）
- `export_metadata.json`

## 5. `export_teacher_v2_dataset_hardware.py`

### 作用

按数据集标签顺序批量运行 Teacher V2，为每张图生成输入相关的 SLM 相位、教师特征和检测结果，用于 DMD/SLM/CCD 硬件实验。所有目录使用相同的四位编号 `0001`、`0002`……，便于视频帧与输入图一一对应。

检测可视化中包含真值框和预测框：真值框为绿色，预测框为红色。

### 用法

```powershell
python src/export_teacher_v2_dataset_hardware.py `
  --checkpoint output/Tv2_dmd640_scratch/teacher_detector_best.pth `
  --data data/military/data.yaml `
  --split val `
  --output output/Tv2_dmd640_scratch/hardware_export_100 `
  --device cuda `
  --limit 100
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--checkpoint` | 是 | 无 | Teacher V2 + Light checkpoint。 |
| `--data` | 否 | `data/military/data.yaml` | 数据集 YAML。 |
| `--split` | 否 | `test` | 导出 `val` 或 `test`。 |
| `--output` | 是 | 无 | 编号硬件文件的根目录。 |
| `--device` | 否 | `cpu` | `cuda`、`cuda:0` 或 `cpu`。批量导出建议服务器使用 CUDA。 |
| `--conf-threshold` | 否 | 配置值 | 检测置信度阈值。 |
| `--nms-threshold` | 否 | 配置值 | NMS IoU 阈值。 |
| `--limit` | 否 | 全部 | 只导出前 N 张，适合先做硬件小批量试验。 |
| `--lut` | 否 | 无 | 灰度到相位标定 LUT，支持 `.npy`、`.csv`、`.txt`。 |
| `--gray-inverted` | 否 | 关闭 | LUT 转换后反转 SLM 灰度驱动。 |
| `--phase-levels` | 否 | `256` | SLM 灰度级数，范围 2–256。 |
| `--export-phase-offset-rad` | 否 | `pi` | 硬件相位偏移，单位 rad。 |

### 主要输出

- `input/`：编号后的 DMD 输入图；
- `teacher_feature/`：仿真教师特征；
- `slm1/`、`slm2/` 等：各层硬件灰度相位图；
- `detection/`：带 GT 与预测框的检测图；
- `json/`：逐图预测信息；
- `manifest.json`：编号、原始图像路径与导出信息的对应关系。

SLM 相位图尺寸由当前光学硬件配置决定，不等同于 DMD 的 `640×640` 输入尺寸；当前配置可能分别导出如 `432×432`、`768×768` 的有效调制区域。

## 6. `export_slm_light_dataset_hardware.py`

### 作用

从配对的光学学生网络 + Light checkpoint 导出真实硬件实验所需的完整数据。与 Teacher V2 的逐图动态相位不同，光学学生网络的相位对整个数据集固定，因此只生成：

- `slm1/phase.png`：SLM1 唯一的静态相位图；
- `slm2/phase.png`：SLM2 唯一的静态相位图；
- `input/0001.png`、`0002.png`……：依次送入 DMD 的 640×640 灰度输入。

两个 SLM 文件都命名为 `phase.png`，因此可将两个目录直接传入 `tools/play_dual_slms.py`。保持两张相位图常亮，只需让 DMD 输入序列逐张切换。

### 用法

```powershell
python src/export_slm_light_dataset_hardware.py `
  --checkpoint output/SLM_Tv2_dmd640_scratch/detector_best.pth `
  --data data/military/data.yaml `
  --split test `
  --output output/SLM_Tv2_dmd640_scratch/hardware_export_100 `
  --device cuda `
  --batch-size 2 `
  --limit 100
```

随后加载两张固定相位图：

```powershell
python tools/play_dual_slms.py `
  --zkwx-input output/SLM_Tv2_dmd640_scratch/hardware_export_100/slm1 `
  --magicholo-input output/SLM_Tv2_dmd640_scratch/hardware_export_100/slm2
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--checkpoint` | 是 | 无 | 同时包含 `student_state_dict` 和 `detector_state_dict` 的单头 SLM + Light checkpoint。 |
| `--data` | 否 | `data/military/data.yaml` | 数据集 YAML，用于图像、标签、类别数和类别名称。 |
| `--split` | 否 | `test` | 导出的数据划分，只能为 `val` 或 `test`。 |
| `--output` | 是 | 无 | 硬件导出根目录。 |
| `--device` | 否 | 学生配置值 | `cuda`、`cuda:0` 或 `cpu`。 |
| `--batch-size` | 否 | `2` | 生成仿真光学强度与检测预览时的批大小。 |
| `--conf-threshold` | 否 | 配置值 | 仿真检测图使用的置信度阈值。 |
| `--nms-threshold` | 否 | 配置值 | 仿真检测图使用的 NMS IoU 阈值。 |
| `--limit` | 否 | 全部 | 只导出排序后的前 N 张。 |
| `--lut` | 否 | 当前配置 | 可选的共享灰度到相位 LUT。 |
| `--gray-inverted` | 否 | 当前配置 | 强制反转两块 SLM 的灰度驱动。 |
| `--no-gray-inverted` | 否 | 当前配置 | 强制不反转两块 SLM 的灰度驱动。 |
| `--phase-levels` | 否 | 当前配置 | 硬件相位灰度级数，范围 2–256。当前两块 SLM 使用 8 位即 256 级。 |
| `--export-phase-offset-rad` | 否 | 当前配置 | 硬件导出前添加的相位偏移，之后环绕到 `[0, 2pi)`。 |

### 主要输出

- `input/`：编号后的真实 DMD 灰度输入；
- `slm1/phase.png`、`slm2/phase.png`：分别为 `432×432`、`768×768` 的当前硬件有效区域静态相位图；
- `raw_optical_intensity/`：仿真的原始 `|U|²` 显示预览；
- `detector_feature/`：归一化/极性处理后真正送入 Light 的特征预览；
- `detection/`：绿色 GT 框与红色预测框；
- `json/`：逐图预测和真值；
- `manifest.json`：checkpoint、光学参数、固定相位和所有编号对应关系。

`raw_optical_intensity/` 与 `detector_feature/` PNG 为方便观察而进行对比度增强，不是相机标定所需的定量数组。脚本会拒绝多头 SLM checkpoint，因为一块实体 SLM 无法同时显示同一层的多套相位。

## 7. `export_slm_phase.py`

### 作用

从光学学生网络 checkpoint 中导出训练得到的静态 SLM 相位。它与 Teacher V2 的逐图动态相位不同：学生网络的相位参数对所有输入共用。

### 用法

```powershell
python src/export_slm_phase.py `
  --checkpoint output/SLM_Tv2_dmd640_scratch/detector_best.pth `
  --output output/SLM_Tv2_dmd640_scratch/slm_phase_export
```

使用 LUT：

```powershell
python src/export_slm_phase.py `
  --checkpoint output/SLM_Tv2_dmd640_scratch/detector_best.pth `
  --output output/SLM_Tv2_dmd640_scratch/slm_phase_export `
  --lut calibration/slm_lut.npy `
  --inverted
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--checkpoint` | 是 | 无 | 光学学生网络 checkpoint。 |
| `--output` | 是 | 无 | 灰度 PNG、相位数组和元数据的输出目录。 |
| `--lut` | 否 | 无 | 灰度到相位 LUT，支持 `.npy`、`.csv`、`.txt`。 |
| `--inverted` | 否 | 关闭 | 强制使用反转的 SLM 灰度驱动；还需结合当前配置判断实际极性。 |

### 主要输出

每层会导出原始/仿真/硬件相位数组以及 `slm1.png`、`slm2.png` 等硬件灰度图，并生成 `phase_export_metadata.json`。这些 NPY 是相位部署与复现实验需要的数值文件，不属于评估脚本的 `optical_fields/` 可视化输出。

## 8. `visualize_slm_phase.py`

### 作用

读取 `export_slm_phase.py` 生成的 SLM 灰度 PNG，并生成适合论文使用的相位伪彩图、中心横纵剖面以及相位分布直方图。

### 用法

```powershell
python src/visualize_slm_phase.py `
  --input output/SLM_Tv2_dmd640_scratch/slm_phase_export `
  --output output/SLM_Tv2_dmd640_scratch/optical_student_slm_vis
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--input` | 是 | 无 | `export_slm_phase.py` 生成的单个 PNG 或目录。目录模式会查找文件名中含 `phase` 或 `slm` 的 PNG。 |
| `--output` | 是 | 无 | 论文图输出目录。 |
| `--lut` | 否 | 无 | 导出阶段使用的灰度到相位 LUT。 |
| `--gray-inverted` | 否 | 关闭 | 强制按反转灰度解释相位。若目录中有元数据，脚本也可继承 LUT/反转设置。 |
| `--profile-index` | 否 | 图像中心 | 横纵相位剖面的行/列索引。 |
| `--hist-bins` | 否 | `64` | 相位分布直方图的 bin 数，最小为 2。 |

### 主要输出

每个输入相位图对应：

- `*_phase_colormap.png`
- `*_phase_profiles.png`
- `*_phase_distribution.png`

## 9. `export_tensorboard_pr_ap.py`

### 作用

从 TensorBoard event 文件读取训练时真实记录的 PR 张量和分类别 AP 标量，生成 PR 曲线与分类别 AP 图。该脚本不会根据单个 AP 数值伪造 PR 曲线，因此日志中必须实际写入相应数据。

当 `--logdir` 指向包含多个运行目录的父目录时，脚本选择最新的 event 日志。

### 用法

```powershell
python src/export_tensorboard_pr_ap.py `
  --logdir output/Tv2_dmd640_scratch `
  --output paper/figures/tv2_pr_ap `
  --step best `
  --formats png pdf
```

显式指定标签模板和类别显示名：

```powershell
python src/export_tensorboard_pr_ap.py `
  --logdir output/Tv2_dmd640_scratch `
  --output paper/figures/tv2_pr_ap `
  --ap-tag-template "MetricsPerClass/{class}/ap50" `
  --map-tag Metrics/mAP50 `
  --class-alias military_tank=Tank `
  --class-alias military_soldier=Soldier
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--logdir` | 是 | 无 | 直接的 TensorBoard run 目录或其父目录。 |
| `--output` | 是 | 无 | 图像输出目录。 |
| `--step` | 否 | `best` | `best`、`latest` 或整数 step；`best` 按最高 mAP50 选择。 |
| `--pr-prefix` | 否 | `PRCurve/` | PR 张量 tag 前缀。 |
| `--ap-tag-template` | 否 | `auto` | 分类别 AP tag 模板，显式模板必须包含 `{class}`。 |
| `--map-tag` | 否 | `auto` | 总体 mAP 标量 tag。 |
| `--class-alias` | 否 | 无 | 可重复传入的 `TAG=LABEL` 类别显示名映射。 |
| `--iou` | 否 | `0.5` | 图标题显示的 IoU 标签，不会重新计算指标。 |
| `--dpi` | 否 | `300` | 栅格图 DPI。 |
| `--formats` | 否 | `png` | 一个或多个输出格式：`png`、`pdf`、`svg`。 |
| `--no-macro-curve` | 否 | 关闭 | 不绘制派生的宏平均 PR 曲线。 |
| `--curve-style` | 否 | `envelope` | `envelope` 绘制精度包络，`raw` 绘制原始点。 |

### 主要输出

- `pr_curves.<格式>`
- `per_class_ap.<格式>`

## 10. `plot_evaluation_report_figure.py`

### 作用

读取数据集评估生成的 `evaluation_report.json`，绘制四联图：

1. 分类别 Precision–Recall 曲线；
2. Precision/Recall/F1 与置信度阈值关系；
3. 分类别 AP50；
4. 小、中、大目标的 recall。

报告需要包含 `per_class`、`pr_data`、`size_recall` 和 `map50` 等完整字段。当前优先用于 `evaluate_teacher_v2_dataset.py` 生成的详细报告；若其他评估脚本生成的报告缺少这些字段，不能直接绘制。

### 用法

```powershell
python src/plot_evaluation_report_figure.py `
  --report output/Tv2_dmd640_scratch/eval_results_val/evaluation_report.json `
  --output-stem paper/figures/tv2_evaluation_summary `
  --dpi 600 `
  --panel-labels
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--report` | 否 | `output/Tv2_dmd640_scratch/eval_results_val/evaluation_report.json` | 数据集评估报告。 |
| `--output-stem` | 否 | `paper/figures/evaluation_report_summary` | 不带扩展名的输出路径。 |
| `--dpi` | 否 | `600` | PNG 分辨率。 |
| `--panel-labels` | 否 | 关闭 | 添加 `a`–`d` 面板编号。 |

脚本同时写出同名 PNG 和 PDF。

## 11. `plot_validation_performance_bars.py`

### 作用

生成数字教师模型与静态光学 SLM 模型在多个数据集上的 mAP50 分组柱状图。默认展示 Fashion 与 Military 的当前结果，也可完全通过命令行替换。

### 用法

使用默认数据：

```powershell
python src/plot_validation_performance_bars.py
```

传入自己的结果：

```powershell
python src/plot_validation_performance_bars.py `
  --datasets Fashion Military `
  --digital 0.970 0.839 `
  --static-slm 0.929 0.537 `
  --title "Validation mAP50" `
  --panel-label a `
  --output-stem paper/figures/current_validation_performance `
  --dpi 600
```

### 参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--datasets` | 否 | `Fashion Military` | 横轴数据集名称，可传多个。 |
| `--digital` | 否 | `0.970 0.839` | 数字教师模型在各数据集上的 mAP50。 |
| `--static-slm` | 否 | `0.929 0.537` | 静态 SLM 模型在各数据集上的 mAP50。 |
| `--title` | 否 | `Current validation performance` | 图标题；传空字符串可隐藏。 |
| `--panel-label` | 否 | 空 | 左上角面板编号。 |
| `--output-stem` | 否 | `paper/figures/current_validation_performance` | 不带扩展名的输出路径。 |
| `--dpi` | 否 | `600` | PNG 分辨率。 |

`--datasets`、`--digital` 和 `--static-slm` 的数量必须一致。脚本同时生成 PNG 与 PDF。

## 常见工作流

### 教师网络评估与论文图

```text
Teacher V2 checkpoint
  -> evaluate_teacher_v2_dataset.py
  -> evaluation_report.json
  -> plot_evaluation_report_figure.py
```

### 光学学生评估与错误数据审查

```text
Student + Light checkpoint
  -> evaluate_slm_light_dataset.py
  -> review_unrecognized_images.txt / review_candidates.jsonl
  -> dataset/review_delete_images.py --list ...
```

### 静态 SLM 相位导出与可视化

```text
Optical student checkpoint
  -> export_slm_phase.py
  -> SLM hardware PNG + phase metadata
  -> visualize_slm_phase.py
```

### 静态 SLM + Light 真实硬件链路

```text
Student + Light checkpoint + dataset
  -> export_slm_light_dataset_hardware.py
  -> 编号 DMD 输入 + SLM1/SLM2 各一张固定相位图
  -> CCD 视频采集与 ROI/透视校正
  -> detect_ccd_light_dataset.py（仅运行 Light 检测头）
```

### Teacher V2 真实硬件链路

```text
Teacher V2 checkpoint + dataset
  -> export_teacher_v2_dataset_hardware.py
  -> 编号 DMD 输入 + 每图动态 SLM 相位
  -> CCD 视频采集与 ROI/透视校正
  -> predict_teacher_v2_single_image.py --captured-feature ...
  -> Light 检测头硬件结果
```

## 常见问题

### 更换 Military 与 Fashion 数据集

评估脚本通过 `--data` 选择数据集，不会修改原有 Military 配置：

```powershell
python src/evaluate_teacher_v2_dataset.py `
  --checkpoint path/to/checkpoint.pth `
  --data data/fashion/data.yaml `
  --split val `
  --output output/fashion_eval `
  --device cuda
```

checkpoint 的类别数和类别顺序必须与 YAML 一致；Military 权重不能在类别定义不同的 Fashion 数据集上直接当作有效模型评估。

### 输出目录中已有旧结果

建议每个 checkpoint、数据集和 split 使用独立目录，避免将不同实验的 JSON、混淆矩阵和审查清单混在一起。例如：

```text
output/<experiment>/eval_results_val/
output/<experiment>/eval_results_test/
```

### CUDA 显存不足

先减小 `--batch-size`。单图脚本仍不足时，再检查模型配置和输入尺寸是否与 checkpoint 一致，不要仅为运行成功随意改变光学采样尺寸或硬件有效像元尺寸。
