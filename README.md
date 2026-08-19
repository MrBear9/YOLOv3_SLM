# YOLOv3_SLM 光学检测

两层相位 SLM + 角谱传播（ASM）+ 轻量 YOLOv8 检测头。当前主流程以
640×640 DMD 输入为基准，并明确区分真实硬件像元与经验传播采样间隔。

## 当前光路配置

| 平面 | 器件/配置 | 真实像元 | 有效传播采样 | 对齐 3.456 mm 的有效区 |
| --- | --- | ---: | ---: | ---: |
| 输入 | DMD 640×640 | 5.4 µm | 5.4 µm | 640×640 |
| SLM1 | `zk_weixing_8p0` | 8.0 µm | 11.0 µm | 432×432 |
| SLM2 | `magicholo_4p5` | 4.5 µm | 6.4 µm | 768×768 |

DMD 口径为 `640 × 5.4 µm = 3.456 mm`。SLM 有效区尺寸由真实像元计算；
ASM 使用从 `HolographSLM` 硬件实验继承的经验采样值。两者不能合并成一个
`PIXEL_SIZE`。详细原因和标定顺序见
[`OPTICAL_GEOMETRY.md`](OPTICAL_GEOMETRY.md)。

## 从零训练

教师配置已设置为 `TEACHER_INIT_MODE="scratch"`，输出到独立目录，避免加载或
覆盖旧的单一像元配置实验：

```powershell
& 'E:\Minicoda3\envs\deeplearn\python.exe' optical_teacher_yolov8_head.py
```

教师完成后，学生默认从新教师检查点
`output/Tv2_dmd640_scratch/teacher_detector_best.pth` 开始训练；学生相位本身仍为
随机初始化：

```powershell
& 'E:\Minicoda3\envs\deeplearn\python.exe' optical_slm_yolov8_head.py
```

不要把旧的单一 `PIXEL_SIZE` 教师或学生检查点混入这条训练链。

## 硬件相位导出

固定学生相位：

```powershell
& 'E:\Minicoda3\envs\deeplearn\python.exe' src/export_slm_phase.py `
  --checkpoint output/SLM_Tv2_dmd640_scratch/optical_student_best.pth `
  --output output/hardware_phase
```

导出的 `slm1.png` 为 432×432，`slm2.png` 为 768×768，分别对应相同的
3.456 mm 有效口径。它们是有效区图案；若硬件 SDK 要求整屏原生分辨率，应居中
嵌入、空白区域填零，禁止再缩放有效区。

输入相关的教师相位可用：

```powershell
& 'E:\Minicoda3\envs\deeplearn\python.exe' src/predict_teacher_v2_single_image.py `
  --image path/to/image.png `
  --checkpoint output/Tv2_dmd640_scratch/teacher_detector_best.pth `
  --output output/single_teacher_v2
```

## 关键代码

- `models/SLM/physical_defaults.py`：DMD/SLM 硬件事实与经验采样预设。
- `models/SLM/config_optical.py`：逐层配置访问器和口径校验。
- `models/SLM/asm_propagation.py`：零填充、带限 ASM。
- `models/SLM/slm_modulation.py`：相位调制、环形相位重采样和硬件导出。
- `models/teacher/physical_simulator.py`：教师的逐层物理传播。

## 仍需实测标定

当前 `11.0 µm / 6.4 µm` 只是有效模型初值。还需逐项标定传播距离/中继倍率、
SLM 灰度到相位 LUT、照明幅度、像差、两块 SLM 与相机的配准。改变其中任何一项
后都应新建输出目录并从教师重新训练。
