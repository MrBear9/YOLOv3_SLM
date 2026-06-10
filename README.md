# 光学YOLOv3_SLM项目

基于光学传播模型和YOLOv3的目标检测系统，结合空间光调制器(SLM)和教师-学生知识蒸馏技术。

## 项目概述

- **光学前端调制**：两层SLM相位调制 + 角谱传播(ASM)
- **YOLOv3检测头**：轻量化多尺度目标检测
- **教师约束机制**：基于知识蒸馏的特征引导
- **涡旋相位初始化**：支持涡旋相位调制
- **多数据集支持**：军事目标、Fashion-MNIST等

## 项目结构

```
YOLOv3_SLM/
├── data/
│   ├── military/                  # 军事目标数据集
│   └── fashion_mnist/             # Fashion-MNIST数据集
├── optical_slm_yolov8_head.py     # 光学SLM YOLOv8检测头
├── optical_teacher_yolov8_head.py # 光学教师YOLOv8检测头
├── test_Phase_weight.py           # 相位层提取工具
├── optical_yolo_detect.py         # 光学YOLO检测脚本
└── README.md
```

## 安装

```bash
pip install torch torchvision opencv-python matplotlib tqdm numpy pillow scipy pyyaml
```

## 使用方法

### 训练模型

```bash
# 光学教师模型训练
python optical_teacher_yolov8_head.py                          # 单卡
torchrun --nproc_per_node=2 optical_teacher_yolov8_head.py     # 多卡

# 光学SLM模型训练
python optical_slm_yolov8_head.py                              # 单卡
torchrun --nproc_per_node=2 optical_slm_yolov8_head.py         # 多卡
```

> `torchrun` 是 PyTorch 官方推荐的 DDP 启动方式，自动设置环境变量，支持多卡训练。

### 目标检测
```bash
python optical_yolo_detect.py
```

### 提取相位层
```bash
python test_Phase_weight.py
```

### 核心参数
- `OPTICAL_MODE`: 调制模式 (phase/amp_phase)
- `WAVELENGTH`: 波长 (默认 532e-9m)
- `PROPAGATION_DISTANCE`: 传播距离 (默认 0.01m)
- `USE_VORTEX_INIT`: 涡旋相位初始化
- `EPOCHS`: 训练轮次
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率

## 数据集

### 军事目标数据集
- 类别：坦克、装甲车、军用卡车、军用直升机
- 配置：`data/military/data.yaml`

### Fashion-MNIST数据集
- 类别：10类服装物品
- 配置：`data/fashion_mnist/data.yaml`

## 核心模块

### SLMLayer
空间光调制器层，支持相位和振幅调制，涡旋相位初始化

### ASMPropagation
角谱传播方法，基于傅里叶变换的光学传播模拟

### OpticalFrontend
两层SLM光学前端：SLM1 → 传播1 → SLM2 → 传播2

### ConvTeacher
教师网络，生成密集特征约束图

### OpticalYOLOv3
完整系统：光学前端 + 检测头 + 教师约束

## 技术特点

- **物理可解释性**：基于真实光学传播模型
- **计算效率**：光学前端替代部分卷积计算
- **端到端训练**：光学层和检测头联合优化
- **知识蒸馏**：教师网络引导光学特征学习
- **多尺度检测**：适应不同大小的目标

## 应用场景

- 军事目标检测（坦克、装甲车等）
- 时尚物品识别
- 光学计算研究
- 混合光学-数字计算

## 许可证

MIT License

## 联系方式

- 项目主页: [MrBear9 (QingX)](https://github.com/MrBear9)
- 邮箱: bear211201@gmail.com