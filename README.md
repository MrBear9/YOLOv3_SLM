# 光学YOLOv3_SLM项目

一个基于光学传播模型和YOLOv3的先进目标检测系统，结合了光学调制和深度学习技术。

## 项目概述

本项目实现了一个创新的光学-深度学习混合目标检测系统，主要特点包括：

- **光学前端调制**：使用空间光调制器(SLM)进行光学特征提取
- **YOLOv3检测头**：基于YOLOv3的多尺度目标检测
- **光学约束机制**：类似MultiScaleTeacher的边缘和纹理引导
- **多数据集支持**：支持军事目标、Fashion-MNIST等数据集

## 项目结构

```
YOLOv3_SLM/
├── data/                           # 数据集目录
│   ├── military/                  # 军事目标数据集
│   │   ├── test/images/           # 测试图像
│   │   └── data.yaml              # 数据集配置
│   └── fashion_mnist/             # Fashion-MNIST数据集
│       ├── FashionMNIST/raw/       # 原始数据
│       └── data.yaml              # 数据集配置
├── output/                        # 输出目录
│   └── fashion_mnist_models/      # 训练模型
├── Optical_class.py               # 光学YOLOv3核心实现
├── train_optical_yolo.py          # 通用训练脚本
├── train_fashion_mnist.py         # Fashion-MNIST训练脚本
├── test_Phase_weight.py           # 相位层提取工具
├── last.py                        # 原始YOLOv3_SLM实现
└── README.md                      # 项目说明
```

## 核心功能模块

### 1. 光学前端调制 (OpticalFrontend)

```python
class OpticalFrontend(nn.Module):
    """光学前端：两层相位调制 + 角谱传播"""
```

**功能特性：**
- 两层相位调制器(SLM1, SLM2)
- 角谱传播方法(ASM)模拟光学传播
- 支持相位和振幅调制模式
- 可配置的光学参数（波长、传播距离等）

### 2. 光学YOLOv3系统 (OpticalYOLOv3)

```python
class OpticalYOLOv3(nn.Module):
    """完整的光学YOLOv3系统：光学前端 + YOLOv3检测头"""
```

**架构特点：**
- **光学前端**：输入图像 → 光学调制 → 光学特征
- **检测头**：YOLOLightHead，三尺度检测(P3/P4/P5)
- **光学约束**：边缘和纹理特征引导

### 3. 光学约束机制 (OpticalConstraint)

```python
class OpticalConstraint(nn.Module):
    """光学层约束机制 - 引导光学调制学习边缘和纹理特征"""
```

**约束组件：**
- **高斯低通滤波器**：平滑约束，学习低频特征
- **Sobel边缘检测器**：边缘约束，学习高频特征
- **特征组合**：类似MultiScaleTeacher的输出格式

## 安装和依赖

### 系统要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (推荐)

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib tqdm
pip install numpy pillow scipy
pip install albumentations (可选)
```

## 快速开始

### 1. 训练光学YOLOv3模型

**通用目标检测训练：**
```bash
python train_optical_yolo.py
```

**Fashion-MNIST时尚物品检测：**
```bash
python train_fashion_mnist.py
```

### 2. 提取光学相位层

```bash
python test_Phase_weight.py
```

### 3. 模型配置参数

在训练脚本中可以配置以下关键参数：

```python
# 光学参数
OPTICAL_MODE = "phase"           # 调制模式：phase/amplitude
WAVELENGTH = 532e-9             # 波长(米)
PROPAGATION_DISTANCE = 0.1       # 传播距离(米)

# 训练参数
EPOCHS = 100                    # 训练轮次
LEARNING_RATE = 0.001           # 学习率
BATCH_SIZE = 8                  # 批次大小
IMG_SIZE = 640                  # 输入图像尺寸

# 损失权重
BOX_WEIGHT = 0.05               # 边界框损失权重
OBJ_WEIGHT = 1.5                # 目标性损失权重
CLS_WEIGHT = 0.15               # 分类损失权重
OPTICAL_CONSTRAINT_WEIGHT = 0.1 # 光学约束损失权重
```

## 数据集配置

### 军事目标数据集

**data/military/data.yaml:**
```yaml
names:
  - 坦克
  - 装甲车
  - 军用卡车
  - 军用直升机
nc: 4
test: data/military/test/images
train: data/military/train/images
val: data/military/val/images
```

### Fashion-MNIST数据集

**data/fashion_mnist/data.yaml:**
```yaml
names:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot
nc: 10
```

## 训练过程监控

### 实时训练指标

```
Epoch 6/100, Train Loss: 0.1234, Val Loss: 0.1456, 
Precision: 0.856, Recall: 0.789, F1: 0.821, LR: 0.000900
```

### 可视化输出

训练过程中会生成以下可视化内容：
- **输入图像**：带真实标注和模型检测结果
- **光学特征**：光学调制后的特征图
- **约束目标**：光学层学习的目标特征
- **多尺度特征**：P3/P4/P5检测特征图

## 模型输出格式

### 检测结果解码

模型输出三个尺度的检测结果：
- **P3** (80×80)：小目标检测
- **P4** (40×40)：中目标检测  
- **P5** (20×20)：大目标检测

每个检测框包含：
```python
[x, y, width, height, confidence, class_id]
```

### 相位层提取

使用 `test_Phase_weight.py` 可以提取训练后的光学相位层：

```python
# 提取相位层
phase_layers = save_phase_layers(
    "output/fashion_mnist_models/fashion_mnist_epoch_90.pth",
    "output/fashion_mnist_phase_layers"
)
```

输出文件包括：
- `.npy`：原始相位数据
- `.png`：相位图像
- `_info.txt`：相位统计信息

## 技术特点

### 1. 光学-深度学习融合
- **物理可解释性**：光学调制过程具有物理意义
- **计算效率**：光学前端减少计算复杂度
- **特征增强**：光学约束引导学习有意义的特征

### 2. 多尺度检测
- **三尺度特征金字塔**：适应不同大小的目标
- **锚框优化**：针对不同数据集优化锚框尺寸
- **非极大值抑制**：去除重叠检测框

### 3. 训练策略优化
- **渐进式约束**：训练稳定后启用光学约束
- **学习率调度**：自适应学习率调整
- **内存优化**：CUDA内存管理和批处理优化

## 性能指标

### 检测性能
- **精确率(Precision)**：检测准确性的度量
- **召回率(Recall)**：目标检出能力的度量  
- **F1分数**：精确率和召回率的综合指标

### 光学特征质量
- **相位分布**：相位层的统计特性
- **特征可解释性**：光学特征与视觉特征的相关性
- **调制效果**：光学调制对检测性能的贡献

## 应用场景

### 1. 军事目标检测
- 坦克、装甲车等军事装备识别
- 复杂背景下的目标检测
- 实时监控和预警系统

### 2. 时尚物品识别
- Fashion-MNIST数据集分类
- 服装款式识别和推荐
- 电商图像分析

### 3. 光学计算研究
- 光学神经网络研究
- 混合光学-数字计算
- 新型计算架构探索

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   BATCH_SIZE = 4
   # 启用内存优化
   torch.cuda.empty_cache()
   ```

2. **依赖项缺失**
   ```bash
   # 安装缺失的包
   pip install missing_package_name
   ```

3. **数据集路径错误**
   ```python
   # 检查data.yaml文件路径
   YAML_PATH = "data/military/data.yaml"
   ```

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 贡献指南

欢迎贡献代码和提出改进建议！

### 代码规范
- 遵循PEP 8代码风格
- 添加详细的文档字符串
- 包含单元测试

### 功能扩展
- 支持新的光学调制方法
- 添加新的数据集支持
- 优化训练策略

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{optical_yolov3_slm,
  title = {光学YOLOv3_SLM: 基于光学传播模型的目标检测系统},
  author = {MrBear9},
  year = {2026},
  url = {https://github.com/MrBear9/YOLOv3_SLM.git}
}
```

## 联系方式

- 项目主页: [[MrBear9 (QingX)](https://github.com/MrBear9)]
- 问题反馈: [[Issues · MrBear9/YOLOv3_SLM](https://github.com/MrBear9/YOLOv3_SLM/issues)]
- 邮箱: bear211201@gmail.com

---

**注意**: 本项目仍在积极开发中，API可能会发生变化。建议定期更新到最新版本。