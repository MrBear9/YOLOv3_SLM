# 光学YOLOv3_SLM项目

一个基于光学传播模型和YOLOv3的先进目标检测系统，结合了空间光调制器(SLM)和教师-学生知识蒸馏技术。

## 项目概述

本项目实现了一个创新的光学-深度学习混合目标检测系统，主要特点包括：

- **光学前端调制**：两层SLM相位调制 + 角谱传播(ASM)的光学特征提取
- **YOLOv3检测头**：轻量化的多尺度目标检测架构
- **教师约束机制**：基于ConvTeacher的知识蒸馏约束
- **涡旋相位初始化**：支持涡旋相位调制和随机扰动
- **多数据集支持**：支持军事目标、Fashion-MNIST等数据集
- **差异化学习率**：光学层和检测头的独立学习率配置

## 项目结构

```
YOLOv3_SLM/
├── data/                           # 数据集目录
│   ├── military/                  # 军事目标数据集
│   │   ├── test/images/           # 测试图像（包含大量军事目标图像）
│   │   └── data.yaml              # 数据集配置
│   └── fashion_mnist/             # Fashion-MNIST数据集
│       ├── FashionMNIST/raw/       # 原始数据文件
│       └── data.yaml              # 数据集配置
├── .gitignore                     # Git忽略文件
├── Optical_class.py               # 光学YOLOv3核心实现
├── train_optical_yolo.py          # 通用训练脚本
├── optical_teacher_yolo_slm.py    # 光学教师YOLO-SLM实现
├── optical_teacher_yolo.py        # 光学教师YOLO实现
├── optical_teacher.py             # 光学教师模型
├── test_Phase_weight.py           # 相位层提取工具
├── tain_optical.py                # 光学训练脚本
├── last.py                        # 原始YOLOv3_SLM实现
├── teacher.py                     # 教师模型实现
└── README.md                      # 项目说明
```

## 核心功能模块

### 1. 空间光调制器层 (SLMLayer)

```python
class SLMLayer(nn.Module):
    """空间光调制器层，支持相位和振幅调制"""
```

**功能特性：**
- 支持纯相位调制和振幅-相位混合调制
- 涡旋相位初始化（可配置涡旋电荷）
- 随机扰动避免过强先验
- 复数场调制：field * exp(1j * phase)

### 2. 角谱传播 (ASMPropagation)

```python
class ASMPropagation(nn.Module):
    """角谱传播方法模拟光学传播过程"""
```

**传播特性：**
- 基于傅里叶变换的光学传播模拟
- 可配置波长、传播距离、像素尺寸
- 频域传递函数计算

### 3. 光学前端系统 (OpticalFrontend)

```python
class OpticalFrontend(nn.Module):
    """两层SLM光学前端：SLM1 → 传播1 → SLM2 → 传播2"""
```

**架构特点：**
- 两层SLM调制器串联
- 中间光学传播过程
- 强度到复数场的转换
- 可选的归一化处理

### 4. 教师网络 (ConvTeacher)

```python
class ConvTeacher(nn.Module):
    """教师网络生成密集约束图"""
```

**网络结构：**
- 三层卷积下采样 + 上采样恢复
- 输出与输入相同分辨率的约束图
- 支持预训练权重加载

### 5. 光学YOLOv3系统 (OpticalYOLOv3)

```python
class OpticalYOLOv3(nn.Module):
    """完整的光学YOLOv3系统：光学前端 + 检测头 + 教师约束"""
```

**完整架构：**
- **光学前端**：输入图像 → 光学调制 → 光学特征
- **检测头**：YOLOLightHead，三尺度检测(P3/P4/P5)
- **教师约束**：ConvTeacher生成目标特征约束
- **知识蒸馏**：教师-学生模式的联合训练

## 安装和依赖

### 系统要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (推荐)

### 安装依赖

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib tqdm
pip install numpy pillow scipy pyyaml
pip install albumentations (可选)
```

## 快速开始

### 1. 训练光学YOLOv3模型

**通用目标检测训练：**
```bash
python train_optical_yolo.py
```

**光学教师模型训练：**
```bash
python optical_teacher_yolo.py
```

**光学教师SLM模型训练：**
```bash
python optical_teacher_yolo_slm.py
```

### 2. 提取光学相位层

```bash
python test_Phase_weight.py
```

### 3. 模型配置参数

在训练脚本中可以配置以下关键参数：

```python
# 光学参数
OPTICAL_MODE = "phase"           # 调制模式：phase/amp_phase
WAVELENGTH = 532e-9             # 波长(米)
PROPAGATION_DISTANCE = 0.01      # 传播距离(米)
PIXEL_SIZE = 6.4e-6             # 像素尺寸(米)

# 涡旋初始化
USE_VORTEX_INIT = True          # 是否使用涡旋初始化
SLM1_VORTEX_CHARGE = 1          # SLM1涡旋电荷
SLM2_VORTEX_CHARGE = -1         # SLM2涡旋电荷
VORTEX_PERTURBATION = 0.12      # 涡旋相位扰动

# 训练参数
EPOCHS = 100                    # 训练轮次
LEARNING_RATE = 1e-3            # 学习率
BATCH_SIZE = 8                  # 批次大小
IMG_SIZE = 640                  # 输入图像尺寸

# 损失权重
BOX_WEIGHT = 0.05               # 边界框损失权重
OBJ_WEIGHT = 1.0                # 目标性损失权重
CLS_WEIGHT = 0.15               # 分类损失权重
TEACHER_CONSTRAINT_WEIGHT = 0.05 # 教师约束损失权重

# 教师网络配置
TEACHER_CHECKPOINT = None       # 教师模型权重路径
TEACHER_INIT_MODE = "checkpoint_or_random" # 初始化模式
FREEZE_TEACHER = True          # 是否冻结教师网络
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

## 训练流程

### 1. 两阶段训练策略

**阶段1：教师约束训练**
- 主要优化光学层参数
- 使用教师网络生成目标特征约束
- 损失函数：教师约束损失 + 检测损失

**阶段2：联合训练**
- 同时优化光学层和检测头
- 教师网络作为特征引导
- 损失函数：检测损失 + 教师约束损失

### 2. 差异化学习率

```python
# 光学层使用较低学习率
optical_lr = LEARNING_RATE * OPTICAL_LR_RATIO  # 通常为0.1
# 检测头使用标准学习率
detector_lr = LEARNING_RATE
```

### 3. 实时训练监控

```
Epoch 6/100, Train Loss: 0.1234, Val Loss: 0.1456, 
Precision: 0.856, Recall: 0.789, F1: 0.821, LR: 0.000900
Teacher Status: Loaded 32 teacher tensors from: checkpoint.pth | Teacher status: frozen
```

### 4. 可视化输出

训练过程中会生成以下可视化内容：
- **输入图像**：带真实标注和模型检测结果
- **光学特征**：光学调制后的特征图
- **教师约束**：教师网络生成的目标特征图
- **多尺度检测**：P3/P4/P5检测特征图
- **相位分布**：SLM层的相位调制图

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

使用 `test_Phase_weight.py` 可以提取和可视化SLM层的相位分布：
- SLM1相位调制图
- SLM2相位调制图  
- 光学传播过程中的场分布

## 技术特点

### 1. 光学优势
- **物理可解释性**：基于真实光学传播模型
- **计算效率**：光学前端替代部分卷积计算
- **特征增强**：光学调制增强目标相关特征

### 2. 深度学习集成
- **端到端训练**：光学层和检测头联合优化
- **知识蒸馏**：教师网络引导光学特征学习
- **多尺度检测**：YOLOv3架构的成熟检测能力

### 3. 可扩展性
- **模块化设计**：各组件可独立替换
- **参数可配置**：支持多种光学和训练配置
- **多数据集支持**：易于扩展到不同应用场景

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