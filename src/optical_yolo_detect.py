
"""
光学YOLO目标检测系统 - Optical YOLO Detection System
==========================================================

功能概述:
本系统实现了一个基于光学图像处理和YOLO目标检测的完整检测流程，主要功能包括：

1. 图像预处理阶段:
   - 使用光学图像处理技术提取亮光区域
   - 自动调整图像尺寸以适应检测模型输入要求
   - 灰度化处理以适配单通道检测模型

2. 目标检测阶段:
   - 加载预训练的光学YOLO检测器权重
   - 实现多尺度特征金字塔检测架构
   - 支持军事目标检测（坦克、战机、军舰等）

3. 结果后处理:
   - 非极大值抑制(NMS)去除重叠检测框
   - 按类别绘制彩色边界框和置信度标签
   - 自动保存检测结果到指定目录

4. 运行模式:
   - 单张图像检测模式：处理指定单张图像
   - 批量检测模式：处理指定目录下所有图像

文件结构说明:
- image_Origin/: 原始输入图像（手动放置的测试图像）
- image_input/: 探测器采集的图像（经过光学传播后的图像）
- imageProcess/: 处理后的图像（亮光区域提取+尺寸调整）
- imageDetect/: 最终检测结果图像（带边界框和标签）

作者: 光学检测系统开发团队  Mr. Bear
版本: 1.0
创建日期: 2026-04-17
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.ops import nms
import yaml
import sys

