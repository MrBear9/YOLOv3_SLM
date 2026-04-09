import torch
import numpy as np
import os
from PIL import Image


def save_phase_layers(pth_file_path, output_dir="output/images"):
    """
    从光学YOLOv3模型的 .pth 文件中提取相位层并保存为原始图像
    专门针对光学前端的两层相位调制参数

    参数:
        pth_file_path: .pth 文件路径
        output_dir: 输出目录路径
    """

    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"输出目录: {os.path.abspath(output_dir)}")

        # 加载 .pth 文件
        checkpoint = torch.load(pth_file_path, map_location='cpu')

        print(f"文件类型: {type(checkpoint)}")

        # 检查文件类型并获取状态字典
        if isinstance(checkpoint, dict):
            print("检测到字典结构，分析内容...")
            print(f"顶层键: {list(checkpoint.keys())}")
            
            # 根据您的错误信息，模型参数在 model_state_dict 中
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("使用 model_state_dict 作为状态字典")
                print(f"model_state_dict 类型: {type(state_dict)}")
                print(f"model_state_dict 键数量: {len(state_dict)}")
                
                # 显示前几个键，确认结构
                keys_list = list(state_dict.keys())
                print(f"前5个键: {keys_list[:5]}")
                
                # 检查是否包含相位层
                phase_keys = [k for k in keys_list if 'phase' in k.lower()]
                if phase_keys:
                    print(f"找到相位相关键: {phase_keys}")
                
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("使用 state_dict")
            else:
                state_dict = checkpoint
                print("使用整个字典作为状态字典")
        else:
            state_dict = checkpoint
            print("直接加载状态字典")

        # 查找光学YOLOv3模型的相位层
        phase_layers = {}
        
        # 光学YOLOv3模型的相位层命名模式
        phase_keywords = ['phase', 'slm', 'optical']
        
        print("搜索光学YOLOv3模型的相位层...")
        for key, tensor in state_dict.items():
            # 查找包含相位相关关键词的层
            if any(keyword in key.lower() for keyword in phase_keywords):
                phase_layers[key] = tensor
                print(f"找到相位层: {key}, 形状: {tensor.shape}")
            # 特别查找光学前端的相位参数
            elif 'optical_frontend' in key and ('phase' in key or 'slm' in key):
                phase_layers[key] = tensor
                print(f"找到光学前端相位层: {key}, 形状: {tensor.shape}")

        if not phase_layers:
            print("警告: 未找到明确的相位层，显示所有张量层进行诊断:")
            # 过滤出张量层
            tensor_layers = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            for key, tensor in tensor_layers.items():
                print(f"  {key}: {tensor.shape}")
                # 自动识别可能的相位层
                if tensor.dim() == 4 and tensor.shape[0] == 1 and tensor.shape[1] == 1:
                    # 可能是相位调制层
                    if key not in phase_layers:
                        phase_layers[key] = tensor
                        print(f"  -> 识别为潜在相位层: {key}")

        if not phase_layers:
            print("将尝试保存所有层进行进一步分析...")
            phase_layers = state_dict

        # 保存每个相位层的原始图像
        print(f"\n开始保存相位层图像...")
        saved_files = []

        for layer_name, tensor in phase_layers.items():
            # 转换为numpy数组
            weight = tensor.cpu().numpy()
            print(f"\n处理层: {layer_name}")
            print(f"  原始形状: {weight.shape}")
            print(f"  数据类型: {weight.dtype}")
            print(f"  数值范围: [{weight.min():.6f}, {weight.max():.6f}]")

            # 处理光学相位层的特殊形状
            if len(weight.shape) == 4:
                # 4D张量: 通常是 (1, 1, H, W) 的光学相位层
                if weight.shape[0] == 1 and weight.shape[1] == 1:
                    # 标准光学相位层: (1, 1, H, W) -> (H, W)
                    weight_2d = weight.squeeze()
                    print(f"  光学相位层: {weight.shape} -> {weight_2d.shape}")
                elif weight.shape[0] == 1:
                    # (1, C, H, W) -> 取第一个通道
                    weight_2d = weight[0, 0] if weight.shape[1] > 0 else weight[0]
                    print(f"  多通道相位层: {weight.shape} -> {weight_2d.shape}")
                else:
                    # 其他4D形状，取第一个切片
                    weight_2d = weight[0, 0] if weight.shape[1] > 0 else weight[0]
                    print(f"  非常规4D相位层: {weight.shape} -> {weight_2d.shape}")

            elif len(weight.shape) == 2:
                # 2D张量: 直接使用
                weight_2d = weight
                print(f"  2D相位层: {weight_2d.shape}")

            elif len(weight.shape) == 1:
                # 1D张量: 尝试转换为2D（光学相位通常是2D）
                size = int(np.sqrt(weight.shape[0]))
                if size * size == weight.shape[0]:
                    weight_2d = weight.reshape(size, size)
                    print(f"  1D转换为2D相位层: {weight.shape} -> {weight_2d.shape}")
                else:
                    # 如果不是方阵，创建近似矩形
                    rows = int(np.sqrt(weight.shape[0]))
                    cols = weight.shape[0] // rows
                    if rows * cols == weight.shape[0]:
                        weight_2d = weight.reshape(rows, cols)
                        print(f"  1D转换为矩形相位层: {weight.shape} -> {weight_2d.shape}")
                    else:
                        print(f"  警告: 无法将1D张量转换为2D，跳过")
                        continue

            else:
                print(f"  警告: 不支持的光学相位层维度 {len(weight.shape)}，跳过")
                continue

            # 生成安全的文件名
            safe_name = layer_name.replace('.', '_').replace(':', '_').replace('/', '_')

            # 1. 保存为原始二进制文件 (.npy)
            npy_path = os.path.join(output_dir, f"{safe_name}_raw.npy")
            np.save(npy_path, weight_2d)
            print(f"  已保存原始数据: {npy_path}")

            # 2. 保存为相位图像文件 (.png)
            # 光学相位数据通常在 [0, 2π] 范围内
            if weight_2d.dtype == np.float32 or weight_2d.dtype == np.float64:
                # 浮点相位数据
                data_min, data_max = weight_2d.min(), weight_2d.max()
                print(f"  相位数据范围: [{data_min:.6f}, {data_max:.6f}]")
                
                # 检查是否为相位数据（通常在0-2π或0-6.28范围内）
                if data_max <= 6.3:  # 约等于2π
                    print("  检测到标准相位数据 [0, 2π]")
                    # 相位数据直接映射到0-255
                    if data_max - data_min > 0:
                        normalized = (weight_2d - data_min) / (data_max - data_min) * 255
                        image_data = normalized.astype(np.uint8)
                    else:
                        image_data = np.full_like(weight_2d, 128, dtype=np.uint8)
                else:
                    # 其他浮点数据，使用16位保存
                    print("  检测到一般浮点数据")
                    if data_max - data_min > 0:
                        normalized = (weight_2d - data_min) / (data_max - data_min) * 65535
                        image_data = normalized.astype(np.uint16)
                    else:
                        image_data = np.full_like(weight_2d, 32768, dtype=np.uint16)
            else:
                # 整数数据
                image_data = weight_2d.astype(np.uint16)

            # 保存图像
            if image_data.dtype == np.uint8:
                img = Image.fromarray(image_data, mode='L')
                img_path = os.path.join(output_dir, f"{safe_name}.png")
                img.save(img_path)
                print(f"  已保存8位相位图像: {img_path}")
            else:
                img = Image.fromarray(image_data)
                img_path = os.path.join(output_dir, f"{safe_name}.png")
                img.save(img_path)
                print(f"  已保存16位原始图像: {img_path}")

            # 3. 保存为文本文件，便于查看
            txt_path = os.path.join(output_dir, f"{safe_name}_info.txt")
            with open(txt_path, 'w') as f:
                f.write(f"图层名称: {layer_name}\n")
                f.write(f"原始形状: {tensor.shape}\n")
                f.write(f"处理后形状: {weight_2d.shape}\n")
                f.write(f"数据类型: {weight_2d.dtype}\n")
                f.write(f"最小值: {weight_2d.min()}\n")
                f.write(f"最大值: {weight_2d.max()}\n")
                f.write(f"平均值: {weight_2d.mean()}\n")
                f.write(f"标准差: {weight_2d.std()}\n\n")

                # 写入相位数据的统计信息
                f.write("相位数据统计:\n")
                f.write(f"  数据范围: [{weight_2d.min():.6f}, {weight_2d.max():.6f}]\n")
                f.write(f"  平均值: {weight_2d.mean():.6f}\n")
                f.write(f"  标准差: {weight_2d.std():.6f}\n")
                
                # 如果是相位数据，计算相位统计
                if weight_2d.max() <= 6.3:
                    f.write("  相位统计:\n")
                    f.write(f"    2π周期数: {weight_2d.max() / (2 * np.pi):.2f}\n")
                    f.write(f"    相位变化范围: {weight_2d.max() - weight_2d.min():.3f} rad\n")
                
                # 写入前10x10矩阵的值
                f.write("\n矩阵前10x10值:\n")
                rows, cols = min(10, weight_2d.shape[0]), min(10, weight_2d.shape[1])
                for i in range(rows):
                    row_vals = [f"{weight_2d[i, j]:.6f}" for j in range(cols)]
                    f.write("  " + "  ".join(row_vals) + "\n")

            print(f"  已保存信息文件: {txt_path}")

            saved_files.append({
                'name': layer_name,
                'original_shape': tensor.shape,
                'shape': weight_2d.shape,
                'dtype': str(weight_2d.dtype),
                'min': weight_2d.min(),
                'max': weight_2d.max(),
                'mean': weight_2d.mean(),
                'std': weight_2d.std(),
                'npy_path': npy_path,
                'img_path': img_path,
                'info_path': txt_path
            })

        # 创建详细的汇总文件
        summary_path = os.path.join(output_dir, "optical_phase_layers_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("光学YOLOv3相位层提取汇总\n")
            f.write("=" * 60 + "\n")
            f.write(f"源文件: {pth_file_path}\n")
            f.write(f"提取时间: {np.datetime64('now')}\n")
            f.write(f"总相位层数: {len(saved_files)}\n")
            f.write(f"模型类型: 光学YOLOv3 (Fashion-MNIST训练)\n\n")

            # 相位层分类统计
            slm_layers = [item for item in saved_files if 'slm' in item['name'].lower()]
            optical_layers = [item for item in saved_files if 'optical' in item['name'].lower()]
            phase_layers = [item for item in saved_files if 'phase' in item['name'].lower()]
            
            f.write("相位层分类统计:\n")
            f.write(f"  SLM相关层: {len(slm_layers)} 个\n")
            f.write(f"  光学相关层: {len(optical_layers)} 个\n")
            f.write(f"  相位相关层: {len(phase_layers)} 个\n\n")

            f.write("各层详细信息:\n")
            f.write("-" * 60 + "\n")

            for i, item in enumerate(saved_files):
                f.write(f"\n{i + 1}. 层名称: {item['name']}\n")
                f.write(f"   形状: {item['shape']}\n")
                f.write(f"   数据类型: {item['dtype']}\n")
                f.write(f"   数值范围: [{item['min']:.6f}, {item['max']:.6f}]\n")
                
                # 相位特定信息
                if item['max'] <= 6.3:
                    f.write(f"   相位范围: {item['max'] - item['min']:.3f} rad\n")
                    f.write(f"   2π周期: {item['max'] / (2 * np.pi):.2f}\n")
                
                f.write(f"   平均值: {item['mean']:.6f}\n")
                f.write(f"   标准差: {np.sqrt(np.mean((weight_2d - item['mean'])**2)):.6f}\n")
                f.write(f"   原始数据: {os.path.basename(item['npy_path'])}\n")
                f.write(f"   图像文件: {os.path.basename(item['img_path'])}\n")
                f.write(f"   信息文件: {os.path.basename(item['info_path'])}\n")

        print(f"\n" + "=" * 70)
        print(f"光学YOLOv3相位层提取完成!")
        print(f"总计提取 {len(saved_files)} 个相位层")
        print(f"输出目录: {os.path.abspath(output_dir)}")
        print(f"汇总文件: {summary_path}")
        
        # 显示关键相位层信息
        if saved_files:
            print("\n关键相位层信息:")
            print("-" * 70)
            for i, item in enumerate(saved_files[:5]):  # 显示前5个
                phase_info = ""
                if item['max'] <= 6.3:
                    phase_info = f" | 相位范围: {item['max'] - item['min']:.3f} rad"
                print(f"{i+1:2d}. {item['name'][:40]:40s} | {str(item['shape']):15s}{phase_info}")
        
        print("=" * 70)

        return saved_files

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_specific_phase_layers(pth_file_path, layer_names=None, output_dir="output/images"):
    """
    提取特定的相位层

    参数:
        pth_file_path: .pth 文件路径
        layer_names: 要提取的特定层名称列表
        output_dir: 输出目录路径
    """
    # 创建特定输出目录
    if layer_names:
        output_dir = os.path.join(output_dir, "specific_layers")

    return save_phase_layers(pth_file_path, output_dir)


# 使用示例
if __name__ == "__main__":
    # 设置输入文件和输出目录
    pth_file = "output/optical_yolo_models/optical_yolo_epoch_100_20260409_062717.pth"  # Fashion-MNIST训练模型
    output_directory = "output/optical_yolo_phase_layers"  # 输出目录

    print("=" * 70)
    print("光学YOLOv3相位层提取工具")
    print("专门用于提取Fashion-MNIST训练的光学模型相位参数")
    print("=" * 70)
    print(f"模型文件: {pth_file}")
    print(f"输出目录: {output_directory}")
    print("=" * 70)

    # 提取所有相位层
    saved_layers = save_phase_layers(pth_file, output_directory)

    if saved_layers:
        print("\n提取的层列表:")
        for i, layer in enumerate(saved_layers):
            print(f"{i + 1:3d}. {layer['name']:40s} -> {layer['img_path']}")

        # 提供进一步处理的选项
        print("\n下一步操作建议:")
        print("1. 查看提取的相位图像文件 (.png)")
        print("2. 分析 .npy 文件中的原始相位数据")
        print("3. 检查汇总文件: optical_phase_layers_summary.txt")
        print("4. 使用相位数据进行光学仿真或可视化")
        print("5. 比较不同训练阶段的相位分布变化")
    else:
        print("\n未能提取任何相位层")

        # 尝试重新加载并显示所有层
        try:
            checkpoint = torch.load(pth_file, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                print("\n模型文件中的所有参数层:")
                print("-" * 70)
                print("序号 | 参数名称                                      | 形状              | 类型")
                print("-" * 70)
                
                # 先显示张量层
                tensor_count = 0
                for i, (key, value) in enumerate(state_dict.items()):
                    if isinstance(value, torch.Tensor):
                        tensor_count += 1
                        # 标记可能的相位层
                        phase_marker = ""
                        if any(keyword in key.lower() for keyword in ['phase', 'slm', 'optical']):
                            phase_marker = " [相位层]"
                        elif value.dim() == 4 and value.shape[0] == 1 and value.shape[1] == 1:
                            phase_marker = " [潜在相位层]"
                        
                        print(f"{tensor_count:3d}. {key:40s} : {str(value.shape):15s} - {value.dtype}{phase_marker}")
                
                # 再显示非张量项
                non_tensor_count = 0
                for i, (key, value) in enumerate(state_dict.items()):
                    if not isinstance(value, torch.Tensor):
                        non_tensor_count += 1
                        value_str = str(value)
                        if len(value_str) > 20:
                            value_str = value_str[:20] + "..."
                        print(f"{non_tensor_count + tensor_count:3d}. {key:40s} : {type(value).__name__:15s} - {value_str}")
        except Exception as e:
            print(f"重新加载失败: {e}")