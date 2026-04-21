import cv2
import numpy as np
import os
from pathlib import Path

def resize_images_to_640x640(input_dir, output_dir, target_size=(640, 640)):
    """
    将输入目录下的所有图片重设为640×640尺寸
    :param input_dir: 输入图片目录路径
    :param output_dir: 输出目录路径
    :param target_size: 目标尺寸 (宽, 高)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 获取输入目录下所有图片文件
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in supported_formats]
    
    if not image_files:
        print(f"在目录 {input_dir} 中未找到支持的图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件需要处理")
    
    processed_count = 0
    for image_file in image_files:
        try:
            # 读取图片
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"无法读取图片: {image_file}")
                continue
            
            # 获取原始尺寸
            original_h, original_w = img.shape[:2]
            target_w, target_h = target_size
            
            # 计算缩放比例，保持原比例
            scale = min(target_w / original_w, target_h / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            
            # 等比例缩放
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 计算填充位置，居中放置
            pad_left = (target_w - new_w) // 2
            pad_right = target_w - new_w - pad_left
            pad_top = (target_h - new_h) // 2
            pad_bottom = target_h - new_h - pad_top
            
            # 填充黑色背景，得到目标尺寸图片
            final_img = cv2.copyMakeBorder(
                resized,
                top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]  # 黑色填充
            )
            
            # 生成输出文件名（保持原文件名，添加后缀）
            output_filename = f"{image_file.stem}_resized{image_file.suffix}"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存图片
            cv2.imwrite(output_path, final_img)
            processed_count += 1
            print(f"处理完成: {image_file.name} -> {output_filename} "
                  f"(原始尺寸: {original_w}×{original_h}, 新尺寸: {target_w}×{target_h})")
            
        except Exception as e:
            print(f"处理图片 {image_file.name} 时出错: {e}")
    
    print(f"\n处理完成！共处理 {processed_count}/{len(image_files)} 个图片文件")
    print(f"输出目录: {output_dir}")

def batch_resize_images():
    """
    批量处理图片：将image_Origin目录下的图片重设为640×640并保存到image_Origin_size目录
    """
    # 设置路径
    base_dir = Path(__file__).parent
    input_dir = base_dir / "image_Origin"
    output_dir = base_dir / "image_Origin_size"
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"错误：输入目录不存在: {input_dir}")
        return
    
    print(f"开始批量处理图片...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 执行批量处理
    resize_images_to_640x640(str(input_dir), str(output_dir))

# ------------------- 示例调用 -------------------
if __name__ == "__main__":
    batch_resize_images()
