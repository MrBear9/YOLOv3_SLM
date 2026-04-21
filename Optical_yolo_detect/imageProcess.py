import cv2
import numpy as np
import os

def extract_and_resize_light_area(input_path, output_dir="imageProcess", target_size=(640, 640)):
    """
    自动截取亮光区域，扩展为指定大小并保存
    :param input_path: 输入图片路径
    :param output_dir: 输出目录
    :param target_size: 目标尺寸 (宽, 高)
    """
    # 1. 读取图片
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {input_path}")
    
    # 2. 转换为灰度图，便于阈值分割
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 自适应阈值分割，提取亮光区域（适配不同亮度）
    # 这里用大津法自动计算阈值，分离亮区和暗背景
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 形态学操作，去除噪点，连通亮区
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 查找轮廓，找到最大的亮区轮廓（数字区域）
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到亮光区域")
    
    # 找到面积最大的轮廓（主数字区域）
    max_contour = max(contours, key=cv2.contourArea)
    
    # 6. 获取轮廓的外接矩形，作为截取区域
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 7. 截取亮光区域（带一点边距，避免裁切过紧）
    padding = -10  # 边距，可根据需要调整
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    cropped = img[y1:y2, x1:x2]
    
    # 8. 等比例缩放 + 填充，扩展为 640×640
    h_crop, w_crop = cropped.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例，保持原比例
    scale = min(target_w / w_crop, target_h / h_crop)
    new_w = int(w_crop * scale)
    new_h = int(h_crop * scale)
    
    # 等比例缩放
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # 计算填充位置，居中放置
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    
    # 填充黑色背景，得到 640×640 图片
    final_img = cv2.copyMakeBorder(
        resized,
        top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # 黑色填充
    )
    
    # 9. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 10. 生成输出文件名
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_processed.png")
    
    # 11. 保存图片
    cv2.imwrite(output_path, final_img)
    print(f"处理完成！结果已保存到: {output_path}")
    
    return final_img, output_path

# ------------------- 示例调用 -------------------
if __name__ == "__main__":
    # 替换为你的输入图片路径
    input_image_path = "image_input/2026-02-06_16-18-00_454.jpg"  
    extract_and_resize_light_area(input_image_path)