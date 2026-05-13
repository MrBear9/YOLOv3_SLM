import numpy as np
import cv2
import os

# ============================================
# 1. 生成 DMD 棋盘格 (二值 0/255)
# ============================================
def generate_dmd_checkerboard(size=640, block=8):
    """
    生成 DMD 二值棋盘格图像。
    参数:
        size: 图像边长（像素）
        block: 棋盘格方块边长（像素）
    返回:
        dtype=np.uint8, 值 0 或 255
    """
    n_blocks = size // block
    base = np.zeros((n_blocks, n_blocks), dtype=np.uint8)
    base[0::2, 0::2] = 1
    base[1::2, 1::2] = 1
    board = np.kron(base, np.ones((block, block), dtype=np.uint8))
    dmd_img = np.where(board == 1, 255, 0).astype(np.uint8)
    return dmd_img

# ============================================
# 2. 生成 SLM 相位棋盘格 (0 相位 / π 相位)
# ============================================
def generate_slm_phase_checkerboard(size=640, block=8):
    """
    生成 SLM 二进制相位棋盘格。
    0 对应 0 相位（灰度 0）
    128 对应 π 相位（灰度 128）
    """
    n_blocks = size // block
    base = np.zeros((n_blocks, n_blocks), dtype=np.uint8)
    base[0::2, 0::2] = 1
    base[1::2, 1::2] = 1
    board = np.kron(base, np.ones((block, block), dtype=np.uint8))
    slm_img = np.where(board == 1, 128, 0).astype(np.uint8)
    return slm_img

# ============================================
# 3. 生成 DMD 全白图案（所有微镜 on）
# ============================================
def generate_dmd_white(size=640):
    """生成 DMD 全白图像，所有像素为 255"""
    return np.full((size, size), 255, dtype=np.uint8)

# ============================================
# 4. 生成 SLM 均匀相位（全 0 相位）
# ============================================
def generate_slm_uniform_phase(size=640, phase_value=0):
    """
    生成 SLM 均匀相位图案。
    phase_value: 灰度值，0 对应 0 相位，通常设为 0。
    """
    return np.full((size, size), phase_value, dtype=np.uint8)

# ============================================
# 5. （可选）生成 DMD 全黑图案
# ============================================
def generate_dmd_black(size=640):
    """生成 DMD 全黑图像，所有像素为 0"""
    return np.zeros((size, size), dtype=np.uint8)

# ============================================
# 主程序：生成并保存所有标定图案
# ============================================
if __name__ == "__main__":
    SIZE = 640          # 图像尺寸
    BLOCK = 64           # 棋盘格方块大小
    outputdir = "Optical_yolo_detect/DMD_SLM_checkerborad"

    if not os.path.exists(outputdir):
        os.makedirs(outputdir, exist_ok=True)

    # 生成各图案
    dmd_checker = generate_dmd_checkerboard(SIZE, BLOCK)
    slm_phase_checker = generate_slm_phase_checkerboard(SIZE, BLOCK)
    dmd_white = generate_dmd_white(SIZE)
    slm_uniform = generate_slm_uniform_phase(SIZE, phase_value=0)
    dmd_black = generate_dmd_black(SIZE)   # 备用

    # 保存为 PNG（无损）
    cv2.imwrite(outputdir + "/dmd_checkerboard.png", dmd_checker)
    cv2.imwrite(outputdir + "/slm_phase_checkerboard.png", slm_phase_checker)
    cv2.imwrite(outputdir + "/dmd_white.png", dmd_white)
    cv2.imwrite(outputdir + "/slm_uniform_phase.png", slm_uniform)
    cv2.imwrite(outputdir + "/dmd_black.png", dmd_black)

    print("所有标定图案已生成并保存为 PNG 文件。")