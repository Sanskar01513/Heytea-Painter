"""
智能图像缩放模块
使用高质量算法防止精度损失
"""

import cv2
import numpy as np


def smart_downscale(image, target_width, target_height):
    """
    智能下采样：使用多级缩放和锐化来保留细节
    
    参数:
        image: 输入图像 (OpenCV 格式)
        target_width: 目标宽度
        target_height: 目标高度
    
    返回:
        缩放后的图像
    """
    h, w = image.shape[:2]
    
    # 如果目标尺寸更大，使用上采样
    if target_width > w or target_height > h:
        return smart_upscale(image, target_width, target_height)
    
    # 计算缩放比例
    scale_w = target_width / w
    scale_h = target_height / h
    
    # 如果缩放比例接近1.0，直接返回
    if scale_w > 0.95 and scale_h > 0.95:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # 多级缩放策略：避免一次性大幅缩小导致信息损失
    current = image.copy()
    
    # 如果缩放比例小于0.5，分多步进行
    while True:
        curr_h, curr_w = current.shape[:2]
        
        # 计算当前到目标的比例
        ratio_w = target_width / curr_w
        ratio_h = target_height / curr_h
        ratio = min(ratio_w, ratio_h)
        
        # 如果接近目标，直接缩放到目标尺寸
        if ratio > 0.5:
            current = cv2.resize(current, (target_width, target_height), 
                               interpolation=cv2.INTER_AREA)
            break
        
        # 否则缩小到当前的一半
        new_w = curr_w // 2
        new_h = curr_h // 2
        current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 应用锐化滤波器来恢复边缘细节
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) / 1.0
    sharpened = cv2.filter2D(current, -1, kernel)
    
    # 混合原图和锐化图（防止过度锐化）
    result = cv2.addWeighted(current, 0.7, sharpened, 0.3, 0)
    
    return result


def smart_upscale(image, target_width, target_height):
    """
    智能上采样：使用边缘保留插值
    
    参数:
        image: 输入图像
        target_width: 目标宽度
        target_height: 目标高度
    
    返回:
        放大后的图像
    """
    h, w = image.shape[:2]
    
    # 计算缩放比例
    scale_w = target_width / w
    scale_h = target_height / h
    
    # 如果缩放比例小于2倍，使用 Lanczos 插值
    if scale_w < 2.0 and scale_h < 2.0:
        return cv2.resize(image, (target_width, target_height), 
                         interpolation=cv2.INTER_LANCZOS4)
    
    # 大倍数放大：分步进行
    current = image.copy()
    
    while True:
        curr_h, curr_w = current.shape[:2]
        
        # 如果已经接近或超过目标尺寸
        if curr_w >= target_width * 0.8 or curr_h >= target_height * 0.8:
            current = cv2.resize(current, (target_width, target_height),
                               interpolation=cv2.INTER_LANCZOS4)
            break
        
        # 放大到2倍
        new_w = min(curr_w * 2, target_width)
        new_h = min(curr_h * 2, target_height)
        current = cv2.resize(current, (new_w, new_h), 
                           interpolation=cv2.INTER_CUBIC)
    
    # 应用边缘增强
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if len(current.shape) == 3 else current
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(current.shape) == 3 else edges
    
    # 轻微混合边缘来增强清晰度
    result = cv2.addWeighted(current, 0.95, edges_colored, 0.05, 0)
    
    return result


def adaptive_resize(image, target_width, target_height, preserve_detail=True):
    """
    自适应缩放：根据缩放方向和比例选择最佳算法
    
    参数:
        image: 输入图像
        target_width: 目标宽度
        target_height: 目标高度
        preserve_detail: 是否优先保留细节（会稍慢）
    
    返回:
        缩放后的图像
    """
    h, w = image.shape[:2]
    
    # 如果尺寸相同，直接返回
    if w == target_width and h == target_height:
        return image.copy()
    
    # 决定使用哪种缩放方法
    scale_ratio = (target_width * target_height) / (w * h)
    
    if preserve_detail:
        # 高质量模式
        if scale_ratio < 1.0:
            return smart_downscale(image, target_width, target_height)
        else:
            return smart_upscale(image, target_width, target_height)
    else:
        # 快速模式
        if scale_ratio < 1.0:
            return cv2.resize(image, (target_width, target_height), 
                            interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(image, (target_width, target_height),
                            interpolation=cv2.INTER_CUBIC)


def calculate_optimal_size(original_w, original_h, canvas_w, canvas_h, 
                          min_scale=0.5, max_scale=2.0):
    """
    计算最优图像尺寸
    
    参数:
        original_w, original_h: 原始图像尺寸
        canvas_w, canvas_h: 画布尺寸
        min_scale: 最小缩放比例（防止过度缩小）
        max_scale: 最大缩放比例（防止过度放大）
    
    返回:
        (optimal_w, optimal_h, scale): 最优尺寸和缩放比例
    """
    # 计算宽高比
    img_ratio = original_w / original_h
    canvas_ratio = canvas_w / canvas_h
    
    # 根据宽高比决定缩放策略
    if abs(img_ratio - canvas_ratio) < 0.1:
        # 宽高比接近：可以填满画布
        target_w = canvas_w
        target_h = canvas_h
    elif img_ratio > canvas_ratio:
        # 图片更宽：按宽度适配
        target_w = canvas_w
        target_h = int(canvas_w / img_ratio)
    else:
        # 图片更高：按高度适配
        target_h = canvas_h
        target_w = int(canvas_h * img_ratio)
    
    # 计算缩放比例
    scale = target_w / original_w
    
    # 限制缩放比例
    if scale < min_scale:
        scale = min_scale
        target_w = int(original_w * scale)
        target_h = int(original_h * scale)
    elif scale > max_scale:
        scale = max_scale
        target_w = int(original_w * scale)
        target_h = int(original_h * scale)
    
    return target_w, target_h, scale


if __name__ == "__main__":
    # 测试代码
    print("智能缩放模块测试")
    print("=" * 60)
    
    # 创建测试图像
    test_img = np.random.randint(0, 255, (2000, 1500, 3), dtype=np.uint8)
    
    # 测试下采样
    print("测试下采样: 2000x1500 → 400x300")
    result = smart_downscale(test_img, 400, 300)
    print(f"结果尺寸: {result.shape[1]}x{result.shape[0]}")
    
    # 测试上采样
    print("\n测试上采样: 400x300 → 1200x900")
    result2 = smart_upscale(result, 1200, 900)
    print(f"结果尺寸: {result2.shape[1]}x{result2.shape[0]}")
    
    # 测试最优尺寸计算
    print("\n测试最优尺寸计算")
    opt_w, opt_h, scale = calculate_optimal_size(2000, 1500, 500, 400)
    print(f"原始: 2000x1500, 画布: 500x400")
    print(f"最优: {opt_w}x{opt_h}, 缩放: {scale:.3f}")
    
    print("=" * 60)
