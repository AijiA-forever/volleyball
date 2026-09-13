import cv2
import numpy as np


def gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    使用高斯模糊进行图像去噪
    :param image: 输入图像
    :param kernel_size: 高斯核大小
    :param sigma: 高斯核标准差
    :return: 去噪后的图像
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def median_filter(image, kernel_size=5):
    """
    使用中值滤波进行图像去噪
    :param image: 输入图像
    :param kernel_size: 滤波核大小
    :return: 去噪后的图像
    """
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    使用双边滤波进行图像去噪，保留边缘细节
    :param image: 输入图像
    :param d: 过滤期间使用的每个像素邻域的直径
    :param sigma_color: 颜色空间中的sigma值，较大的值意味着更远的颜色会混合在一起
    :param sigma_space: 坐标空间中的sigma值，较大的值意味着更远的像素会相互影响
    :return: 去噪后的图像
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def non_local_means_denoising(image, h=10, h_for_color_components=10, template_window_size=7, search_window_size=21):
    """
    使用非局部均值去噪方法
    :param image: 输入图像
    :param h: 控制过滤器强度的参数
    :param h_for_color_components: 与h类似，但仅用于彩色图像的每个颜色分量
    :param template_window_size: 用于计算权重的模板窗口大小
    :param search_window_size: 用于查找匹配区域的搜索窗口大小
    :return: 去噪后的图像
    """
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, h=h, hColor=h_for_color_components,
                                             templateWindowSize=template_window_size,
                                             searchWindowSize=search_window_size)
    else:
        return cv2.fastNlMeansDenoising(image, h=h,
                                       templateWindowSize=template_window_size,
                                       searchWindowSize=search_window_size)


def apply_denoising(image, denoising_method='gaussian', **kwargs):
    """
    应用指定的去噪方法
    :param image: 输入图像
    :param denoising_method: 去噪方法，可以是 'gaussian', 'median', 'bilateral', 'non_local_means'
    :param kwargs: 去噪方法的参数
    :return: 去噪后的图像
    """
    if denoising_method == 'gaussian':
        return gaussian_blur(image, **kwargs)
    elif denoising_method == 'median':
        return median_filter(image, **kwargs)
    elif denoising_method == 'bilateral':
        return bilateral_filter(image, **kwargs)
    elif denoising_method == 'non_local_means':
        return non_local_means_denoising(image, **kwargs)
    else:
        raise ValueError(f"不支持的去噪方法: {denoising_method}")


def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    调整图像的亮度和对比度
    :param image: 输入图像
    :param alpha: 对比度调整因子（1.0 不变，>1 增加对比度，<1 降低对比度）
    :param beta: 亮度调整值（0 不变，正值增加亮度，负值降低亮度）
    :return: 调整后的图像
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def sharpen(image, kernel_size=(3, 3), amount=1.0):
    """
    锐化图像
    :param image: 输入图像
    :param kernel_size: 锐化核大小
    :param amount: 锐化程度
    :return: 锐化后的图像
    """
    # 创建锐化核
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]]) * amount
    sharpened = cv2.filter2D(image, -1, kernel)
    return cv2.convertScaleAbs(sharpened)


def apply_enhancement(image, enhancement_method, **kwargs):
    """
    应用指定的图像增强方法
    :param image: 输入图像
    :param enhancement_method: 增强方法，可以是 'brightness_contrast', 'sharpen'
    :param kwargs: 增强方法的参数
    :return: 增强后的图像
    """
    if enhancement_method == 'brightness_contrast':
        return adjust_brightness_contrast(image, **kwargs)
    elif enhancement_method == 'sharpen':
        return sharpen(image, **kwargs)
    else:
        raise ValueError(f"不支持的增强方法: {enhancement_method}")


def preprocess_image(image, denoising_params=None, enhancement_params=None):
    """
    综合预处理图像：先去噪（可选），再增强（可选）
    :param image: 输入图像
    :param denoising_params: 去噪参数，格式为 {'method': 'gaussian', 'kernel_size': (5, 5), ...}
    :param enhancement_params: 增强参数，格式为 {'method': 'brightness_contrast', 'alpha': 1.2, 'beta': 10, ...}
    :return: 预处理后的图像
    """
    processed_image = image.copy()
    
    # 应用去噪
    if denoising_params is not None:
        method = denoising_params.pop('method', 'gaussian')
        processed_image = apply_denoising(processed_image, method, **denoising_params)
    
    # 应用增强
    if enhancement_params is not None:
        method = enhancement_params.pop('method', 'brightness_contrast')
        processed_image = apply_enhancement(processed_image, method, **enhancement_params)
    
    return processed_image