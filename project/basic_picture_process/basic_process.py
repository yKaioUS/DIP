import cv2
import numpy as np
import os

def rotate_image(img, angle):
    """
    图片旋转
    :param img: 输入图像
    :param angle: 旋转角度（顺时针）
    :return: 旋转后的图像
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def resize_image(img, scale_percent):
    """
    图片缩放
    :param img: 输入图像
    :param scale_percent: 缩放百分比（0-100）
    :return: 缩放后的图像
    """
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def adjust_contrast(img, alpha):
    """
    调整对比度
    :param img: 输入图像
    :param alpha: 对比度系数（1.0-3.0）
    :return: 调整后的图像
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def adjust_brightness(img, beta):
    """
    调整亮度
    :param img: 输入图像
    :param beta: 亮度增量（-100到100）
    :return: 调整后的图像
    """
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)

def convert_grayscale(img):
    """
    灰度化
    :param img: 输入图像
    :return: 灰度图像
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def histogram_equalization(img):
    """
    直方图均衡化
    :param img: 输入图像
    :return: 均衡化后的图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 进行直方图均衡化
    equalized = cv2.equalizeHist(gray)
    # 转回BGR格式
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def gaussian_blur(img, kernel_size):
    """
    高斯模糊
    :param img: 输入图像
    :param kernel_size: 核大小（奇数）
    :return: 模糊后的图像
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def sharpen_image(img):
    """
    图像锐化
    :param img: 输入图像
    :return: 锐化后的图像
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def add_noise(img, noise_type='gaussian'):
    """
    添加噪声
    :param img: 输入图像
    :param noise_type: 噪声类型（gaussian/salt_pepper）
    :return: 带噪声的图像
    """
    if noise_type == 'gaussian':
        row,col,ch = img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        return cv2.add(img, gauss.astype(np.uint8))
    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(img)
        # Salt模式
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
        out[coords] = 255
        # Pepper模式
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
        out[coords] = 0
        return out

def edge_detection(img):
    """
    边缘检测（Canny）
    :param img: 输入图像
    :return: 边缘检测结果
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def morphological_operation(img, operation='erode', kernel_size=3):
    """
    形态学操作（腐蚀/膨胀）
    :param img: 输入图像
    :param operation: 操作类型（erode/dilate）
    :param kernel_size: 核大小
    :return: 处理后的图像
    """
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    if operation == 'erode':
        return cv2.erode(img, kernel, iterations=1)
    elif operation == 'dilate':
        return cv2.dilate(img, kernel, iterations=1)

def process_image(image_path, operation, **params):
    """图像处理总入口"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return None

        operations = {
            'rotate': rotate_image,
            'resize': resize_image,
            'contrast': adjust_contrast,
            'brightness': adjust_brightness,
            'grayscale': convert_grayscale,
            'histogram': histogram_equalization,
            'blur': gaussian_blur,
            'sharpen': sharpen_image,
            'noise': add_noise,
            'edge': edge_detection,
            'morph': morphological_operation
        }
        
        if operation not in operations:
            print(f"不支持的操作: {operation}")
            return None
        
        if operation == 'grayscale' or operation == 'edge':
            img = operations[operation](img)
        elif operation == 'morph':
            # 特殊处理形态学操作，因为operation参数名冲突
            morph_type = params.pop('operation', 'erode')  # 从params中取出operation参数
            img = operations[operation](img, operation=morph_type, **params)
        else:
            img = operations[operation](img, **params)
        
        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取项目根目录（basic_picture_process的上一级）
        root_dir = os.path.dirname(current_dir)
        # 构建result目录的路径
        result_dir = os.path.join(root_dir, 'result')
        
        print(f"保存图片到目录: {result_dir}")  # 添加调试信息
        
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        name = os.path.splitext(os.path.basename(image_path))[0]
        name = f"processed_{name}.jpg"
        output_path = os.path.join(result_dir, name)
        print(f"输出文件路径: {output_path}")  # 添加调试信息
        
        success = cv2.imwrite(output_path, img)

        if not success:
            print(f"保存图片失败: {output_path}")
            return None

        return f'result/{name}'
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return None
