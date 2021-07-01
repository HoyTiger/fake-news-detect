import os
import math
import codecs
import numpy as np
from PIL import Image, ImageEnhance
from src.config import train_parameters


def resize_img(img, target_size):
    """
    强制缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


def random_crop(img, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((train_parameters['input_size'][1], train_parameters['input_size'][2]), Image.BILINEAR)
    return img


def rotate_image(img):
    """
    图像增强，增加随机旋转角度
    """
    angle = np.random.randint(-14, 15)
    img = img.rotate(angle)
    return img


def random_brightness(img):
    """
    图像增强，亮度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    """
    图像增强，对比度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    """
    图像增强，饱和度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    """
    图像增强，色度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['hue_prob']:
        hue_delta = train_parameters['image_enhance_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_color(img):
    """
    概率的图像增强
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.35:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    elif prob < 0.7:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def custom_image_reader(piclists):
    """
    自定义用户图片读取器，先初始化图片种类，数量
    :param file_list:
    :param data_dir:
    :param mode:
    :return:
    """
    data = []
    temp = np.zeros(train_parameters['input_size']).astype('float32')
    for piclist in piclists:
        if piclist == '-1':
            data.append(temp)
        else:
            picName = piclist.split('\t')[0]
            path = os.path.join("dataset/truth_pic", picName)
            if not os.path.exists(path):
                path = os.path.join("dataset/rumor_pic/", picName)
        
            img = Image.open(path)

            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if train_parameters['image_enhance_strategy']['need_distort']:
                    img = distort_color(img)
                if train_parameters['image_enhance_strategy']['need_rotate']:
                    img = rotate_image(img)
                if train_parameters['image_enhance_strategy']['need_crop']:
                    img = random_crop(img, train_parameters['input_size'])
                if train_parameters['image_enhance_strategy']['need_flip']:
                    mirror = int(np.random.uniform(0, 2))
                    if mirror == 1:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # HWC--->CHW && normalized
                img = np.array(img).astype('float32')
                img -= train_parameters['mean_rgb']
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img /= 255.0                 # 像素值归一化
                data.append(img)
            except Exception as e:
                data.append(temp)
    return data
