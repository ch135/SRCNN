import scipy.misc
import scipy.ndimage
import numpy as np
import h5py
import tensorflow as tf
import os
import glob
"""
:Author: 陈浩
:Date: 2018/11/23
:Content: 图像超分辨率处理工具包
"""

FLAGS = tf.app.flags.FLAGS


def input_setup(sess, config):
    """
    :param sess:
    :param config:
    :return:
    :Content: 读取图像；切分图像；把数据保存到h5文件中
    """
    if config.is_train:
        data = pre_imdata(sess, dataset="Train")
    else:
        data = pre_imdata(sess, dataset="Test")

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size)/2

    if config.is_train:
        for i in range(len(data)):
            input_, label_ = preprocess(data[i], config.scale)
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape

            for x in range(0, h-config.image_size+1, config.stride):
                for y in range(0, w-config.image_size+1, config.stride):
                    sub_input = input_[x: x+config.image_size, y:y+config.image_size]  # [33*33]
                    sub_label = label_[x+int(padding):x+int(padding)+config.label_size,
                                         y+int(padding):y+int(padding)+config.label_size]   # [21*21]

                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)
    else:
        input_, label_ = preprocess(data[8], config.scale)
        if len(input_) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        # nx ny 用于合成测试好的图像块
        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x: x + config.image_size, y:y + config.image_size]  # [33*33]
                sub_label = label_[x + int(padding):x + int(padding) + config.label_size,
                            y + int(padding):y + int(padding) + config.label_size]  # [21*21]

                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    arrdate = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence)   # [?, 21, 21, 1]
    save_data(sess, arrdate, arrlabel)

    if not config.is_train:
        return nx, ny


def pre_imdata(sess, dataset):
    """
    :param sess:
    :param dataset: 训练集或测试集的图像名字
    :return: 图像路径集合
    """
    if FLAGS.is_train:
        data_dir = os.path.join(os.getcwd(), dataset)   # os.getcwd() 获取当前绝对路径
    else:
        data_dir = os.path.join(os.sep, os.path.join(os.getcwd(), dataset), "Set14")  # os.sep 适配不同系统的路径符号
    data = glob.glob(os.path.join(data_dir, "*.bmp"))   # glob.glob() 查询对应路径的数据
    return data


def preprocess(path, scale=3):
    """
    :param path: 图像路径
    :param scale: 放大因子
    :return: 双三线性插值处理后的图像 + 高频信息
    :Content: 图像预处理
        （1）以 YCbCr 形式读取图像
        （2）图像正则化
        （3）图像 bicubic interpolating 处理
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)
    # normalized
    image = image / 255
    label_ = label_ / 255

    """
    Bilinear interpolation would be order=1, 
    nearest is order=0, 
    and cubic is the default (order=3).
    """
    input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale))
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.))

    return input_, label_


def save_data(sess, data, labels):
    """
    :param sess:
    :param data: 图像数据
    :param labels: 图像高频信息
    :Content 保存训练过程中的图像数据
    """
    if FLAGS.is_train:
        path = os.path.join(os.getcwd(), "checkpoint/train.h5")
    else:
        path = os.path.join(os.getcwd(), "checkpoint/test.h5")
    with h5py.File(path, "w") as hf:
        hf.create_dataset("data", data=data)
        hf.create_dataset("label", data=labels)


def read_data(path):
    """
    :param path:
    :return: 训练过程中保存的图像数据 + 高频信息
    :Content: 读取存储在 .h5 中的图像数据
    """
    with h5py.File(path, "r") as hf:
        data = np.array(hf.get("data"))
        label_ = np.array(hf.get("label"))
        return data, label_


def merge(images, size):
    """
    :param images:
    :param size:
    :return: 合并后的图像
    :Content: 图像块合并
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1],1))
    for index, image in enumerate(images):
        i = index % size[1]
        j = index // size[1]
        img[h*j:h*j+h, w*i:w*i+w, :] = image
    return img


def imread(path, is_grayscale=True):
    """
    :Content: 读取图像
    :return: YCbCr 图像
    """
    if is_grayscale:
        return scipy.misc.imread(path, mode="YCbCr", flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode="YCbCr").astype(np.float)


def imsave(path, image):
    """
    :param path:
    :param image:
    :return:
    :Content: 保存图像
    """
    return scipy.misc.imsave(path, image)


def modcrop(image, scale=3):
    """
    :param image:
    :param scale: 扩展因子
    :return:缩小的图像
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        label_ = image[:h, :w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        label_ = image[:h, :w]
    return label_

