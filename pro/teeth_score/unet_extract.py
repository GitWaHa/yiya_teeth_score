#!/usr/bin/python3.6
# coding=utf-8
#
# 龋齿识别,针对术中图片

from model import *
from data import *

import os
import cv2
import labelxml
from imutils import paths


def image_proc(img, target_size = (256,256)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img


def unet_extract_fillteeth(roi_image):
    model = unet()
    model.load_weights("unet_membrane.hdf5")

    # 获得网络的输入图片
    pre_image = image_proc(roi_image)
    # 调用模型进行分割
    results = model.predict(pre_image)  # shape: (1, 256, 256, 1)
    # 网络输出图片(float型,范围0-1)
    predicted_image = results[0, :, :, 0]

    print("\n[INFO] Predict Done")

    # 输出图片二值化,得标记(二值化后仍为float型)
    ret, mark = cv2.threshold(predicted_image, 0.85, 255, cv2.THRESH_BINARY)
    # 将float型转换为uint8型
    mark_uint8 = np.array(mark).astype(np.uint8)

    # 标记滤波,查找最大外轮廓
    _, contours, hierarchy = cv2.findContours(mark_uint8.copy(),
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(mark_uint8, [maxcnt], -1, 255, -1)  # 填充最大轮廓
        cv2.drawContours(roi_image, [maxcnt], -1, (0, 0, 255), 1)  # 在源图中显示最大轮廓
        # 创建滤波后的标记模版
        mark_filted = np.zeros(mark.shape[0:2], dtype=np.uint8)
        # 在模版上画出最大轮廓(即滤除外部小轮廓,并填充内部小轮廓)
        cv2.drawContours(mark_filted, [maxcnt], -1, 255, -1)

    return mark_filted
