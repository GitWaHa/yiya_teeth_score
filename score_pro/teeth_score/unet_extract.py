#!/usr/bin/python3.6
# coding=utf-8
#
# 龋齿识别,针对术中图片

from teeth_score.U_net.code_python.model import *

import os
import cv2
from imutils import paths

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TARGET_INPUT_SIZE = (128, 128)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (1, )
# PATH_MODEL_HDF5 = "./U_net/model_hdf5/unet_128.hdf5"
PATH_MODEL_HDF5 = "D:/WorkingFolder/Git/teeth_pro/score_pro/teeth_score/U_net/model_hdf5/unet_128.hdf5"

TARGET_FILL_INPUT_SIZE = (128, 128)
MODEL_FILL_INPUT_SIZE = TARGET_FILL_INPUT_SIZE + (3, )
# PATH_MODEL_HDF5 = "./U_net/model_hdf5/unet_128.hdf5"
PATH_MODEL_FILL_HDF5 = "D:/WorkingFolder/Git/teeth_pro/score_pro/teeth_score/U_net/model_hdf5/unet_128_fill.hdf5"


def image_proc(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1, ))
    img = np.reshape(img, (1, ) + img.shape)
    return img


def image_fillarea_proc(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    # cv2.imshow('rgb', img)
    img = img / 255
    img = np.reshape(img, (1, ) + img.shape)
    return img


model = unet(input_size=MODEL_INPUT_SIZE)
model.load_weights(PATH_MODEL_HDF5)
model_fill = unet(input_size=MODEL_FILL_INPUT_SIZE)
model_fill.load_weights(PATH_MODEL_FILL_HDF5)


# print("unet模型已加载")
def unet_extract_fillteeth(roi_image):
    # 获得网络的输入图片
    pre_image = image_proc(roi_image)
    # 调用模型进行分割
    results = model.predict(pre_image)  # shape: (1, 256, 256, 1)
    # 网络输出图片(float型,范围0-1)
    predicted_image = results[0, :, :, 0]
    # print("\n[INFO] Predict Done")

    mark_uint8 = np.zeros((128, 128), dtype=np.uint8)
    mark_uint8[predicted_image > 0.8] = 255

    # 标记滤波,查找最大外轮廓
    _, contours, hierarchy = cv2.findContours(mark_uint8.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    # 创建滤波后的标记模版
    mark_filted = np.zeros(mark_uint8.shape[0:2], dtype=np.uint8)
    if contours:
        maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(mark_uint8, [maxcnt], -1, 255, -1)
        # 在模版上画出最大轮廓(即滤除外部小轮廓,并填充内部小轮廓)
        cv2.drawContours(mark_filted, [maxcnt], -1, 255, -1)

    return mark_filted


def unet_extract_fillarea(roi_image):
    # 获得网络的输入图片
    pre_image = image_fillarea_proc(roi_image)
    # 调用模型进行分割
    results = model_fill.predict(pre_image)  # shape: (1, 256, 256, 1)
    # 网络输出图片(float型,范围0-1)
    predicted_image = results[0, :, :, 0]

    mark_uint8 = np.zeros((128, 128), dtype=np.uint8)
    mark_uint8[predicted_image > 0.6] = 255
    # cv2.imshow('mark_p', mark_uint8)
    return mark_uint8
