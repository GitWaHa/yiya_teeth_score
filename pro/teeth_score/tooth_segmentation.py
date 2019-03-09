#!/usr/bin/env python
# coding: utf-8

from model import *
from data import *

import os
import cv2
import random
from imutils import paths

random.seed(12345)


def image_get(img_path, target_size = (256,256), flag_multi_class = False, as_gray = True):
    img = cv2.imread(img_path, 0) # 读取为灰度图
    img = img / 255
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    img = np.reshape(img,(1,)+img.shape)
    return img


def showMarkers(image, _marker):
    # 创建彩色可视化标记图
    markers_show = np.zeros((_marker.shape[0], _marker.shape[1], 3), dtype=np.uint8)
    # 使用彩色色填充标记区域
    markers_show[_marker == 255] = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
    # markers_show[_marker == 0] = (255,255,255)
    # 标记显示到原图上
    img_mix = cv2.addWeighted(image.copy(), 0.5, markers_show, 0.5, 1)
    # cv2.imshow('markers_show',markers_show)
    cv2.imshow('img_mix',img_mix)
    cv2.moveWindow('img_mix',700,300)
    # 返回标记彩色可视化图片
    return img_mix, markers_show



# test your model and save predicted results
print("\n[INFO] test unet\n")

# ===================================测试图片列表=====================================
# test_paths = sorted(glob.glob(os.path.join("data/membrane/test",'*.png')))
# # 测试图片数量
# imagenum_test = len(test_paths)
# # 测试图片生成器
# testGene = testGenerator(test_paths)
# model = unet()
# model.load_weights("unet_membrane.hdf5")
# results = model.predict_generator(testGene,imagenum_test,verbose=1)
# saveResult("data/membrane/test_output", test_paths, results)

# ===================================测试单张图片======================================
image_path = 'image/src_0114_2028_22_0.png'
# 原始RGB图片
src_image = cv2.imread(image_path)
# 网络的输入图片
predict_image = image_get(image_path)
model = unet()
model.load_weights("unet_membrane.hdf5")
results = model.predict(predict_image) # shape: (1, 256, 256, 1)
# 网络输出图片
predicted_image = results[0,:,:,0]
print("\n[INFO] Predict Done\n")
# 输出图片二值化,得标记
ret, mark = cv2.threshold(predicted_image.copy(), 0.2, 255, cv2.THRESH_BINARY)
img_mix, markers_show = showMarkers(src_image, mark)
# 结果显示
cv2.imshow('predict',predicted_image)
# cv2.imshow('mark',mark)
cv2.waitKey(0)

