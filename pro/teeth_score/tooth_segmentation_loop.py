#!/usr/bin/env python
# coding: utf-8

from model import *
from data import *

import os
import cv2
import random
from imutils import paths

random.seed(12345)

curr_img_num = 0


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
    img_mix = cv2.addWeighted(image.copy(), 0.8, markers_show, 0.2, 1)
    # cv2.imshow('markers_show',markers_show)
    cv2.imshow('img_mix',img_mix)
    cv2.moveWindow('img_mix',340,350)
    # 返回标记彩色可视化图片
    return img_mix, markers_show


def get_pix(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print ((x,y),predicted_image[y][x])


def tooth_seg(image_path,model):
    src_image = cv2.imread(image_path)
    # 网络的输入图片
    predict_image = image_get(image_path)
    # 调用模型进行分割
    results = model.predict(predict_image) # shape: (1, 256, 256, 1)
    # 网络输出图片(float型,范围0-1)
    predicted_image = results[0,:,:,0]
    print("\n[INFO] Predict Done")
    # 输出图片二值化,得标记(二值化后仍为float型)
    ret, mark = cv2.threshold(predicted_image, 0.85, 255, cv2.THRESH_BINARY)
    # 将float型转换为uint8型
    mark_uint8 = np.array(mark).astype(np.uint8)
    cv2.imshow("mark_uint8", mark_uint8)
    cv2.moveWindow('mark_uint8',340,0)
    # 标记滤波,查找最大外轮廓
    # print (mark_uint8.shape,mark_uint8.dtype)
    _, contours, hierarchy = cv2.findContours(
    mark_uint8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
        cv2.drawContours(mark_uint8, [maxcnt], -1, 255, -1) # 填充最大轮廓
        cv2.drawContours(src_image, [maxcnt], -1, (0,0,255), 1) # 在源图中显示最大轮廓
        # for i in contours:
        #     cv2.drawContours(src_image, [i], -1, (0,0,255), -1) # 轮廓可视化
        # 创建滤波后的标记模版
        mark_filted = np.zeros(mark.shape[0:2], dtype=np.uint8)
        # 在模版上画出最大轮廓(即滤除外部小轮廓,并填充内部小轮廓)
        cv2.drawContours(mark_filted, [maxcnt], -1, 255, -1)

    cv2.imshow("mark_filted", mark_filted)
    cv2.moveWindow('mark_filted',620,0)
    cv2.imshow("src_image", src_image)
    cv2.moveWindow('src_image',0,350)
    # 标记显示到原图上
    img_mix, markers_show = showMarkers(src_image, mark_filted)
    # 结果显示
    cv2.imshow('predict',predicted_image)
    cv2.setMouseCallback('predict', get_pix)
    cv2.moveWindow('predict',0,0)



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

# =================================循环测试单张图片====================================
# 载入 模型
model = unet()
model.load_weights("unet_membrane.hdf5")
# imagePaths = sorted(list(paths.list_images('image'))) # 包括子文件夹
imagePaths = sorted(glob.glob(os.path.join("image",'*.png'))) # 不包括子文件夹

tooth_seg(imagePaths[curr_img_num],model)

while(1):
    key = cv2.waitKey(0)
    # 下一张图片
    if key == ord('d'):
        curr_img_num += 1
        tooth_seg(imagePaths[curr_img_num],model)
        print ('[INFO] The %dth image selected:'%(curr_img_num+1), imagePaths[curr_img_num])
    # 上一张图片
    if key == ord('a'):
        curr_img_num -= 1
        if curr_img_num < 0:
            curr_img_num = 0
        tooth_seg(imagePaths[curr_img_num],model)
        print ('[INFO] The %dth image selected:'%(curr_img_num+1), imagePaths[curr_img_num])
    if key == 27: # ESC退出
        break

