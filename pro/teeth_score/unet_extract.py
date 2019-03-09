# !/usr/bin/env python
# coding: utf-8
#
# 龋齿识别,针对术中图片

from model import *
from data import *

import os
import cv2
import random
import labelxml
from imutils import paths

random.seed(12345)

curr_img_num = 0
image = None
mode = 0 # 标注模式
label_flag = 0 # 标注动作标志
PointStart = [0,0] # 起点坐标
PointEnd = [0,0] # 终点坐标
image_roi = None # 感兴趣区域
predicted_image = None # 模型输出的预测图
move_x = 800 # 图像显示平移
image_roi_gray = None



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
    cv2.moveWindow('img_mix',320+move_x,350)
    # 返回标记彩色可视化图片
    return img_mix, markers_show


def get_pix(event,x,y,flags,param):
    global predicted_image
    if event==cv2.EVENT_LBUTTONDOWN:
        print ((x,y),predicted_image[y][x])


# 打印坐标
def get_pix_roi_gray(event, x, y, flags, param):
    global image_roi_gray
    if event == cv2.EVENT_LBUTTONDOWN:
        print((x, y), image_roi_gray[y][x])


def read_img(img_num=0):
    global image
    image = cv2.imread(imagePaths[img_num]) #读取图像
    # print (image.shape[0],image.shape[1])

    # 缩放到屏幕可显示范围内
    if (image.shape[0] > 1000):
        zoom = 1000/image.shape[0]
        image = cv2.resize(image, (0, 0), fx=zoom, fy=zoom,
                                   interpolation=cv2.INTER_NEAREST)
    cv2.imshow("image", image)
    cv2.moveWindow('image',0,0)
    cv2.setMouseCallback('image', get_roi)

    return imagePaths[img_num]


# 利用HSV的H通道获取牙齿区域掩码
def get_tooth_region(image):
    # global soft_marker
    (height,width) = image.shape[:2]
    # ================================= HSV空间滤波 ========================================
    HSV = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    # cv2.imshow('HSV', HSV)
    # cv2.imshow('H', HSV[..., 0])

    HSV_H = HSV[..., 0]
    marker = np.zeros((HSV_H.shape[0], HSV_H.shape[1]), dtype=np.uint8)
    # 11,12,13,14,15
    for r in range(height):
        for c in range(width):
            if HSV_H[r][c] > 8 and HSV_H[r][c] < 25:
                marker[r][c] = 255
            else:
                marker[r][c] = 0
    cv2.imshow("marker", marker)

    # # 外部轮廓查找,滤除标记图中内部为白点的小块区域
    # _, contours, hierarchy = cv2.findContours(
    # marker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if contours:
    #     for i in contours:
    #         # cv2.drawContours(image, [i], -1, (0,0,255), 1) # 轮廓可视化
    #         if cv2.contourArea(i) < 200:
    #             # 填充面积小的轮廓
    #             cv2.drawContours(marker, [i], -1, 0, -1)
    # cv2.imshow("marker_filted", marker)
    # # cv2.imshow("image_contour", image)

    return marker


# 关键要素判断(AA1)
def key_elements_judge(roi_img, PointLU, PointRD, src_img_shape):
    left_tooth_point_num = 0
    right_tooth_point_num = 0
    key_elements_score = 7 # 关键要素得分

    # print (PointLU,PointRD)
    roi_width = PointRD[0] - PointLU[0]
    tooth_region = get_tooth_region(roi_img)
    (height,width) = tooth_region.shape[:2]
    (src_height,src_width) = src_img_shape[:2]
    # 统计第一列和最后一列白点数
    for r in range(height):
        if tooth_region[r][0] == 255:
            left_tooth_point_num += 1
        if tooth_region[r][width-1] == 255:
            right_tooth_point_num += 1

    # print (left_tooth_point_num, right_tooth_point_num)

    # 参照牙存在情况:1-有牙 2-不完整 3-最里侧
    left_tooth_status = 0
    right_tooth_status = 0
    # 认为一侧白点超过一定数量,并且原始图像感兴趣区域左侧/右侧还有较大空间,则这一侧有牙
    if left_tooth_point_num > height/4:
        if PointLU[0] > roi_width/2:
            left_tooth_status = 1
            print('[INFO] 左侧有牙')
        else:
            left_tooth_status = 2
            print('[INFO] 左侧牙不完整,扣3分')
            key_elements_score -= 3 # 左侧无参照,扣3分
    else:
        left_tooth_status = 3
        print('[INFO] 最左侧牙')

    if right_tooth_point_num > height/4:
        if src_width-PointRD[0] > roi_width/2:
            right_tooth_status = 1
            print('[INFO] 右侧有牙')
        else:
            right_tooth_status = 2
            print('[INFO] 右侧牙不完整,扣3分')
            key_elements_score -= 3 # 右侧无参照,扣3分
    else:
        right_tooth_status = 3
        print('[INFO] 最右侧牙')

    if left_tooth_status==3:
        if right_tooth_status == 1:
            print('[INFO] 最左侧牙有右侧参照牙')
        else:
            key_elements_score -= 3 # 最左侧牙无右侧参照牙,再扣3分
            print('[INFO] 最左侧牙无右侧参照牙,再扣3分')

    if right_tooth_status==3:
        if left_tooth_status == 1:
            print('[INFO] 最右侧牙有左侧参照牙')
        else:
            key_elements_score -= 3 # 最右侧牙无左侧参照牙,再扣3分
            print('[INFO] 最右侧牙无左侧参照牙,再扣3分')

    print('[INFO] AA1参照牙有无得分(共7分):', key_elements_score,'\n')


# 判别龋齿
def caries_judge(img,mark):
    caries_point_num = 0        # 黑点个数
    caries_point_thresh = 95    # 灰度阈值
    black_level_score = 10      # 黑色深浅得分
    black_num_score = 10        # 黑色大小得分
    min_gray_value = 255        # 牙齿区域最小灰度值

    (height,width) = img.shape[0:2]
    # 黑色显示模版
    black_point_show = np.zeros((height,width), dtype = np.uint8)
    for r in range(height):
        for c in range(width):
            if img[r][c] < caries_point_thresh and mark[r][c] == 255: # 分割后的牙齿区域中灰度值小于阈值的像素
                # print((r,c),img[r][c])
                black_point_show[r][c] = 255
                caries_point_num += 1
                if  img[r][c] < min_gray_value: # 获取最低灰度值
                    min_gray_value = img[r][c]
    # 可视化龋齿黑点
    cv2.imshow('black_point_show', black_point_show)

    # 根据黑点数量计算黑色大小得分
    print ('caries_point_num:',caries_point_num)
    black_num_score -= (caries_point_num//100) # 每100个黑点减一分,需根据输入图片大小调整
    black_num_score = 0 if black_num_score <= 0 else black_num_score
    print('\n[INFO] BB1黑色大小得分(共10分):', black_num_score)
    # 根据最低灰度值计算黑色深浅得分
    print ('min_gray_value', min_gray_value)
    if min_gray_value < 255: # 即有灰度值低于阈值的像素点,最小值更新过
        black_level_score -= ((caries_point_thresh-min_gray_value)//5) # 每低于阈值5灰度值扣一分
        black_level_score = 0 if black_level_score <= 0 else black_level_score
    print('[INFO] BB1黑色深浅得分(共10分):', black_level_score)
    print('[INFO] BB1项目最终得分(共20分):', black_num_score+black_level_score,'\n')


def get_roi(event, x, y, flags, param):
    global label_flag, image_roi, image_roi_gray
    if event == cv2.EVENT_LBUTTONDOWN:
        label_flag = 1  # 左键按下
        PointStart[0],PointStart[1] = x, y  # 记录起点位置
    elif event == cv2.EVENT_LBUTTONUP and label_flag == 1:  # 左键按下后检测弹起
        label_flag = 2  # 左键弹起
        PointEnd[0], PointEnd[1] = x, y  # 记录终点位置
        PointEnd[1] = PointStart[1]+(PointEnd[0]-PointStart[0])  # 形成正方形
        # 提取ROI
        if PointEnd[0] != PointStart[0] and PointEnd[1] != PointStart[1]:  # 框出了矩形区域,而非点
            print ("\nSPoint =", (PointStart[0], PointStart[1]))
            print ("EPoint =", (PointEnd[0], PointEnd[1]))
            # 获取矩形框左上角以及右下角点坐标
            PointLU = [0,0] # 左上角点
            PointRD = [0,0] # 右下角点
            # 左上角点xy坐标值均较小
            PointLU[0] = min(PointStart[0], PointEnd[0])
            PointLU[1] = min(PointStart[1], PointEnd[1])
            # 右下角点xy坐标值均较大
            PointRD[0] = max(PointStart[0], PointEnd[0])
            PointRD[1] = max(PointStart[1], PointEnd[1])
            # roi宽度
            roi_width = PointRD[0] - PointLU[0]
            print ("Width = %d"%roi_width, '\n')
            # 保存标注信息
            spliname = os.path.splitext(os.path.basename(imagePaths[curr_img_num]))[0]
            print (spliname)
            labelxml.create_xml(xml_name=('./labelxml/'+spliname+'.xml'),
                  path=(imagePaths[curr_img_num]), width=str(roi_width), height=str(roi_width),
                  xmin=str(PointLU[0]), ymin=str(PointLU[1]),
                  xmax=str(PointRD[0]), ymax=str(PointRD[1]))

            # 提取ROI
            image_roi = image[PointLU[1]:PointRD[1], PointLU[0]:PointRD[0]]  # 先y再x
            # cv2.imshow('roi', image_roi)
            image_roi = cv2.resize(image_roi, (256, 256))
            image_roi_raw = image_roi.copy()            # 后面要在roi上画轮廓,此处保存原始roi
            cv2.imshow('roi_resize', image_roi)
            image_roi_gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            cv2.imshow('image_roi_gray', image_roi_gray)
            cv2.setMouseCallback('image_roi_gray', get_pix_roi_gray)

            # 感兴趣区域输入unet,进行牙齿分割
            mark_filted, img_roi_mix = tooth_seg(image_roi, model)
            # 龋齿判断
            caries_judge(image_roi_gray, mark_filted)
            # 关键要素有无(AA1)
            key_elements_judge(image_roi_raw, PointLU, PointRD, image.shape)

            # 将分割后添加了轮廓即区域标记的roi放回原图
            # img_roi_mix_resize = cv2.resize(img_roi_mix, (roi_width, roi_width))
            # image[PointLU[1]:PointRD[1],PointLU[0]:PointRD[0]] = img_roi_mix_resize
            # cv2.imshow('image_roi_mix', image)
            # 将分割后添加了轮廓标记的roi放回原图
            img_roi_mix_resize = cv2.resize(image_roi, (roi_width, roi_width))
            image_copy = image.copy()
            image_copy[PointLU[1]:PointRD[1], PointLU[0]:PointRD[0]] = img_roi_mix_resize
            # 更新待分割图片
            cv2.rectangle(image_copy, (PointStart[0], PointStart[1]), (PointEnd[0], PointEnd[1]), (0, 255, 0), 1)
            cv2.imshow('image', image_copy)

    elif event == cv2.EVENT_MOUSEMOVE and label_flag == 1:          # 左键按下后获取当前坐标, 并更新标注框
        PointEnd[0], PointEnd[1] = x, y                             # 记录当前位置
        PointEnd[1] = PointStart[1]+(PointEnd[0]-PointStart[0])      # 形成正方形
        image_copy = image.copy()
        cv2.rectangle(image_copy, (PointStart[0], PointStart[1]), (PointEnd[0], PointEnd[1]), (0, 255, 0), 1) # 根据x坐标画正方形
        cv2.imshow('image', image_copy)


def image_proc(img, target_size = (256,256)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape+(1,))
    img = np.reshape(img, (1,)+img.shape)
    return img


def tooth_seg(src_image, model):
    global predicted_image
    # 获得网络的输入图片
    predict_image = image_proc(src_image)
    # 调用模型进行分割
    results = model.predict(predict_image)  # shape: (1, 256, 256, 1)
    # 网络输出图片(float型,范围0-1)
    predicted_image = results[0, :, :, 0]
    print("\n[INFO] Predict Done")
    # 输出图片二值化,得标记(二值化后仍为float型)
    ret, mark = cv2.threshold(predicted_image, 0.85, 255, cv2.THRESH_BINARY)
    # 将float型转换为uint8型
    mark_uint8 = np.array(mark).astype(np.uint8)
    cv2.imshow("mark_uint8", mark_uint8)
    cv2.moveWindow('mark_uint8', 320+move_x, 0)
    # 标记滤波,查找最大外轮廓
    # print (mark_uint8.shape,mark_uint8.dtype)
    _, contours, hierarchy = cv2.findContours(mark_uint8.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    cv2.moveWindow('mark_filted',640+move_x,0)
    cv2.imshow("src_image", src_image)
    cv2.moveWindow('src_image', 0+move_x, 350)
    # 标记显示到原图上
    img_roi_mix, markers_show = showMarkers(src_image, mark_filted)
    # 结果显示
    cv2.imshow('predict', predicted_image)
    cv2.setMouseCallback('predict', get_pix)
    cv2.moveWindow('predict', 0 + move_x, 0)

    return mark_filted,img_roi_mix


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

    # cv2.imshow("mark_filted", mark_filted)
    # cv2.moveWindow('mark_filted', 640 + move_x, 0)

    return mark_filted


# ============================================ main =============================================
# print("\n[INFO] tooth segmentation\n")
# # 载入模型
# model = unet()
# model.load_weights("unet_membrane.hdf5")
# # imagePaths = sorted(list(paths.list_images('data_generator/raw_data1'))) # 包括子文件夹
# imagePaths = sorted(glob.glob(os.path.join("/home/waha/Desktop/pro_teeth/JPG_TEST/患者4/", '*.jpg'))) # 不包括子文件夹
# # 读取第一张待分割图片
# read_img(curr_img_num)
#
# while(1):
#     key = cv2.waitKey(0)
#     # 下一张图片
#     if key == ord('d'):
#         curr_img_num += 1
#         cv2.destroyAllWindows()
#         read_img(curr_img_num)
#         print ('\n[INFO] The %dth image selected:'%(curr_img_num+1), imagePaths[curr_img_num])
#     # 上一张图片
#     if key == ord('a'):
#         curr_img_num -= 1
#         if curr_img_num < 0:
#             curr_img_num = 0
#         cv2.destroyAllWindows()
#         read_img(curr_img_num)
#         print ('\n[INFO] The %dth image selected:'%(curr_img_num+1), imagePaths[curr_img_num])
#     if key == 27: # ESC退出
#         break
# print("end of main")
