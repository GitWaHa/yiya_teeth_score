# coding: utf-8
import cv2
import numpy as np
import os
import datetime
from imutils import paths


imagePaths = sorted(list(paths.list_images('original_image')))
print (imagePaths)
# for imagePath in imagePaths:
#     print (imagePath)
outputPath = './labeled_data/'

# print(datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
time_now = datetime.datetime.now().strftime("%m%d_%H%M_%S")
print ('time:',time_now)

image = None
mask = None
mode = 0 # 标注模式
label_flag = 0 # 标注动作标志
PointStart = [0,0] # 起点坐标
PointEnd = [0,0] # 终点坐标
image_roi = None # 感兴趣区域
image_roi_bak = None # 感兴趣区域备份
prev_pt = None # 上一个点
save_cnt = -1 # 保存图片数
raw_img_num = 0 # 当前读取的原始图


# # 连续标记
# def draw_label(event,x,y,flags,param):
#     global prev_pt, label_flag
#     pt = (x, y)
#     prev_pt = pt
#     if event == cv2.EVENT_LBUTTONDOWN:
#         label_flag = 1 # 左键按下
#     elif event == cv2.EVENT_LBUTTONUP:
#         label_flag = 0 # 左键放开
#         # prev_pt = None
#     if label_flag == 1:
#         cv2.line(image_roi, prev_pt, pt, (255,255,255), 2)
#         cv2.line(mask, prev_pt, pt, (255,255,255), 2)
#     cv2.imshow('roi_resize', image_roi)
#     cv2.imshow('mask', mask)


# 区域标记
def draw_label(event,x,y,flags,param):
    global prev_pt, label_flag, mask
    pt = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        label_flag = 1
        prev_pt = pt
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     label_flag = 0
    elif event == cv2.EVENT_LBUTTONUP and label_flag==1:
        label_flag = 0
    elif event==cv2.EVENT_MOUSEMOVE and label_flag==1:
        cv2.line(image_roi, prev_pt, pt, (255,255,255), 2)
        cv2.line(mask, prev_pt, pt, (255,255,255), 2)
        cv2.imshow('roi_resize', image_roi)
        cv2.imshow('mask', mask)


# 用法: cv2.setMouseCallback('src_image', draw_label)
def get_roi(event,x,y,flags,param):
    global label_flag, image_roi, image_roi_bak, mask
    if event==cv2.EVENT_LBUTTONDOWN:
        label_flag = 1 # 左键按下
        PointStart[0],PointStart[1] = x, y # 记录起点位置
    elif event==cv2.EVENT_LBUTTONUP and label_flag==1: # 左键按下后检测弹起
        label_flag = 2 # 左键弹起
        PointEnd[0],PointEnd[1] = x, y # 记录终点位置
        PointEnd[1] = PointStart[1]+(PointEnd[0]-PointStart[0]) # 形成正方形
        # 提取ROI
        if PointEnd[0] != PointStart[0] and PointEnd[1] != PointStart[1]: # 框出了矩形区域,而非点
            print ("SPoint =", (PointStart[0], PointStart[1]))
            print ("EPoint =", (PointEnd[0],PointEnd[1]), '\n')
            # 获取矩形框左上角以及右下角点坐标
            PointLU = [0,0] # 左上角点
            PointRD = [0,0] # 右下角点
            # 左上角点xy坐标值均较小
            PointLU[0] = min(PointStart[0], PointEnd[0])
            PointLU[1] = min(PointStart[1], PointEnd[1])
            # 右下角点xy坐标值均较大
            PointRD[0] = max(PointStart[0], PointEnd[0])
            PointRD[1] = max(PointStart[1], PointEnd[1])
            # 提取ROI
            image_roi = image[PointLU[1]:PointRD[1],PointLU[0]:PointRD[0]] # 先y再x
            # cv2.imshow('roi', image_roi)
            image_roi = cv2.resize(image_roi, (256, 256))
            image_roi_bak = image_roi.copy()
            cv2.imshow('roi_resize', image_roi)
            cv2.setMouseCallback('roi_resize', draw_label)
            # 清空mask
            mask = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.imshow('mask', mask)

    elif event==cv2.EVENT_MOUSEMOVE and label_flag==1: # 左键按下后获取当前坐标, 并更新标注框
        PointEnd[0],PointEnd[1] = x, y # 记录当前位置
        PointEnd[1] = PointStart[1]+(PointEnd[0]-PointStart[0]) # 形成正方形
        image_copy=image.copy()
        cv2.rectangle(image_copy, (PointStart[0], PointStart[1]), (PointEnd[0], PointEnd[1]), (0, 255, 0), 1) # 根据x坐标画正方形
        cv2.imshow('image', image_copy)


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

    mask = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imshow('mask', mask)
    cv2.moveWindow('mask',1300,350)

    roi_resize = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.imshow('roi_resize', roi_resize)
    cv2.moveWindow('roi_resize',900,350)

    return imagePaths[img_num]




# 读取并显示图片
read_img(raw_img_num)

while(1):
    key = cv2.waitKey(50)
    # print (key)
    if key == 27: # ESC退出
        break
    # 保存图片
    if key == ord('s'):
        print ('\n[INFO] Save image time ' + datetime.datetime.now().strftime("%H:%M:%S"))
        save_cnt += 1
        if mode == 0:
            cv2.imwrite(outputPath + 'src_' + time_now + '_' + str(save_cnt) + '.png', image_roi_bak)
            cv2.imwrite(outputPath + 'mask_' + time_now + '_' + str(save_cnt) + '.png', mask)
            cv2.imwrite(outputPath + 'gray_' + time_now + '_' + str(save_cnt) + '.png', cv2.cvtColor(image_roi_bak, cv2.COLOR_BGR2GRAY))
        elif mode == 1:
            cv2.imwrite('./test_data/' + 'test_' + time_now + '_' + str(save_cnt) + '.png', cv2.cvtColor(image_roi_bak, cv2.COLOR_BGR2GRAY))
            cv2.imwrite('./test_data/' + 'src_' + time_now + '_' + str(save_cnt) + '.png', image_roi_bak)
    # 清除标记
    if key == ord('r'):
        print ('[INFO] Back roi')
        image_roi= image_roi_bak.copy()
        mask = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.imshow('roi_resize', image_roi)
        cv2.imshow('mask', mask)
    # if key == 32: # 空格
    # 下一张图片
    if key == ord('d'):
        raw_img_num += 1
        print('img_num:', raw_img_num)
        img_name = read_img(raw_img_num)
        print ('[INFO] The %dth image selected:'%(raw_img_num+1), img_name)
    if key == ord('a'):
        raw_img_num -= 1
        if raw_img_num < 0:
            raw_img_num = 0
        print('img_num:', raw_img_num)
        img_name = read_img(raw_img_num)
        print ('[INFO] The %dth image selected:'%(raw_img_num+1), img_name)
    # 切换模式
    if key == ord('w'):
        if mode == 0:
            mode = 1
            print ('[INFO] mode: generate test image')
        elif mode == 1:
            mode = 0
            print ('[INFO] mode: generate every image')


