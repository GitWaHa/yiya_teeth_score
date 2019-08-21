#!/usr/bin/python3.6
# coding=utf-8

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os, re, time, copy, sys
from shutil import copyfile

from teeth_score.U_net.unet_extract import unet_extract_fillteeth, unet_extract_fillarea
from teeth_score.AlexNet.classify_teethtype import classify_teethtype
from teeth_score.Yolo3.yolo_rect import *

import teeth_score.config as myconfig
work_floder = myconfig.WORK_FLODER

TEETH_IMAGE_SET_ROW = 480
TEETH_IMAGE_SET_COL = 480
label_flag = 0
PointEnd = [0, 0]
PointStart = [0, 0]


class Img_info:
    def __init__(self):
        self.patient_name = 0
        self.operation_time = 0
        self.fillteeth_type = 0
        self.fillteeth_name = 0
        self.fillteeth_region = 0
        self.fillteeth_num = 0
        self.doctor_name = 0
        self.img_type = 0

        self.upload_time = 0
        self.imgfloder_path = 0

    def get_info(self, img_dir, use_deploy=1):
        if use_deploy == 0:
            pattern = r"(.*)-(.*)-(.*)-(.*)-(.*)\.(.*)"
            info = list(re.findall(pattern, img_dir)[0])
            self.upload_time = info[0]
            self.patient_name = info[1]
            self.operation_time = info[2]
            self.fillteeth_name = info[3]
            self.doctor_name = info[4]
            self.img_type = info[5]

            self.fillteeth_region = list(self.fillteeth_name)[0]
            self.fillteeth_num = list(self.fillteeth_name)[1]

            if self.fillteeth_num == '1' or self.fillteeth_num == '2' or self.fillteeth_num == '3':
                self.fillteeth_type = '门牙'
            else:
                self.fillteeth_type = '后牙'
        else:
            str_img_path = img_dir.split("/")
            img_name = str_img_path[len(str_img_path) - 1]
            self.imgfloder_path = '/'.join(str_img_path[0:len(str_img_path) - 2])
            print(self.imgfloder_path)
            pattern = r"(.*)-(.*)-(.*)-(.*)-(.*)\.(.*)"
            info = list(re.findall(pattern, img_name)[0])
            self.upload_time = info[0]
            self.patient_name = info[1]
            self.operation_time = info[2]
            self.fillteeth_name = info[3]
            self.doctor_name = info[4]
            self.img_type = info[5]

            self.fillteeth_region = list(self.fillteeth_name)[0]
            self.fillteeth_num = list(self.fillteeth_name)[1]

            if self.fillteeth_num == '1' or self.fillteeth_num == '2' or self.fillteeth_num == '3':
                self.fillteeth_type = '门牙'
            else:
                self.fillteeth_type = '后牙'

        return

    def print_info(self):
        print("患者姓名：", self.patient_name)
        print("手术时间：", self.operation_time)
        print("牙位信息：", self.fillteeth_name)
        print("患牙类型：", self.fillteeth_type)
        print("医生姓名：", self.doctor_name)
        print("图片格式：", self.img_type)


class Teeth:
    def __init__(self):
        # self.AREA_K = 1.0  # 面积得分*此系数 对包含相邻牙齿得分进行扣分
        # self.THR_HEIGHT = 0.25  # 大于 ROI*THR_HEIGHT 判断有相邻牙齿
        # self.THR_WIDTH = 0.5  # 大于 ROI*THR_WIDTH 判断相邻牙齿完整

        self.src_image = 0
        self.src_gray_image = 0
        self.dst_all_mark = 0
        self.dst_fill_mark = 0
        self.dst_other_mark = 0
        self.dst_fillarea_mark = 0
        self.fill_rect = (0, 0, 0, 0)
        self.neighbor_rect = (0, 0, 0, 0)
        self.img_info = Img_info()
        self.neighbor_flag = 0  # 相邻牙齿标志位 0：无， 1：左， 2：右， 3：左右

    # / *清除私有成员数据 * /
    def clear(self):
        # print("clear data")
        pass

    # / *读取照片 * /
    def read_image(self, image_path):
        # self.src_image = cv2.imread(image_path)
        self.src_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                                      -1)
        return

    # / *提取所有需要的牙齿，包括单个患牙，全部牙齿，其他牙齿 * /
    def extract_all(self, img_path):
        self.read_image(img_path)
        self.resize(TEETH_IMAGE_SET_ROW, TEETH_IMAGE_SET_ROW)
        self.extract_all_teeth()
        # yolo3检测获得各个牙齿矩形框与类别
        rect_list = detect_img(img_path)
        if len(rect_list) != 0:
            rect_list = result_resize(rect_list,
                                      np.max(self.src_image.shape[0:2]))
            result_show(rect_list, self.src_image)

        # 获得所补牙位置矩形框（col1,row1,col2,row2）
        self.fill_rect = self.get_fill_site(rect_list,
                                            self.img_info.fillteeth_name)

        # 对所补牙类型进行判断（俯视与平视），并分割所补牙
        self.img_info.fillteeth_type = self.classify_filltype(
            self.src_image.copy(), self.fill_rect)
        self.dst_fill_mark = self.extract_fill_teeth(self.src_image.copy(),
                                                     self.fill_rect)
        # 针对术中，分割所补区域
        if self.img_info.operation_time == '术中':
            self.dst_fillarea_mark = self.extract_fillarea(self.dst_fill_mark)

        # 针对术后，获得所补牙邻牙信息，并分割邻牙
        self.neighbor_flag, self.neighbor_rect = self.get_fillarea_info(
                rect_list, self.img_info.fillteeth_name)
        print(self.neighbor_flag, self.neighbor_rect)
        if self.img_info.operation_time == '术后':
            self.dst_other_mark = self.extract_neighbor_teeth(
                self.src_image.copy(), self.neighbor_rect)
        return

    # / *调整图片大小 * /
    def resize(self, set_rows, set_cols):
        img_rows, img_cols = self.src_image.shape[:2]

        if img_rows >= img_cols and img_rows > set_rows:
            resize_k = set_rows / img_rows
            self.src_image = cv2.resize(self.src_image,
                                        (int(resize_k * img_cols), set_rows),
                                        interpolation=cv2.INTER_AREA)
        elif img_cols > img_rows and img_cols > set_cols:
            resize_k = set_cols / img_cols
            self.src_image = cv2.resize(self.src_image,
                                        (set_cols, int(resize_k * img_rows)),
                                        interpolation=cv2.INTER_AREA)

        # 初始化
        self.src_gray_image = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2GRAY)
        self.dst_all_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        self.dst_fill_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        self.dst_other_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        self.dst_fillarea_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        return

    # / *分割单个患牙 * /
    def extract_fill_teeth(self, src_img, fill_rect):
        w = fill_rect[2] - fill_rect[0]
        h = fill_rect[3] - fill_rect[1]
        img_rows, img_cols = src_img.shape[:2]

        min_row = int(my_limit(fill_rect[1] - h * 0.1, 0, img_rows))
        max_row = int(my_limit(fill_rect[3] + h * 0.1, 0, img_rows))
        min_col = int(my_limit(fill_rect[0] - w * 0.1, 0, img_cols))
        max_col = int(my_limit(fill_rect[2] + h * 0.1, 0, img_cols))

        roi_img = src_img[min_row:max_row, min_col:max_col]
        row, col = roi_img.shape[:2]

        # unet获得目标牙齿的bin
        # roi_img = cv2.resize(roi_img, (256, 256))
        mark_bin = unet_extract_fillteeth(roi_img)
        mark_bin = cv2.resize(mark_bin, (col, row))
        mark_bin = my_erode_dilate(mark_bin, 2, 2, (10, 10))

        # 将roi图转换到全图
        fill_mark = np.zeros((img_rows, img_cols), np.uint8)
        fill_mark[min_row:max_row, min_col:max_col] = mark_bin

        return fill_mark

    def classify_filltype(self, src_img, fill_rect):
        w = fill_rect[2] - fill_rect[0]
        h = fill_rect[3] - fill_rect[1]
        img_rows, img_cols = src_img.shape[:2]

        min_row = int(my_limit(fill_rect[1] - h * 0.1, 0, img_rows))
        max_row = int(my_limit(fill_rect[3] + h * 0.1, 0, img_rows))
        min_col = int(my_limit(fill_rect[0] - w * 0.1, 0, img_cols))
        max_col = int(my_limit(fill_rect[2] + h * 0.1, 0, img_cols))

        roi_img = src_img[min_row:max_row, min_col:max_col]
        row, col = roi_img.shape[:2]

        # 分类目标牙齿（后牙1与非后牙0）
        label = classify_teethtype(roi_img)
        if label == 0:
            # if self.img_info.fillteeth_type == '门牙':
            #     print('判断相同')
            # else:
            #     print('判断不同')
            fill_type = '门牙'
        elif label == 1:
            # if self.img_info.fillteeth_type == '后牙':
            #     print('判断相同')
            # else:
            #     print('判断不同')
            fill_type = '后牙'
        return fill_type

    # / *根据yolo获得所补牙位置信息 * /
    def get_fill_site(self, rect_data, fill_name):
        fill_region = list(fill_name)[0]
        fill_num = list(fill_name)[1]
        fill_rect = []
        class_label = np.array(rect_data[:, 0]).astype(np.str)
        arg_fill = np.where(class_label == fill_num)
        if len(arg_fill) != 0:
            rect_fill = rect_data[arg_fill[0][0], :]
            fill_rect = rect_fill[2:].astype(np.int)
        return fill_rect

    def get_fillarea_info(self, rect_data, fill_name):
        fill_region = list(fill_name)[0]
        fill_num = list(fill_name)[1]
        neighbor_rect = []
        neighbor_flag = 0

        class_label = np.array(rect_data[:, 0]).astype(np.str)
        arg_fill = np.where(class_label == fill_num)
        if len(arg_fill) != 0:
            if arg_fill[0][0] + 1 < len(rect_data):
                neighbor_flag += 2
                rect_neighbor = rect_data[arg_fill[0][0] + 1, :]
                neighbor_rect = rect_neighbor[2:].astype(np.int)

            if arg_fill[0][0] - 1 >= 0:
                neighbor_flag += 1
                rect_neighbor = rect_data[arg_fill[0][0] - 1, :]
                neighbor_rect = rect_neighbor[2:].astype(np.int)

        return neighbor_flag, neighbor_rect

    # / *获得患牙邻牙 * /
    def extract_neighbor_teeth(self, src_img, neighbor_rect):
        w = neighbor_rect[2] - neighbor_rect[0]
        h = neighbor_rect[3] - neighbor_rect[1]
        img_rows, img_cols = src_img.shape[:2]

        min_row = int(my_limit(neighbor_rect[1] - h * 0.1, 0, img_rows))
        max_row = int(my_limit(neighbor_rect[3] + h * 0.1, 0, img_rows))
        min_col = int(my_limit(neighbor_rect[0] - w * 0.1, 0, img_cols))
        max_col = int(my_limit(neighbor_rect[2] + h * 0.1, 0, img_cols))

        roi_img = src_img[min_row:max_row, min_col:max_col]
        row, col = roi_img.shape[:2]

        # unet获得目标牙齿的bin
        # roi_img = cv2.resize(roi_img, (256, 256))
        mark_bin = unet_extract_fillteeth(roi_img)
        mark_bin = cv2.resize(mark_bin, (col, row))
        mark_bin = my_erode_dilate(mark_bin, 2, 2, (10, 10))

        # 将roi图转换到全图
        fillarea_mark = np.zeros((img_rows, img_cols), np.uint8)
        fillarea_mark[min_row:max_row, min_col:max_col] = mark_bin

        return fillarea_mark

    def extract_fillarea(self, fill_mark):
        img_rows, img_cols = fill_mark.shape[:2]
        img, contours, hierarchy = cv2.findContours(fill_mark.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col, row, w, h = cv2.boundingRect(maxcnt)

            row1 = my_limit(row - 10, 0, img_rows)
            row2 = my_limit(row + h + 10, 0, img_rows)
            col1 = my_limit(col - 10, 0, img_cols)
            col2 = my_limit(col + w + 10, 0, img_cols)
            img_teeth_bgr = self.bin_to_rgb(fill_mark)
            mark_bin = unet_extract_fillarea(
                img_teeth_bgr[row1:row2, col1:col2, :])
            mark_bin = cv2.resize(mark_bin, (col2 - col1, row2 - row1))
            mark_bin = my_erode_dilate(mark_bin, 1, 1, (5, 5), order=1)
            # 还原至原图
            fillarea_mark = np.zeros((img_rows, img_cols), np.uint8)
            fillarea_mark[row1:row2, col1:col2] = mark_bin
            # 可视化所补区域
            fillarea = self.bin_to_rgb(fillarea_mark)
            cv2.imshow('mark_area', fillarea)

            return fillarea_mark

    # / *展示最终结果照片 * /
    def img_show(self):
        fill_teeth = self.bin_to_rgb(self.dst_fill_mark)
        cv2.imshow("fill_teeth", fill_teeth)
        # all_teeth = self.bin_to_rgb(self.dst_all_mark)
        # cv2.imshow("all_teeth", all_teeth)

        other_teeth = self.bin_to_rgb(self.dst_other_mark)
        cv2.imshow("other_teeth", other_teeth)
        # cv2.imshow("原图", self.src_image)
        return

    # / *提取照片中的全部牙齿 * /
    def extract_all_teeth(self):
        self.filter_to_bin()

        # 大津阈值
        src_img_copy = self.bin_to_rgb(self.dst_all_mark)
        thr = my_otsu_hsv(self.src_image, 0, 20)
        self.dst_all_mark = my_threshold_hsv(src_img_copy, thr)
        self.dst_all_mark = my_fill_hole(self.dst_all_mark)
        self.dst_all_mark = my_erode_dilate(self.dst_all_mark, 4, 4, (5, 5))

        # 仅保存最大轮廓
        img, contours, hierarchy = cv2.findContours(self.dst_all_mark.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            mark_filted = np.zeros(self.dst_all_mark.shape[0:2],
                                   dtype=np.uint8)
            cv2.drawContours(mark_filted, [maxcnt], -1, 255, -1)
            self.dst_all_mark = mark_filted

        return 1

    # / * hsv过滤图片到bin  * /
    def filter_to_bin(self):
        hsv_image = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2HSV)
        img_rows, img_cols = hsv_image.shape[:2]

        H, S, V = cv2.split(hsv_image)

        self.dst_all_mark[(H <= 120) & (H >= 5)] = 255

        self.dst_all_mark = my_erode_dilate(self.dst_all_mark, 2, 6, (5, 5))
        # cv2.imshow("dst_all_mark", self.dst_all_mark)

        return 1

    # / *将二值化图映射到原图 * /
    def bin_to_rgb(self, bin_img):
        img_rows, img_cols = bin_img.shape[:2]
        re_dst_image = np.zeros(self.src_image.shape, np.uint8)

        re_dst_image[bin_img == 255] = self.src_image[bin_img == 255]

        return re_dst_image

    # / * 获得所补牙矩形框的位置信息，调试时使用 * /
    def get_roi(self, event, x, y, flags, param):
        global label_flag
        if event == cv2.EVENT_LBUTTONDOWN:
            label_flag = 1  # 左键按下
            PointStart[0], PointStart[1] = x, y  # 记录起点位置
        elif event == cv2.EVENT_LBUTTONUP and label_flag == 1:  # 左键按下后检测弹起
            label_flag = 2  # 左键弹起
            PointEnd[0], PointEnd[1] = x, y  # 记录终点位置
            PointEnd[1] = PointStart[1] + (PointEnd[0] - PointStart[0]
                                           )  # 形成正方形
            # 提取ROI
            if PointEnd[0] != PointStart[0] and PointEnd[1] != PointStart[
                    1]:  # 框出了矩形区域,而非点
                print("row", (PointStart[1] + PointEnd[1]) // 2)
                print("col", (PointStart[0] + PointEnd[0]) // 2)
                # 获取矩形框左上角以及右下角点坐标
                PointLU = [0, 0]  # 左上角点
                PointRD = [0, 0]  # 右下角点
                # 左上角点xy坐标值均较小
                PointLU[0] = min(PointStart[0], PointEnd[0])
                PointLU[1] = min(PointStart[1], PointEnd[1])
                # 右下角点xy坐标值均较大
                PointRD[0] = max(PointStart[0], PointEnd[0])
                PointRD[1] = max(PointStart[1], PointEnd[1])
                # roi宽度
                roi_width = PointRD[0] - PointLU[0]
                print("r = %d" % (roi_width // 2))
                try:
                    f = open(param[0], 'r')
                except IOError:
                    print("缺少必要文件 site.text")
                    return
                lines = [',\n', ',\n', ',\n']
                for i in range(3):
                    lines[i] = f.readline()
                f.close()
                f = open(param[0], 'w')
                if param[1] == "术前":
                    lines[0] = str(
                        (PointStart[1] + PointEnd[1]) // 2) + " " + str(
                            (PointStart[0] + PointEnd[0]) // 2)
                    lines[0] += " " + str(roi_width // 2) + ',\n'
                elif param[1] == "术中":
                    lines[1] = str(
                        (PointStart[1] + PointEnd[1]) // 2) + " " + str(
                            (PointStart[0] + PointEnd[0]) // 2)
                    lines[1] += " " + str(roi_width // 2) + ',\n'
                elif param[1] == "术后":
                    lines[2] = str(
                        (PointStart[1] + PointEnd[1]) // 2) + " " + str(
                            (PointStart[0] + PointEnd[0]) // 2)
                    lines[2] += " " + str(roi_width // 2) + ',\n'
                s = ''.join(lines)
                print(s)
                f.write(s)
                f.close()

        elif event == cv2.EVENT_MOUSEMOVE and label_flag == 1:  # 左键按下后获取当前坐标, 并更新标注框
            PointEnd[0], PointEnd[1] = x, y  # 记录当前位置
            PointEnd[1] = PointStart[1] + (PointEnd[0] - PointStart[0]
                                           )  # 形成正方形
            image_copy = copy.deepcopy(self.src_image)
            cv2.rectangle(image_copy, (PointStart[0], PointStart[1]),
                          (PointEnd[0], PointEnd[1]), (0, 255, 0),
                          1)  # 根据x坐标画正方形
            cv2.imshow('get_roi', image_copy)
        return

    # / * 获得某个位置的像素值，调试时使用 * /
    def get_point_value(self, event, x, y, flags, param):
        global label_flag
        if event == cv2.EVENT_LBUTTONDOWN:
            label_flag = 1  # 左键按下
        elif event == cv2.EVENT_LBUTTONUP and label_flag == 1:  # 左键按下后检测弹起
            label_flag = 2  # 左键弹起
        elif event == cv2.EVENT_MOUSEMOVE and label_flag == 1:  # 左键按下后获取当前坐标
            point_col, point_row = x, y  # 记录当前位置
            print("像素值：", param[point_row, point_col])
        return


# / *判断此次运行程序前，照片数目，命名是否正确 * /
def pro_require(img_names):
    jpg_num = 0
    first_flag = 0
    second_flag = 0
    third_flag = 0
    correct_img_names = [0 for i in range(3)]
    for i in range(len(img_names)):
        img_str = img_names[i].split(".")
        if img_str[1].lower() == "jpg":
            jpg_num += 1
            img_name_str = img_str[0].split("-")
            operation_time = img_name_str[2]
            # operation_time = img_name_str[1]

            if operation_time == '术前' and first_flag == 0:
                first_flag = 1
                correct_img_names[0] = img_names[i]
            elif operation_time == '术中' and second_flag == 0:
                second_flag = 1
                correct_img_names[1] = img_names[i]
            elif operation_time == '术后' and third_flag == 0:
                third_flag = 1
                correct_img_names[2] = img_names[i]
    if first_flag == 1 and second_flag == 1 and third_flag == 1 and jpg_num == 3:
        return 1, correct_img_names
    return 0, correct_img_names


def my_limit(a, min_a, max_a):
    if a < min_a:
        a = min_a
    elif a > max_a:
        a = max_a

    return a


# 大津阈值针对HSV
def my_otsu_hsv(src_image, start, end, channel='H'):
    hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    img_rows, img_cols = hsv_image.shape[:2]
    H, S, V = cv2.split(hsv_image)

    if channel == 'H':
        thr_channel = H
    elif channel == 'S':
        thr_channel = S
    elif channel == 'V':
        thr_channel = V

    pixel_count = [0 for x in range(256)]
    h_sum = 0
    h_sum_count = 0
    threshold = 0
    for i in range(start, end):
        pixel_count[i] = np.sum(thr_channel == i)  # 所需像素个数
        h_sum_count += pixel_count[i]  # 总数
    if h_sum_count != 0:
        pixel_pro = [pixel_count[i] / h_sum_count for i in range(256)]

        delta_max = 0
        for i in range(start, end):
            w0 = w1 = u0_temp = u1_temp = 0.0
            for j in range(start, end):
                if j <= i:  # //背景部分
                    w0 += pixel_pro[j]  # //背景像素比例
                    u0_temp += j * pixel_pro[j]
                else:  # //前景部分
                    w1 += pixel_pro[j]  # //前景像素比例
                    u1_temp += j * pixel_pro[j]
            if w0 != 0 and w1 != 0:
                u0 = u0_temp / w0  # //背景像素点的平均灰度
                u1 = u1_temp / w1  # //前景像素点的平均灰度
                delta_temp = (w0 * w1 * pow((u0 - u1), 2))
                # // 当类间方差delta_temp最大时，对应的i就是阈值T
                if delta_temp > delta_max:
                    delta_max = delta_temp
                    threshold = i

    return threshold


# 二值化针对HSV
def my_threshold_hsv(src_image, thr, channel='H'):
    hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
    img_rows, img_cols = hsv_image.shape[:2]
    H, S, V = cv2.split(hsv_image)

    if channel == 'H':
        thr_channel = H
    elif channel == 'S':
        thr_channel = S
    elif channel == 'V':
        thr_channel = V

    bin_image = np.zeros((img_rows, img_cols), np.uint8)

    bin_image[thr_channel > thr] = 255

    return bin_image


# 填充孔洞
def my_fill_hole(bin_image):
    im_fill = bin_image.copy()
    h, w = bin_image.shape[:2]

    im_fill[0, :] = 0
    im_fill[:, 0] = 0
    im_fill[h - 1, :] = 0
    im_fill[:, w - 1] = 0

    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_fill, mask, (0, 0), 255)

    im_fill_inv = cv2.bitwise_not(im_fill)
    bin_image = bin_image | im_fill_inv

    return bin_image


def my_erode_dilate(bin_image, erode_num, dilate_num, size, order=0):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    if order == 0:
        for i in range(0, erode_num):
            bin_image = cv2.erode(bin_image, element)
        for i in range(0, dilate_num):
            bin_image = cv2.dilate(bin_image, element)
    else:
        for i in range(0, dilate_num):
            bin_image = cv2.dilate(bin_image, element)
        for i in range(0, erode_num):
            bin_image = cv2.erode(bin_image, element)
    return bin_image

    # 标记分水岭算法提取单个所补牙
    # min_row = int(my_limit(site[0] - radius, 0, self.src_image.shape[0]))
    # max_row = int(my_limit(site[0] + radius, 0, self.src_image.shape[0]))
    # min_col = int(my_limit(site[1] - radius, 0, self.src_image.shape[1]))
    # max_col = int(my_limit(site[1] + radius, 0, self.src_image.shape[1]))

    # self.dst_fill_image = copy.deepcopy(self.src_image)
    # sure_bg = np.zeros((self.src_image.shape[0], self.src_image.shape[1]), np.uint8)
    # for r in range(min_row, max_row):  # 10 155
    #     for c in range(min_col, max_col):  # 70 205
    #         sure_bg[r, c] = 255

    # sure_fg = np.zeros((self.src_image.shape[0], self.src_image.shape[1]), np.uint8)
    # for r in range(site[0]-radius//2, site[0]+radius//2):
    #     for c in range(site[1]-radius//2, site[1]+radius//2):
    #         sure_fg[r, c] = 255
    # # cv2.imshow("sure_fg", sure_fg)

    # # 未知的区域
    # unknow = cv2.subtract(sure_bg, sure_fg)
    # # cv2.imshow("unknow", unknow)

    # # 标记
    # ret, markers = cv2.connectedComponents(sure_bg)  # 将确定的背景标记为0,其他为非零整数
    # markers = markers + 1  # 将确定的背景记为1
    # markers[unknow == 255] = 0  # 将确未知区域标记为0

    # markers = cv2.watershed(self.src_image, markers)

    # for r in range(0, self.src_image.shape[0]):
    #     for c in range(0, self.src_image.shape[1]):
    #         if markers[r, c] == 2:
    #             self.dst_fill_mark[r, c] = 255
    # return