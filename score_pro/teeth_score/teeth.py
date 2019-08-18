#!/usr/bin/python3.6
# coding=utf-8

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os, re, time, copy, sys
from shutil import copyfile

from teeth_score.U_net.unet_extract import unet_extract_fillteeth, unet_extract_fillarea
from teeth_score.AlexNet.classify_teethtype import classify_teethtype
from teeth_score.Yolo3.yolo_rect import detect_img

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
        self.pro_path = work_floder + 'JPG_TEST'

    def get_info(self, img_dir, use_deploy=0):
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

            # self.fillteeth_type = info[3]
            # pattern = r"(.*)-(.*)-(.*)-(.*)\.(.*)"
            # info = list(re.findall(pattern, img_dir)[0])
            # self.upload_time = info[0]
            # self.patient_name = info[0]
            # self.operation_time = info[1]
            # self.fillteeth_type = info[2]
            # self.doctor_name = info[3]
            # self.img_type = info[4]
        else:
            str_img_path = img_dir.split("/")
            img_name = str_img_path[len(str_img_path) - 1]
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
        self.AREA_K = 1.0  # 面积得分*此系数 对包含相邻牙齿得分进行扣分
        self.THR_HEIGHT = 0.25  # 大于 ROI*THR_HEIGHT 判断有相邻牙齿
        self.THR_WIDTH = 0.5  # 大于 ROI*THR_WIDTH 判断相邻牙齿完整

        self.src_image = 0
        self.src_gray_image = 0
        self.dst_all_mark = 0
        self.dst_fill_mark = 0
        self.dst_other_mark = 0
        self.dst_fillarea_mark = 0
        self.neighbor_ok = True
        self.site = (0, 0)
        self.radius = 0
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

        # 二值化
        self.src_gray_image = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2GRAY)
        self.dst_all_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        self.dst_fill_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        self.dst_other_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        self.dst_fillarea_mark = np.zeros(self.src_image.shape[:2], np.uint8)
        return

    # / * hsv过滤图片到bin * /
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

    # / *分割单个患牙 * /
    def find_fill_teeth(self, all_mark, site, radius=0):
        if radius != 0:
            min_row = int(
                my_limit(site[0] - radius, 0, self.src_image.shape[0]))
            max_row = int(
                my_limit(site[0] + radius, 0, self.src_image.shape[0]))
            min_col = int(
                my_limit(site[1] - radius, 0, self.src_image.shape[1]))
            max_col = int(
                my_limit(site[1] + radius, 0, self.src_image.shape[1]))
        else:
            min_row = int(my_limit(site[0], 0, self.src_image.shape[0]))
            max_row = int(my_limit(site[2], 0, self.src_image.shape[0]))
            min_col = int(my_limit(site[1], 0, self.src_image.shape[1]))
            max_col = int(my_limit(site[3], 0, self.src_image.shape[1]))

        roi_img = self.src_image[min_row:max_row, min_col:max_col]
        row, col = roi_img.shape[:2]

        # unet获得目标牙齿的bin
        # roi_img = cv2.resize(roi_img, (256, 256))
        mark_bin = unet_extract_fillteeth(roi_img)
        mark_bin = cv2.resize(mark_bin, (col, row))
        mark_bin = my_erode_dilate(mark_bin, 2, 2, (10, 10))

        # 将roi图转换到全图
        self.dst_fill_mark[min_row:max_row, min_col:max_col] = mark_bin
        return

    def classify_fillteeth(self, site, radius=0):
        if radius != 0:
            min_row = int(
                my_limit(site[0] - radius, 0, self.src_image.shape[0]))
            max_row = int(
                my_limit(site[0] + radius, 0, self.src_image.shape[0]))
            min_col = int(
                my_limit(site[1] - radius, 0, self.src_image.shape[1]))
            max_col = int(
                my_limit(site[1] + radius, 0, self.src_image.shape[1]))
        else:
            min_row = int(my_limit(site[0], 0, self.src_image.shape[0]))
            max_row = int(my_limit(site[2], 0, self.src_image.shape[0]))
            min_col = int(my_limit(site[1], 0, self.src_image.shape[1]))
            max_col = int(my_limit(site[3], 0, self.src_image.shape[1]))

        # dst_all_rgb = self.bin_to_rgb(all_mark)
        roi_img = self.src_image[min_row:max_row, min_col:max_col]
        row, col = roi_img.shape[:2]

        # 分类目标牙齿（后牙1与非后牙0）
        label = classify_teethtype(roi_img)
        if label == 0:
            # if self.img_info.fillteeth_type == '门牙':
            #     print('判断相同')
            # else:
            #     print('判断不同')
            self.img_info.fillteeth_type = '门牙'
        elif label == 1:
            # if self.img_info.fillteeth_type == '后牙':
            #     print('判断相同')
            # else:
            #     print('判断不同')
            self.img_info.fillteeth_type = '后牙'

    def find_neighbor_info(self, dst_all_mark, dst_fill_mark, site, radius=-1):
        self.neighbor_flag = 0

        src_row, src_col = dst_all_mark.shape[:2]

        # 坐标转换
        if radius != -1:
            min_row = int(my_limit(site[0] - radius, 0, src_row))
            max_row = int(my_limit(site[0] + radius, 0, src_row))
            min_col = int(my_limit(site[1] - radius, 0, src_col))
            max_col = int(my_limit(site[1] + radius, 0, src_col))
        else:
            min_row = int(my_limit(site[0], 0, src_row))
            max_row = int(my_limit(site[2], 0, src_row))
            min_col = int(my_limit(site[1], 0, src_col))
            max_col = int(my_limit(site[3], 0, src_col))

        # 区域可视化
        # roi = dst_all_mark[min_row:max_row, min_col:max_col]
        # cv2.imshow("aa1_roi", roi)
        img, contours, hierarchy = cv2.findContours(dst_fill_mark.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col, row, w, h = cv2.boundingRect(maxcnt)

        roi_height = max_row - min_row
        roi_width = max_col - min_col

        # 统计患牙坐标周围第一列和最后一列白点数
        left_tooth_point_num = np.sum(dst_all_mark[0:src_row, min_col] == 255)
        right_tooth_point_num = np.sum(dst_all_mark[0:src_row, max_col -
                                                    1] == 255)

        # 参照牙存在情况:1-有牙 2-不完整 3-最里侧
        # 认为一侧白点超过一定数量,并且原始图像感兴趣区域左侧/右侧还有较大空间,则这一侧有牙
        if left_tooth_point_num > roi_height * self.THR_HEIGHT:
            if col <= 10:
                left_tooth_status = 3
            elif min_col > roi_width * 0.7 * self.THR_WIDTH:
                left_tooth_status = 1
                self.neighbor_flag = 1
            else:
                left_tooth_status = 2
        else:
            left_tooth_status = 3

        if right_tooth_point_num > roi_height * self.THR_HEIGHT:
            if (col + w) >= (src_col - 10):
                right_tooth_status = 3
            elif src_col - max_col > roi_width * 0.7 * self.THR_WIDTH:
                right_tooth_status = 1
                self.neighbor_flag = 2
            else:
                right_tooth_status = 2
        else:
            right_tooth_status = 3

        if left_tooth_status == 3 and right_tooth_status == 3:
            self.neighbor_flag = 0

        elif left_tooth_status == 1 and right_tooth_status == 1:
            self.neighbor_flag = 3

    # / *将全部牙齿与单个患牙相减，得到除患牙外的其他牙齿 * /
    def find_neighbor_teeth(self, all_mark, fill_mark, site, radius=-1):
        self.find_neighbor_info(all_mark, fill_mark, site, radius)

        if self.neighbor_flag == 0:
            return 0, False

        img_rows, img_cols = all_mark.shape[:2]
        dst_all_rgb = self.bin_to_rgb(all_mark)

        if radius != -1:
            ra = radius
        else:
            min_row = int(my_limit(site[0], 0, self.src_image.shape[0]))
            max_row = int(my_limit(site[2], 0, self.src_image.shape[0]))
            min_col = int(my_limit(site[1], 0, self.src_image.shape[1]))
            max_col = int(my_limit(site[3], 0, self.src_image.shape[1]))
            ra = abs(max_col - min_col) // 2
        site_c = site[1]
        site_r = site[0]

        # 仅保存最大轮廓
        img, contours, hierarchy = cv2.findContours(fill_mark.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col, row, w, h = cv2.boundingRect(maxcnt)
            if self.neighbor_flag == 1 or self.neighbor_flag == 3:
                site_c = int(col - w * 0.2)
                min_val = 10000
                init_val = np.sum(all_mark[:, site_c] == 255)
                min_c = 0
                while site_c > 0:
                    val = np.sum(all_mark[:, site_c] == 255)
                    if val > init_val:
                        init_val = val
                    if val < init_val - 5:
                        if val < min_val:
                            min_val = val
                            min_c = site_c
                    if val - min_val > 5:
                        break
                    site_c -= 1
                # ra = (col - min_c)//2
                site_c = (min_c + col) // 2
            elif self.neighbor_flag == 2:
                site_c = int(col + w * 1.2)
                min_val = 10000
                init_val = np.sum(all_mark[:, site_c] == 255)
                min_c = img_cols
                for c in range(site_c, img_cols):
                    val = np.sum(all_mark[:, c] == 255)
                    if val > init_val:
                        init_val = val
                    if val < init_val - 5:
                        if val < min_val:
                            min_val = val
                            min_c = c
                    if val - min_val > 5:
                        break

                # ra = (min_c - col-w)//2
                site_c = (min_c + col + w) // 2

        row_list = []
        for r in range(img_rows):
            if all_mark[r, site_c] == 255:
                row_list.append(r)
        if len(row_list):
            site_r = int(np.mean(row_list))
            # print(site_r)
        else:
            print("error:寻找邻牙位置")
            return 0, False

        min_row = int(my_limit(site_r - ra * 2, 0, img_rows))
        max_row = int(my_limit(site_r + ra * 2, 0, img_rows))
        min_col = int(my_limit(site_c - ra * 1.3, 0, img_cols))
        max_col = int(my_limit(site_c + ra * 1.3, 0, img_cols))

        # 标记分水岭
        sure_bg = np.zeros((img_rows, img_cols), np.uint8)
        for r in range(min_row, max_row):  # 10 155
            for c in range(min_col, max_col):  # 70 205
                sure_bg[r, c] = 255

        sure_fg = np.zeros((img_rows, img_cols), np.uint8)
        for r in range(site_r - ra // 6, site_r + ra // 6):
            for c in range(site_c - ra // 6, site_c + ra // 6):
                sure_fg[r, c] = 255
        # cv2.imshow("sure_fg", sure_fg)

        # 未知的区域
        unknow = cv2.subtract(sure_bg, sure_fg)
        # cv2.imshow("unknow", unknow)

        # 标记
        ret, markers = cv2.connectedComponents(sure_bg)  # 将确定的背景标记为0,其他为非零整数
        markers = markers + 1  # 将确定的背景记为1
        markers[unknow == 255] = 0  # 将确未知区域标记为0

        markers = cv2.watershed(dst_all_rgb, markers)
        # cv2.imshow("dst_all_rgb", dst_all_rgb)

        other = np.zeros((img_rows, img_cols), dtype=np.uint8)
        other[markers == 2] = 255

        other = my_erode_dilate(other, 4, 4, (5, 5))

        # # unet 提取相邻牙齿
        # roi_img = dst_all_rgb[min_row:max_row, min_col:max_col]
        # row, col = roi_img.shape[:2]

        # # unet获得目标牙齿的bin
        # roi_img = cv2.resize(roi_img, (128, 128))
        # mark_bin = unet_extract_fillteeth(roi_img)
        # mark_bin = cv2.resize(mark_bin, (col, row))

        # other[min_row:max_row, min_col:max_col] = mark_bin

        return other, True

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

    def find_fill_area(self, fill_teeth_bin):
        img_rows, img_cols = self.src_image.shape[:2]
        img, contours, hierarchy = cv2.findContours(fill_teeth_bin.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col, row, w, h = cv2.boundingRect(maxcnt)

            row1 = my_limit(row - 10, 0, img_rows)
            row2 = my_limit(row + h + 10, 0, img_rows)
            col1 = my_limit(col - 10, 0, img_cols)
            col2 = my_limit(col + w + 10, 0, img_cols)
            img_teeth_bgr = self.bin_to_rgb(fill_teeth_bin)
            mark_bin = unet_extract_fillarea(
                img_teeth_bgr[row1:row2, col1:col2, :])
            mark_bin = cv2.resize(mark_bin, (col2 - col1, row2 - row1))
            mark_bin = my_erode_dilate(mark_bin, 1, 1, (5, 5), order=1)
            self.dst_fillarea_mark[row1:row2, col1:col2] = mark_bin
            # 可视化所补区域
            fillarea = self.bin_to_rgb(self.dst_fillarea_mark)
            cv2.imshow('mark_area', self.dst_fillarea_mark)

    # / *提取所有需要的牙齿，包括单个患牙，全部牙齿，其他牙齿 * /
    def extract_all(self,
                    use_deploy,
                    current_path=0,
                    img_name=0,
                    img_path=0,
                    site=0):
        if use_deploy == 0:
            img_path = os.path.join(current_path, img_name)
            txt_path = os.path.join(current_path, "site.txt")

        self.read_image(img_path)
        self.resize(TEETH_IMAGE_SET_ROW, TEETH_IMAGE_SET_ROW)
        detect_img(img_path)

        if use_deploy == 0:
            self.get_fill_teeth_site(use_deploy,
                                     self.img_info.operation_time,
                                     txt_path=txt_path)
        else:
            self.get_fill_teeth_site(use_deploy,
                                     self.img_info.operation_time,
                                     site_str=site)

        self.extract_all_teeth()

        if use_deploy == 0:
            self.find_fill_teeth(self.dst_all_mark, self.site, self.radius)
            self.classify_fillteeth(self.site, self.radius)
        else:
            self.find_fill_teeth(self.dst_all_mark, self.site)
            self.classify_fillteeth(self.site)

        if self.img_info.operation_time == '术中':
            self.find_fill_area(self.dst_fill_mark)

        if self.img_info.operation_time == '术后':
            if use_deploy == 0:
                self.dst_other_mark, self.neighbor_ok = self.find_neighbor_teeth(
                    self.dst_all_mark, self.dst_fill_mark, self.site,
                    self.radius)
            else:
                self.dst_other_mark, self.neighbor_ok = self.find_neighbor_teeth(
                    self.dst_all_mark, self.dst_fill_mark, self.site)
        return

    # / *根据site.txt文件过得所补牙位置信息 * /
    def get_fill_teeth_site(self, use_deploy, time, txt_path=0, site_str=0):
        if use_deploy == 0:
            try:
                f = open(txt_path)
            except IOError:
                print("缺少必要文件 site.text")
                return
            site_str = f.read()
            site_str = site_str.split(",\n")

            if time == '术前':
                str_temp = site_str[0].split()
                self.site = (int(str_temp[0]), int(str_temp[1]))
                self.radius = int(str_temp[2])
            elif time == '术中':
                str_temp = site_str[1].split()
                self.site = (int(str_temp[0]), int(str_temp[1]))
                self.radius = int(str_temp[2])
            elif time == '术后':
                str_temp = site_str[2].split()
                self.site = (int(str_temp[0]), int(str_temp[1]))
                self.radius = int(str_temp[2])

            f.close()
        else:
            if time == '术前':
                self.site = site_str[0:4]
            elif time == '术中':
                self.site = site_str[4:8]
            elif time == '术后':
                self.site = site_str[8:12]

        return

    # / *展示最终结果照片 * /
    def img_show(self):
        fill_teeth = self.bin_to_rgb(self.dst_fill_mark)
        all_teeth = self.bin_to_rgb(self.dst_all_mark)
        # cv2.imshow("dst_all_mark", self.dst_all_mark)
        # col_num_list = []
        # x_list = [x for x in range(self.dst_all_mark.shape[1])]
        # for c in range(self.dst_all_mark.shape[1]):
        #     col_num_list.append(np.sum(self.dst_all_mark[:,c]==255))

        # plt.plot(x_list,col_num_list,'g-s')
        # plt.show()
        if self.neighbor_ok != False:
            other_teeth = self.bin_to_rgb(self.dst_other_mark)
            cv2.imshow("other_teeth", other_teeth)
        # cv2.imshow("原图", self.src_image)
        cv2.imshow("fill_teeth", fill_teeth)
        cv2.imshow("all_teeth", all_teeth)
        return

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