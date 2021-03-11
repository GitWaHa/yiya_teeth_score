#!/usr/bin/python3.6
# coding=utf-8

import cv2
import numpy as np
from sklearn.cluster import KMeans
import filetype
import math
import os
from teeth import *
from indicators import *
from sklearn.externals import joblib
from network.ResNet50.classify_bb1 import classify_bb1

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import config as myconfig
work_floder = myconfig.WORK_FLODER

BB4_STANDARD_IMGDIR = os.path.join(work_floder, 'mode/BB4_standard_template/')
TXT_DIR = os.path.join(work_floder, 'mode/lr_mode/')


class Teeth_Grade():
    def __init__(self):
        self.aa1 = Indicators_AA1()
        self.aa2 = Indicators_AA2()
        self.aa3 = Indicators_AA3()
        self.bb1 = Indicators_BB1()
        self.bb2 = Indicators_BB2()
        self.bb3 = Indicators_BB3()
        self.bb4 = Indicators_BB4()
        self.roi_site_k_list = 0
        self.str_score2cmd = [0, 0, 0]
        self.grade = 0
        self.print_flag = 0
        self.print_lr_x_flag = 0
        self.lr_x_str = ['0', '0', '0']
        self.lr_flag = 1

    def clear(self):
        self.aa1.clear()
        self.aa2.clear()
        self.aa3.clear()
        self.bb1.clear()
        self.bb2.clear()
        self.bb3.clear()
        self.bb4.clear()
        self.grade = 0

    def score_aa1(self, neighbor_flag, fillteeth_num):
        key_elements_score = self.aa1.CONTAINS_NEIGHBOR_SCORE

        if fillteeth_num == '7' and neighbor_flag > 0:
            self.aa1.contains_neighbor = self.aa1.CONTAINS_NEIGHBOR_SCORE
            self.aa1.neighbor_num = 1
            if self.print_flag == 1:
                print('AA1[INFO]：7号牙，有一颗邻牙')
        elif neighbor_flag == 3:
            self.aa1.contains_neighbor = self.aa1.CONTAINS_NEIGHBOR_SCORE
            self.aa1.neighbor_num = 2
            if self.print_flag == 1:
                print('AA1[INFO]：有两颗邻牙')
        elif neighbor_flag == 0:
            self.aa1.contains_neighbor = 0
            self.aa1.neighbor_num = 0
            if self.print_flag == 1:
                print('AA1[INFO]：无邻牙')
        else:
            self.aa1.contains_neighbor = 4
            self.aa1.neighbor_num = 1
            if self.print_flag == 1:
                print('AA1[INFO]：非7号牙，无邻牙')

        self.aa1.undefined = 3  # 直接给3分
        self.aa1.sum()
        return 1

    def score_aa2(self, dst_all_mark, fill_rect, operation_time,
                  fillteeth_num):
        img_row, img_col = dst_all_mark.shape[:2]
        if self.print_flag == 1:
            print('AA2[INFO]：图片大小', img_row, img_col)
        img_center_row = img_row / 2
        img_center_col = img_col / 2
        fill_cetter_row = (fill_rect[1] + fill_rect[3]) / 2
        fill_cetter_col = (fill_rect[0] + fill_rect[2]) / 2

        # 中心位置偏差评分
        # 患牙与图片中心点距离
        distance = math.sqrt((img_center_row - fill_cetter_row)**2 +
                             (img_center_col - fill_cetter_col)**2)

        # 距离占x的比例
        ratio = distance / math.sqrt(img_col**2 + img_row**2)
        if self.print_flag == 1:
            print('AA2[INFO] 图片中心距离的比例', ratio)
        if operation_time == '术前':  # 术前
            full_scores = self.aa2.CENTER_BIAS_SCORE_FIRST  # 满分
            cut_scores = self.aa2.CENTER_BIAS_SUBTRACT_FIRST  # 按比例扣分
        else:  # 术中和术后
            full_scores = self.aa2.CENTER_BIAS_SCORE_OTHER
            cut_scores = self.aa2.CENTER_BIAS_SUBTRACT_OTHER
        if (self.aa2.CENTER_BIAS_SUBTRACT_START < ratio <
                self.aa2.CENTER_BIAS_SUBTRACT_START_1):
            center_point_scores = full_scores - cut_scores
        elif (self.aa2.CENTER_BIAS_SUBTRACT_START_1 < ratio <
              self.aa2.CENTER_BIAS_SUBTRACT_START_2):
            center_point_scores = full_scores - cut_scores * 2
        elif (self.aa2.CENTER_BIAS_SUBTRACT_START_2 < ratio <
              self.aa2.CENTER_BIAS_SUBTRACT_START_3):
            center_point_scores = full_scores - cut_scores * 3
        elif ratio > self.aa2.CENTER_BIAS_SUBTRACT_START_3:
            center_point_scores = full_scores - cut_scores * 4
        else:
            center_point_scores = full_scores

        if fillteeth_num == '7':
            self.aa2.center_bias = full_scores
        else:
            self.aa2.center_bias = center_point_scores

        # 面积大小占比评分
        pic_area = img_row * img_col  # 图片整体面积
        area_fill = (fill_rect[3] - fill_rect[1]) * (fill_rect[2] -
                                                     fill_rect[0])
        area_ratio = area_fill * (self.aa1.neighbor_num + 1) / pic_area  # 面积比例
        if self.print_flag == 1:
            print('AA2[INFO] 面积占比', area_ratio)

        if self.aa2.AREA_RATIO_SUBTRACT_START_MIN <= area_ratio <= self.aa2.AREA_RATIO_SUBTRACT_START_MAX:
            ratio_diff = 0
        elif area_ratio < self.aa2.AREA_RATIO_SUBTRACT_START_MIN:
            ratio_diff = self.aa2.AREA_RATIO_SUBTRACT_START_MIN - area_ratio
        else:
            ratio_diff = area_ratio - self.aa2.AREA_RATIO_SUBTRACT_START_MAX

        self.aa2.area_ratio = my_limit(
            self.aa2.AREA_RATIO_SCORE -
            math.ceil(ratio_diff / self.aa2.AREA_RATIO_SUBTRACT_RATIO) *
            self.aa2.AREA_RATIO_SUBTRACT, 0,
            self.aa2.AREA_RATIO_SCORE)  # 按比例扣分
        # ××××××××××××××××××××××××××××× 反作用到AA1指标 ×××××××××××××××××××××××××
        # self.aa1.contains_neighbor *= my_limit(
        #     ((self.aa2.area_ratio + 1) / self.aa2.AREA_RATIO_SCORE) *
        #     self.aa1.AREA_K, 0, 1)
        # self.aa1.sum()

        # 提取全部牙齿轮廓，利用线性拟合一条直线用来判断角度
        _, contours, hierarchy = cv2.findContours(dst_all_mark.copy(),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))

            vx, vy, x, y = cv2.fitLine(maxcnt, cv2.DIST_L2, 0, 0.01, 0.01)
            if self.print_flag == 1:
                print('AA2[INFO] 角度', math.atan(vy / vx) / math.pi * 180)

            # 角度可视化
            # lefty = int((-x * vy / vx) + y)
            # righty = int(((dst_all_mark.shape[1] - x) * vy / vx) + y)
            # img = cv2.line(dst_all_mark, (dst_all_mark.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)
            # cv2.imshow("Canvas", img)

            if operation_time == '术前':
                self.aa2.first_angle = math.atan(vy / vx) / math.pi * 180
                self.aa2.shooting_angle = self.aa2.SHOOTING_ANGLE_SCORE_FIRST
            else:
                img_angle = math.atan(vy / vx) / math.pi * 180
                dif_angle = abs(img_angle - self.aa2.first_angle)
                if self.print_flag == 1:
                    print('AA2[INFO] 拍摄角度之差', dif_angle)
                self.aa2.shooting_angle = my_limit(
                    self.aa2.SHOOTING_ANGLE_SCORE_OTHER -
                    (dif_angle // self.aa2.SHOOTING_ANGLE_SUBTRACT_ANGLE) *
                    self.aa2.SHOOTING_ANGLE_SUBTRACT, 0,
                    self.aa2.SHOOTING_ANGLE_SCORE_OTHER)  # 按比例扣分

        if area_ratio < self.aa2.AREA_RATIO_MIN:
            return 0

        self.aa2.sum()
        return 1

    def score_aa3(self, img_path):
        # 获取文件类型
        file_kind = filetype.guess(img_path)

        # 不是jpg文件，直接0分
        if self.print_flag == 1:
            print('AA3[INFO] 文件类型', file_kind.extension)
        if file_kind.extension != 'jpg':
            self.aa3.img_type = 0
            return 0
        else:
            self.aa3.img_type = self.aa3.IMG_TYPE_SCORE

        # img = cv2.imread(img_path, 0)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img_row, img_col = img.shape[:2]

        # 照片比例16：9（在3%的误差范围内4分，必要条件，达不到则AA3直接0分）绝对误差
        if self.print_flag == 1:
            print('AA3[INFO] 照片比例（标准1.77）', (img_col / img_row))
        if abs((img_col / img_row) -
               self.aa3.IMG_RATIO_STD) > self.aa3.IMG_RATIO_ERROR:
            self.aa3.img_ratio = 0
        else:
            self.aa3.img_ratio = self.aa3.IMG_RATIO_SCORE

        # 分辨率得分
        if self.print_flag == 1:
            print(
                'AA3[INFO] 分辨率（%d）' % (self.aa3.IMG_RESOLUTION_SUBTRACT_START),
                (img_col * img_row))
        if img_col * img_row >= self.aa3.IMG_RESOLUTION_SUBTRACT_START:
            self.aa3.img_resolution = self.aa3.IMG_RESOLUTION_SCORE
        else:
            resolution_diff = self.aa3.IMG_RESOLUTION_SUBTRACT_START - img_col * img_row
            self.aa3.img_resolution = my_limit(
                self.aa3.IMG_RESOLUTION_SCORE - math.ceil(
                    resolution_diff / self.aa3.IMG_RESOLUTION_SUBTRACT_SIZE) *
                self.aa3.IMG_RESOLUTION_SUBTRACT, 0,
                self.aa3.IMG_RESOLUTION_SCORE)  # 按比例扣分

        self.aa3.sum()
        return 1

    def score_bb1(self,
                  gray_img,
                  fillarea_img,
                  operation_time,
                  teeth_type=0,
                  rgb_img=0):
        if operation_time == '术前':
            return
        if operation_time == '术后':
            self.bb1.grade = 20
            return

        good_bad_label = classify_bb1(rgb_img)
        # print('good_bad_label ', good_bad_label)
        if np.sum(fillarea_img) == 0:
            self.bb1.grade = 19
            return

        self.lr_x_str = ['0', '0', '0']
        fillarea_img_copy = fillarea_img.copy()
        B, G, R = cv2.split(rgb_img)
        if teeth_type == '门牙':
            thresh = np.mean(gray_img[fillarea_img_copy == 255]) - 40  # 灰度阈值
        else:
            thresh = np.mean(gray_img[fillarea_img_copy == 255]) - 25  # 灰度阈值

        point_num = 0  # 黑点个数
        level_score = self.bb1.BLACK_DEPTH_SCORE  # 黑色深浅得分
        num_score = self.bb1.BLACK_SIZE_SCORE  # 黑色大小得分
        min_gray_value = 255  # 牙齿区域最小灰度值

        rows, cols = gray_img.shape[0:2]

        # 若检测不到，则此次补牙不需清除龋坏，随机给固定分数
        if np.sum(fillarea_img_copy == 255) == 0:
            self.bb1.grade = np.random.randint(18, 20)
            return

        gray_roi_img = gray_img.copy()
        gray_roi_img[fillarea_img_copy == 0] = 0
        black_point_show = np.zeros((rows, cols), dtype=np.uint8)
        black_point_show[(fillarea_img_copy == 255)
                         & (gray_img < thresh)] = 255
        # 可视化龋齿黑点
        # cv2.imshow('black_point_show', black_point_show)
        # R[fillarea_img_copy==0]=0
        # # cv2.imshow('rgb_img', rgb_img[:,:,2])
        # cv2.imshow('rgb_img2', R)
        # cv2.setMouseCallback('rgb_img2', print_value, R)

        # 计算黑色点块的数量
        img, contours, hierarchy = cv2.findContours(black_point_show.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        black_contours_n = len(contours)

        for i in range(0, len(contours)):
            area_fill = cv2.contourArea(contours[i])
            # print(area_fill)
            if area_fill >= 10:
                cv2.drawContours(black_point_show, [contours[i]], -1, 0, -1)
                if black_contours_n > 1:
                    black_contours_n -= 1

        # cv2.imshow('black_point_show3', black_point_show)

        # 计算黑色像素数量
        point_num = np.sum(black_point_show == 255)

        # 计算最小灰度值
        if point_num != 0:
            min_gray_value = np.min(gray_roi_img[black_point_show == 255])

        if self.print_lr_x_flag == 1:
            self.lr_x_str[0] = str(point_num) + ',' + str(
                my_limit(thresh - min_gray_value, 0, 255)) + ',' + str(
                    my_limit(black_contours_n, 0, 5))
            print(self.lr_x_str[0])

        # 根据黑点数量计算黑色大小得分
        if self.print_flag == 1:
            print('BB1[INFO] 黑点数量', point_num)
        num_score -= ((point_num + self.bb1.BLACK_SIZE_SUBTRACT_VALUE/2) // self.bb1.BLACK_SIZE_SUBTRACT_VALUE) * \
                             self.bb1.BLACK_SIZE_SUBTRACT
        if black_contours_n != 0 and point_num >= 3:
            num_score = my_limit(num_score, 0, self.bb1.BLACK_DEPTH_SCORE - 1)

        # 根据最低灰度值计算黑色深浅得分
        if self.print_flag == 1:
            print('BB1[INFO] 最小灰度值', min_gray_value)
        if min_gray_value < thresh:
            level_score -= (((thresh - min_gray_value) // self.bb1.BLACK_DEPTH_SUBTRACT_VALUE)+1) \
                                   * self.bb1.BLACK_DEPTH_SUBTRACT  # 每低于阈值灰度值扣一分
            if black_contours_n != 0:
                level_score = my_limit(level_score, 0,
                                       self.bb1.BLACK_DEPTH_SCORE)

        self.bb1.black_depth = level_score
        self.bb1.black_size = num_score
        self.bb1.sum()
        black_contours_n = my_limit(black_contours_n, 0, 5)
        self.bb1.grade -= black_contours_n
        self.bb1.grade = my_limit(self.bb1.grade, 10, 19)

        if self.lr_flag == 1:
            lr1 = joblib.load(TXT_DIR + 'bb1.pkl')
            bb1x = np.array([[
                point_num,
                my_limit(thresh - min_gray_value, 0, 255), black_contours_n
            ]])
            self.bb1.grade = int(lr1.predict(bb1x)[0][0])
            self.bb1.grade = my_limit(self.bb1.grade, 10, 19)

        return

    def score_bb2(self, fill_mark, operation_time):
        if operation_time == '术前':
            return
        elif operation_time == '术中':
            return

        if self.lr_flag == 0:
            self.bb2.oneself_diff = self.bb3.oneself_diff * 2
            self.bb2.sum()
        return

    def score_bb3_get_roi(self, fill_mark, fillarea_mark):
        fillarea_copy = fillarea_mark.copy()
        fill_copy = fill_mark.copy()
        img, contours, hierarchy = cv2.findContours(fill_copy,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col_f, row_f, w_f, h_f = cv2.boundingRect(maxcnt)
        else:
            return
        img, contours, hierarchy = cv2.findContours(fillarea_copy,
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

        site_k_list = []
        site_list = []
        for i in range(0, len(contours)):
            col, row, w, h = cv2.boundingRect(contours[i])
            col_k = abs((col_f - col) / w_f)
            row_k = abs((row_f - row) / h_f)
            w_k = abs(w / w_f)
            h_k = abs(h / h_f)
            site_k_list.append((col_k, row_k, w_k, h_k))
            site_list.append((col, row, w, h))

            # 可视化
            # cv2.rectangle(fillarea_copy, (col, row), (col + w, row + h), 255,
            #               1)
            # cv2.imshow("mark_rectangle", fillarea_copy)

        return site_k_list

    def score_bb3(self, src_image, fill_mark, other_mark, fillarea_mark,
                  operation_time):
        if operation_time == '术前':
            return
        elif operation_time == '术中':
            self.roi_site_k_list = self.score_bb3_get_roi(
                fill_mark, fillarea_mark)
            return

        if self.roi_site_k_list == 0:
            return

        if len(self.roi_site_k_list) == 0:
            self.bb3.other_diff = np.random.randint(9, 10)
            self.bb3.oneself_diff = np.random.randint(9, 10)
            self.bb3.sum()
            self.bb2.grade = 18
            return

        hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
        img_rows, img_cols = hsv_image.shape[:2]
        H, S, V = cv2.split(hsv_image)
        B, G, R = cv2.split(src_image)
        # cv2.imshow('H', H)
        # cv2.imshow('S', S)

        # gray_img = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray_img', gray_img)

        fillarea_mark = np.zeros((img_rows, img_cols), np.uint8)
        otherarea_mark = np.zeros((img_rows, img_cols), np.uint8)

        for i in range(0, len(self.roi_site_k_list)):
            col_k, row_k, w_k, h_k = self.roi_site_k_list[i]

            # 获得补牙及周围区域
            img, contours, hierarchy = cv2.findContours(
                fill_mark.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
                col, row, w, h = cv2.boundingRect(maxcnt)
                roi_col = int(col + col_k * w)
                roi_row = int(row + row_k * h)
                roi_w = int(w_k * w)
                roi_h = int(h_k * h)

                # 所补区域
                fillarea_mark[roi_row:roi_row + roi_h, roi_col:roi_col +
                              roi_w] = 255
                fillarea_mark[fill_mark == 0] = 0
            else:
                self.bb3.other_diff = 9
                self.bb3.oneself_diff = 9
                self.bb3.sum()
                self.bb2.grade = 18
                return

            # 获得邻牙区域
            img_other, contours_other, hierarchy_other = cv2.findContours(
                other_mark.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_other:
                maxcnt = max(contours_other, key=lambda x: cv2.contourArea(x))
                col, row, w, h = cv2.boundingRect(maxcnt)
                roi_col_other = int(col + col_k * w)
                roi_row_other = int(row + row_k * h)
                roi_w_other = int(w_k * w)
                roi_h_other = int(h_k * h)

                # 邻牙区域
                otherarea_mark[roi_row_other:roi_row_other +
                               roi_h_other, roi_col_other:roi_col_other +
                               roi_w_other] = 255
                otherarea_mark[other_mark == 0] = 0
            else:
                self.bb3.other_diff = 5
                self.bb3.sum()
                return

        # 所补区域周围区域
        oneself_around_mark = my_erode_dilate(fillarea_mark, 0, 5, (10, 10))
        oneself_around_mark[fillarea_mark == 255] = 0
        oneself_around_mark[fill_mark == 0] = 0

        # 位置矩形框可视化
        fillarea_mark_show = bin_to_rgb(src_image, fillarea_mark)
        otherarea_mark_show = bin_to_rgb(src_image, otherarea_mark)
        oneself_around_mark_show = bin_to_rgb(src_image, oneself_around_mark)
        # cv2.imshow("fillarea_mark", fillarea_mark_show)
        # cv2.imshow("otherarea_mark", otherarea_mark_show)
        # cv2.imshow("oneself_around_mark", oneself_around_mark_show)

        fill_b_avr = np.mean(B[fillarea_mark != 0])
        fill_g_avr = np.mean(G[fillarea_mark != 0])
        fill_r_avr = np.mean(R[fillarea_mark != 0])

        other_b_avr = np.mean(B[otherarea_mark != 0])
        other_g_avr = np.mean(G[otherarea_mark != 0])
        other_r_avr = np.mean(R[otherarea_mark != 0])

        oneself_other_b_avr = np.mean(B[oneself_around_mark == 255])
        oneself_other_g_avr = np.mean(G[oneself_around_mark == 255])
        oneself_other_r_avr = np.mean(R[oneself_around_mark == 255])

        fill_h_avr = np.mean(H[fillarea_mark != 0])
        fill_s_avr = np.mean(S[fillarea_mark != 0])
        fill_v_avr = np.mean(V[fillarea_mark != 0])

        other_h_avr = np.mean(H[otherarea_mark != 0])
        other_s_avr = np.mean(S[otherarea_mark != 0])
        other_v_avr = np.mean(V[otherarea_mark != 0])

        oneself_other_h_avr = np.mean(H[oneself_around_mark == 255])
        oneself_other_s_avr = np.mean(S[oneself_around_mark == 255])
        oneself_other_v_avr = np.mean(V[oneself_around_mark == 255])

        # 根据均值方差的差值评分，差值越大分越低
        if self.print_flag == 1:
            print('BB3[INFO] 相邻牙齿色差', abs(fill_h_avr - other_h_avr),
                  abs(fill_s_avr - other_s_avr))
            print('BB3[INFO] 自己牙齿色差', abs(fill_h_avr - oneself_other_h_avr),
                  abs(fill_s_avr - oneself_other_s_avr))
        h_avr = my_limit(
            5 - (abs(fill_h_avr - other_h_avr) / self.bb3.MAX_AVR_DIFF_H) * 5,
            0, 5)
        s_avr = my_limit(
            5 - (abs(fill_s_avr - other_s_avr) / self.bb3.MAX_AVR_DIFF_S) * 5,
            0, 5)
        oneself_h_avr = my_limit(
            5 -
            (abs(fill_h_avr - oneself_other_h_avr) / self.bb3.MAX_AVR_DIFF_H) *
            5, 0, 5)
        oneself_s_avr = my_limit(
            5 -
            (abs(fill_s_avr - oneself_other_s_avr) / self.bb3.MAX_AVR_DIFF_S) *
            5, 0, 5)

        if self.print_lr_x_flag == 1:
            self.lr_x_str[1] = str(
                abs(fill_b_avr - oneself_other_b_avr)) + ',' + str(
                    abs(fill_g_avr - oneself_other_g_avr)) + ',' + str(
                        abs(fill_r_avr - oneself_other_r_avr))
            print(self.lr_x_str[1])
            self.lr_x_str[2] = self.lr_x_str[1] + ',' + str(
                abs(fill_b_avr - other_b_avr)) + ',' + str(
                    abs(fill_g_avr - other_g_avr)) + ',' + str(
                        abs(fill_r_avr - other_r_avr))
            print(self.lr_x_str[2])

        self.bb3.other_diff = h_avr + s_avr
        self.bb3.oneself_diff = oneself_h_avr + oneself_s_avr
        self.roi_site_k_list = 0
        self.bb3.sum()

        if self.lr_flag == 1:
            lr3 = joblib.load(TXT_DIR + 'bb3.pkl')
            bb3x = np.array([[
                abs(fill_b_avr - oneself_other_b_avr),
                abs(fill_g_avr - oneself_other_g_avr),
                abs(fill_r_avr - oneself_other_r_avr),
                abs(fill_b_avr - other_b_avr),
                abs(fill_g_avr - other_g_avr),
                abs(fill_r_avr - other_r_avr)
            ]])
            self.bb3.grade = int(lr3.predict(bb3x)[0][0])
            self.bb3.grade = my_limit(self.bb3.grade, 10, 19)
            lr2 = joblib.load(TXT_DIR + 'bb2.pkl')
            bb2x = np.array([[
                abs(fill_b_avr - oneself_other_b_avr),
                abs(fill_g_avr - oneself_other_g_avr),
                abs(fill_r_avr - oneself_other_r_avr)
            ]])
            self.bb2.grade = int(lr2.predict(bb2x)[0][0])
            self.bb2.grade = my_limit(self.bb2.grade, 10, 19)
            # print(self.bb3.grade)

        return

    def score_bb4(self, src_gray_img, fill_mark, operation_time,
                  fillteeth_type, fillteeth_num):
        if fillteeth_type == '门牙':
            self.bb4.grade = self.bb4.GAP_SCORE
            return
        elif operation_time == '术中':
            return
        elif operation_time == '术前':
            return
        # 选取对应模版
        if fillteeth_num == '6' or fillteeth_num == '7':
            std_img = cv2.imdecode(
                np.fromfile(BB4_STANDARD_IMGDIR + 'test_6-7.png',
                            dtype=np.uint8), -1)
        elif fillteeth_num == '4' or fillteeth_num == '5':
            std_img = cv2.imdecode(
                np.fromfile(BB4_STANDARD_IMGDIR + 'test_4-5.png',
                            dtype=np.uint8), -1)
        else:
            std_img = cv2.imdecode(
                np.fromfile(BB4_STANDARD_IMGDIR + 'test_4-5.png',
                            dtype=np.uint8), -1)

        img, contours, hierarchy = cv2.findContours(fill_mark.copy(),
                                                    cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col, row, w, h = cv2.boundingRect(maxcnt)

            fill_gray = src_gray_img[row:row + h, col:col + w]
            fill_bin = fill_mark[row:row + h, col:col + w]
            fill_gray = cv2.resize(fill_gray, (100, 100))
            fill_bin = cv2.resize(fill_bin, (100, 100))
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            fill_bin = cv2.erode(fill_bin, element)

            #Sobel边缘检测
            dst = cv2.fastNlMeansDenoising(fill_gray,
                                           h=5,
                                           templateWindowSize=7,
                                           searchWindowSize=21)
            # cv2.imshow("dst",dst)
            sobelX = cv2.Sobel(dst, cv2.CV_64F, 1, 0)  #x方向的梯度
            sobelY = cv2.Sobel(dst, cv2.CV_64F, 0, 1)  #y方向的梯度
            sobelX = np.uint8(np.absolute(sobelX))  #x方向梯度的绝对值
            sobelY = np.uint8(np.absolute(sobelY))  #y方向梯度的绝对值
            sobelCombined = cv2.bitwise_or(sobelX, sobelY)  #
            # cv2.imshow("Sobel Combined", sobelCombined)
            #Canny边缘检测
            # edge_output = cv2.Canny(fill_gray, 40, 120)

            grain_show = np.zeros(fill_gray.shape[0:2], dtype=np.uint8)
            for r in range(fill_gray.shape[0]):
                for c in range(fill_gray.shape[1]):
                    if fill_bin[r][c] == 255:
                        grain_show[r][c] = sobelCombined[r, c]

            fill_canny = np.zeros((200, 200), dtype=np.uint8)
            fill_canny[50:150, 50:150] = grain_show
            rows, cols = fill_canny.shape[0:2]

            # 旋转90度，再次匹配，取最大值
            max_val_list = []
            for i in range(10):
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 36 * i, 1)
                dst = cv2.warpAffine(fill_canny, M, (cols, rows))
                res = cv2.matchTemplate(dst, std_img, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                max_val_list.append(max_val)
            max_val = np.max(max_val_list)
            if self.print_flag == 1:
                print("BB4[INFO]:沟壑匹配度", max_val)
            # # 纹路可视化
            # cv2.imshow("std_img", grain_show)
            # cv2.setMouseCallback('std_img', draw_num, grain_show)
            # key = 0
            # while key != ord('s'):
            #     key = cv2.waitKey(0)
            #     if key == 27:
            #         cv2.destroyAllWindows()
            #         return
            # io.imsave("./BB4_standard_template/test.png",grain_show)

            if max_val >= self.bb4.GAP_SUBTRACT_START:
                texture_scores = self.bb4.GAP_SCORE
            else:
                texture_scores = self.bb4.GAP_SCORE - int(
                    ((self.bb4.GAP_SUBTRACT_START - max_val) /
                     self.bb4.GAP_SUBTRACT_RATIO) * self.bb4.GAP_SUBTRACT)

            self.bb4.gap = texture_scores
        self.bb4.sum()
        return

    def score_all(self, teeth_pro, output_to_txt=False):
        self.score_aa1(teeth_pro.neighbor_flag,
                       teeth_pro.img_info.fillteeth_num)
        is_ok = self.score_aa2(teeth_pro.dst_all_mark, teeth_pro.fill_rect,
                               teeth_pro.img_info.operation_time,
                               teeth_pro.img_info.fillteeth_num)
        if is_ok == 0:
            self.grade = self.aa1.grade + self.aa2.grade + self.aa3.grade
            self.grade += self.bb1.grade + self.bb2.grade + self.bb3.grade + self.bb4.grade
            # self.creat_score_txt(teeth_pro.img_info)
            return 0

        self.score_bb1(teeth_pro.src_gray_image,
                       teeth_pro.dst_fillarea_mark,
                       teeth_pro.img_info.operation_time,
                       teeth_type=teeth_pro.img_info.fillteeth_type,
                       rgb_img=teeth_pro.src_image)
        self.score_bb3(teeth_pro.src_image, teeth_pro.dst_fill_mark,
                       teeth_pro.dst_other_mark, teeth_pro.dst_fillarea_mark,
                       teeth_pro.img_info.operation_time)
        self.score_bb2(teeth_pro.dst_fill_mark,
                       teeth_pro.img_info.operation_time)
        self.score_bb4(teeth_pro.src_gray_image, teeth_pro.dst_fill_mark,
                       teeth_pro.img_info.operation_time,
                       teeth_pro.img_info.fillteeth_type,
                       teeth_pro.img_info.fillteeth_num)

        self.grade = self.aa1.grade + self.aa2.grade + self.aa3.grade
        self.grade += self.bb1.grade + self.bb2.grade + self.bb3.grade + self.bb4.grade

        if output_to_txt:
            self.creat_score_txt(teeth_pro.img_info)
        # else:
        #     if teeth_pro.img_info.operation_time == "术前":
        #         self.str_score2cmd[0] = str(self.aa1.grade) + " " + str(self.aa2.grade) + " " + str(self.aa3.grade) + " " + \
        #                             str(self.bb1.grade) + " " + str(self.bb2.grade) + " " + str(self.bb3.grade) + " " + \
        #                             str(self.bb4.grade) + " " + str(self.grade)
        #     if teeth_pro.img_info.operation_time == "术中":
        #         self.str_score2cmd[1] = str(self.aa1.grade) + " " + str(self.aa2.grade) + " " + str(self.aa3.grade) + " " + \
        #                                 str(self.bb1.grade) + " " + str(self.bb2.grade) + " " + str(self.bb3.grade) + " " + \
        #                                 str(self.bb4.grade) + " " + str(self.grade)
        #     if teeth_pro.img_info.operation_time == "术后":
        #         self.str_score2cmd[2] = str(self.aa1.grade) + " " + str(self.aa2.grade) + " " + str(self.aa3.grade) + " " + \
        #                                 str(self.bb1.grade) + " " + str(self.bb2.grade) + " " + str(self.bb3.grade) + " " + \
        #                                 str(self.bb4.grade) + " " + str(self.grade)
        #         print("aabb", "#", teeth_pro.img_info.patient_name, "#",
        #               self.str_score2cmd[0], "#", self.str_score2cmd[1], "#",
        #               self.str_score2cmd[2])

        # print score info
        if self.print_flag == 1:
            self.print()

        if self.print_lr_x_flag == 1 and teeth_pro.img_info.operation_time == "术后":
            f = open(os.path.join('D:/WorkingFolder/Python/LinearRegression',
                                  'lr_x.txt'),
                     'a',
                     encoding='utf-8')
            f.write(teeth_pro.img_info.patient_name + " ")
            f.write(self.lr_x_str[0] + " ")
            f.write(self.lr_x_str[1] + " ")
            f.write(self.lr_x_str[2] + " ")
            f.write("\n")
            f.close()
        return 1

    def getScores(self):
        return (self.aa1.grade, self.aa2.grade, self.aa3.grade, self.bb1.grade,
                self.bb2.grade, self.bb3.grade, self.bb4.grade)

    def print(self):
        self.aa1.print()
        self.aa2.print()
        self.aa3.print()
        self.bb1.print()
        self.bb2.print()
        self.bb3.print()
        self.bb4.print()

    def creat_score_txt(self, img_info):
        if os.access(os.path.join(img_info.imgfloder_path, 'score.txt'),
                     os.F_OK):
            f = open(os.path.join(img_info.imgfloder_path, 'score.txt'),
                     'a',
                     encoding='utf-8')
            f.write("\n")
            f.write(img_info.patient_name + "-" + img_info.operation_time +
                    "-")
            f.write(img_info.fillteeth_type + "-" + img_info.doctor_name + "-")
            f.write(
                str(self.aa1.grade) + "-" + str(self.aa2.grade) + "-" +
                str(self.aa3.grade) + "-")
            f.write(
                str(self.bb1.grade) + "-" + str(self.bb2.grade) + "-" +
                str(self.bb3.grade) + "-")
            f.write(str(self.bb4.grade) + "-" + str(self.grade))
        else:
            print("首次，需创建TXT文件")
            f = open(os.path.join(img_info.imgfloder_path, 'score.txt'),
                     'w',
                     encoding='utf-8')
            f.write("0xaaee0xaaff0x55550xa5a5" + "\n")
            f.write(img_info.patient_name + "-" + img_info.operation_time +
                    "-")
            f.write(img_info.fillteeth_type + "-" + img_info.doctor_name + "-")
            f.write(
                str(self.aa1.grade) + "-" + str(self.aa2.grade) + "-" +
                str(self.aa3.grade) + "-")
            f.write(
                str(self.bb1.grade) + "-" + str(self.bb2.grade) + "-" +
                str(self.bb3.grade) + "-")
            f.write(str(self.bb4.grade) + "-" + str(self.grade))
        f.close()
        return


def draw_num(event, x, y, flags, param):
    global label_flag
    pen_w = 5
    pt = (x, y)
    prev_pt = pt
    if event == cv2.EVENT_LBUTTONDOWN:
        label_flag = 1  # 左键按下
    elif event == cv2.EVENT_LBUTTONUP:
        label_flag = 0  # 左键放开
        # prev_pt = None
    if label_flag == 1:
        param[y - pen_w:y + pen_w, x - pen_w:x + pen_w] = 0
        # print(y, x)
    cv2.imshow('std_img', param)


def print_value(event, x, y, flags, param):
    global label_flag
    pen_w = 5
    pt = (x, y)
    prev_pt = pt
    if event == cv2.EVENT_LBUTTONDOWN:
        label_flag = 1  # 左键按下
    elif event == cv2.EVENT_LBUTTONUP:
        label_flag = 0  # 左键放开
        # prev_pt = None
    if label_flag == 1:
        print(param[y, x])


# / *将二值化图映射到原图 * /
def bin_to_rgb(rgb_img, bin_img):
    img_rows, img_cols = bin_img.shape[:2]
    re_dst_image = np.zeros(rgb_img.shape, dtype=np.uint8)

    re_dst_image[bin_img == 255] = rgb_img[bin_img == 255]

    return re_dst_image