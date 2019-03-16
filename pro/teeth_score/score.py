#!/usr/bin/python3.6
# coding=utf-8

import cv2
import numpy as np
import filetype
import math
import os
from teeth import my_limit
from indicators import *


class Teeth_Grade():
    def __init__(self):
        self.aa1 = Indicators_AA1()
        self.aa2 = Indicators_AA2()
        self.aa3 = Indicators_AA3()
        self.bb1 = Indicators_BB1()
        self.bb2 = Indicators_BB2()
        self.bb3 = Indicators_BB3()
        self.bb4 = Indicators_BB4()
        self.grade = 0

    def clear(self):
        self.aa1.clear()
        self.aa2.clear()
        self.aa3.clear()
        self.bb1.clear()
        self.bb3.clear()
        self.bb4.clear()
        self.grade = 0

    def score_aa1(self, dst_all_mark, site, radius):
        key_elements_score = self.aa1.CONTAINS_NEIGHBOR_SCORE
        self.aa1.neighbor_num = 0

        src_row, src_col = dst_all_mark.shape[:2]

        # 坐标转换
        min_row = int(my_limit(site[0] - radius, 0, src_row))
        max_row = int(my_limit(site[0] + radius, 0, src_row))
        min_col = int(my_limit(site[1] - radius, 0, src_col))
        max_col = int(my_limit(site[1] + radius, 0, src_col))

        # 区域可视化
        # roi = dst_all_mark[min_row:max_row, min_col:max_col]
        # cv2.imshow("aa1_roi", roi)

        roi_height = max_row - min_row
        roi_width = max_col - min_col

        # 统计患牙坐标周围第一列和最后一列白点数
        left_tooth_point_num = 0
        right_tooth_point_num = 0
        for r in range(src_row):
            if dst_all_mark[r][min_col] == 255:
                left_tooth_point_num += 1
            if dst_all_mark[r][max_col-1] == 255:
                right_tooth_point_num += 1

        # 参照牙存在情况:1-有牙 2-不完整 3-最里侧
        # 认为一侧白点超过一定数量,并且原始图像感兴趣区域左侧/右侧还有较大空间,则这一侧有牙
        if left_tooth_point_num > roi_height * self.aa1.THR_HEIGHT:
            if min_col > roi_width * self.aa1.THR_WIDTH:
                left_tooth_status = 1
                self.aa1.neighbor_num += 1
                print('AA1[INFO] 左侧有牙')
            else:
                left_tooth_status = 2
                print('AA1[INFO] 左侧有牙，不完整,扣3分')
                key_elements_score -= 3  # 左侧无参照,扣3分
        else:
            left_tooth_status = 3
            print('AA1[INFO] 左侧无牙')

        if right_tooth_point_num > roi_height * self.aa1.THR_HEIGHT:
            if src_col - max_col > roi_width * self.aa1.THR_WIDTH:
                right_tooth_status = 1
                self.aa1.neighbor_num += 1
                print('AA1[INFO] 右侧有牙')
            else:
                right_tooth_status = 2
                print('AA1[INFO] 右侧有牙，不完整,扣3分')
                key_elements_score -= 3  # 右侧无参照,扣3分
        else:
            right_tooth_status = 3
            print('AA1[INFO] 右侧无牙')

        if left_tooth_status == 3 and right_tooth_status == 3:
            key_elements_score = 1              # 左右都无参照牙齿
            self.aa1.neighbor_num = 0
            print('AA1[INFO] 左右都无参照牙齿,0分')
        elif (left_tooth_status == 3 and right_tooth_status == 2) or \
             (left_tooth_status == 2 and right_tooth_status == 3):
            key_elements_score = 1              # 只有单侧牙，且不完整
            self.aa1.neighbor_num = 0
            print('AA1[INFO] 只有单侧牙，且不完整,1分')

        self.aa1.contains_neighbor = key_elements_score
        self.aa1.undefined = 3          # 直接给3分
        self.aa1.sum()
        return 1

    def score_aa2(self, dst_all_mark, dst_fill_mark, site, operation_time):
        img_row, img_col = dst_all_mark.shape[:2]
        print('AA2[INFO]：图片大小', img_row, img_col)
        img_center_row = img_row / 2
        img_center_col = img_col / 2

        # 中心位置偏差评分
        # 患牙与图片中心点距离
        distance = math.sqrt((img_center_row - float(site[0])) ** 2 + (img_center_col - float(site[1])) ** 2)

        # 距离占x的比例
        ratio = distance / math.sqrt(img_col**2 + img_row**2)
        print('AA2[INFO] 图片中心距离的比例', ratio)
        if operation_time == '术前':  # 术前
            full_scores = self.aa2.CENTER_BIAS_SCORE_FIRST       # 满分
            cut_scores = self.aa2.CENTER_BIAS_SUBTRACT_FIRST     # 按比例扣分
        else:                    # 术中和术后
            full_scores = self.aa2.CENTER_BIAS_SCORE_OTHER
            cut_scores = self.aa2.CENTER_BIAS_SUBTRACT_OTHER
        if ratio > self.aa2.CENTER_BIAS_SUBTRACT_START:
            # 考虑四舍五入， 偏10%~15%扣1.5分，以此类推，扣完为止
            center_point_scores = full_scores - int((ratio-self.aa2.CENTER_BIAS_SUBTRACT_START)/self.aa2.CENTER_BIAS_SUBTRACT_RATIO+0.5) * cut_scores
            center_point_scores = my_limit(center_point_scores, 0, full_scores)
        else:
            center_point_scores = full_scores

        self.aa2.center_bias = center_point_scores

        # 面积大小占比评分
        # 标记滤波,查找最大外轮廓
        _, contours, hierarchy = cv2.findContours(dst_fill_mark.copy(),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            area_fill = cv2.contourArea(maxcnt)

            # print("所补牙具有多少颗相邻牙：", self.aa1.neighbor_num)
            pic_area = img_row * img_col  # 图片整体面积
            area_ratio = area_fill * (self.aa1.neighbor_num+1) / pic_area  # 面积比例
            print('AA2[INFO] 面积占比', area_ratio)

            if self.aa2.AREA_RATIO_SUBTRACT_START_MIN <= area_ratio <= self.aa2.AREA_RATIO_SUBTRACT_START_MAX:
                ratio_diff = 0
            elif area_ratio < self.aa2.AREA_RATIO_SUBTRACT_START_MIN:
                ratio_diff = self.aa2.AREA_RATIO_SUBTRACT_START_MIN - area_ratio
            else:
                ratio_diff = area_ratio - self.aa2.AREA_RATIO_SUBTRACT_START_MAX

            self.aa2.area_ratio = my_limit(self.aa2.AREA_RATIO_SCORE - 
                                           math.ceil(ratio_diff/self.aa2.AREA_RATIO_SUBTRACT_RATIO)*self.aa2.AREA_RATIO_SUBTRACT,
                                           0, self.aa2.AREA_RATIO_SCORE)  # 按比例扣分
            self.aa1.contains_neighbor *= (self.aa2.area_ratio/self.aa2.AREA_RATIO_SCORE) * self.aa1.AREA_K
            self.aa1.sum()

        # 提取全部牙齿轮廓，利用线性拟合一条直线用来判断角度
        _, contours, hierarchy = cv2.findContours(dst_all_mark.copy(),
                                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))

            vx, vy, x, y = cv2.fitLine(maxcnt, cv2.DIST_L2, 0, 0.01, 0.01)
            print('AA2[INFO] 角度', math.atan(vy/vx) / math.pi * 180)

            # 角度可视化
            # lefty = int((-x * vy / vx) + y)
            # righty = int(((dst_all_mark.shape[1] - x) * vy / vx) + y)
            # img = cv2.line(dst_all_mark, (dst_all_mark.shape[1] - 1, righty), (0, lefty), (0, 255, 0), 2)
            # cv2.imshow("Canvas", img)

            if operation_time == '术前':
                self.aa2.first_angle = math.atan(vy/vx) / math.pi * 180
                self.aa2.shooting_angle = self.aa2.SHOOTING_ANGLE_SCORE_FIRST
            else:
                img_angle = math.atan(vy/vx) / math.pi * 180
                dif_angle = abs(img_angle - self.aa2.first_angle)
                print('AA2[INFO] 拍摄角度之差', dif_angle)
                self.aa2.shooting_angle = my_limit(self.aa2.SHOOTING_ANGLE_SCORE_OTHER - 
                                                  (dif_angle//self.aa2.SHOOTING_ANGLE_SUBTRACT_ANGLE)*self.aa2.SHOOTING_ANGLE_SUBTRACT,
                                                  0, self.aa2.SHOOTING_ANGLE_SCORE_OTHER)  # 按比例扣分

        if self.aa2.area_ratio == 0:
            return 0

        self.aa2.sum()
        return 1

    def score_aa3(self, img_path):
        # 获取文件类型
        file_kind = filetype.guess(img_path)

        # 不是jpg文件，直接0分
        print('AA3[INFO] 文件类型', file_kind.extension)
        if file_kind.extension != 'jpg':
            self.aa3.img_type = 0
            return 0
        else:
            self.aa3.img_type = self.aa3.IMG_TYPE_SCORE

        img = cv2.imread(img_path, 0)
        img_row, img_col = img.shape[:2]

        # 照片比例16：9（在3%的误差范围内4分，必要条件，达不到则AA3直接0分）绝对误差
        print('AA3[INFO] 照片比例（标准1.77）', (img_col / img_row))
        if abs((img_col / img_row) - self.aa3.IMG_RATIO_STD) > self.aa3.IMG_RATIO_ERROR:
            self.aa3.img_ratio = 0
            # return 0
        else:
            self.aa3.img_ratio = self.aa3.IMG_RATIO_SCORE

        # 分辨率得分
        print('AA3[INFO] 分辨率（标准1049088）', (img_col * img_row))
        if img_col * img_row >= self.aa3.IMG_RESOLUTION_SUBTRACT_START:
            self.aa3.img_resolution = self.aa3.IMG_RESOLUTION_SCORE
        else:
            resolution_diff = self.aa3.IMG_RESOLUTION_SUBTRACT_START - img_col * img_row
            self.aa3.img_resolution = my_limit(self.aa3.IMG_RESOLUTION_SCORE - 
                                               math.ceil(resolution_diff/self.aa3.IMG_RESOLUTION_SUBTRACT_SIZE)*self.aa3.IMG_RESOLUTION_SUBTRACT,
                                               0, self.aa3.IMG_RESOLUTION_SCORE)  # 按比例扣分

        self.aa3.sum()
        return 1

    def score_bb1(self, gray_img, mark_img, operation_time):
        if operation_time == '术前':
            return
        if operation_time == '术后':
            self.bb1.grade = self.bb1.BLACK_DEPTH_SCORE + self.bb1.BLACK_SIZE_SCORE
            return
        caries_point_num = 0                                 # 黑点个数
        caries_point_thresh = self.bb1.THR_GRAY              # 灰度阈值
        black_level_score = self.bb1.BLACK_DEPTH_SCORE       # 黑色深浅得分
        black_num_score = self.bb1.BLACK_SIZE_SCORE          # 黑色大小得分
        min_gray_value = 255                                 # 牙齿区域最小灰度值

        (height, width) = gray_img.shape[0:2]
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mark_img = cv2.erode(mark_img, element)
        # 黑色显示模版
        black_point_show = np.zeros((height, width), dtype=np.uint8)
        for r in range(height):
            for c in range(width):
                # 分割后的牙齿区域中灰度值小于阈值的像素
                if gray_img[r][c] < caries_point_thresh and mark_img[r][c] == 255:
                    black_point_show[r][c] = 255
                    caries_point_num += 1
                    if gray_img[r][c] < min_gray_value:  # 获取最低灰度值
                        min_gray_value = gray_img[r][c]
        # 可视化龋齿黑点
        # cv2.imshow('black_point_show', black_point_show)

        # 根据黑点数量计算黑色大小得分
        # 每100个黑点减一分,需根据输入图片大小调整
        print('BB1[INFO] 黑点数量', caries_point_num)
        black_num_score -= ((caries_point_num + self.bb1.BLACK_SIZE_SUBTRACT_VALUE/2) // self.bb1.BLACK_SIZE_SUBTRACT_VALUE) * \
                             self.bb1.BLACK_SIZE_SUBTRACT
        if black_num_score <= 0:
            black_num_score = 0

        # 根据最低灰度值计算黑色深浅得分
        print('BB1[INFO] 最小灰度值', min_gray_value)
        if min_gray_value < 255:
            black_level_score -= ((caries_point_thresh - min_gray_value) // self.bb1.BLACK_DEPTH_SUBTRACT_VALUE) \
                                   * self.bb1.BLACK_DEPTH_SUBTRACT  # 每低于阈值灰度值扣一分
            if black_level_score <= 0:
                black_level_score = 0

        self.bb1.black_depth = black_level_score
        self.bb1.black_size = black_num_score
        self.bb1.sum()
        return

    def score_bb2(self, fill_mark, operation_time):
        if operation_time == '术前':
            return
        elif operation_time == '术后':
            self.bb2.grade = 20
            return

        img, contours, hierarchy = cv2.findContours(fill_mark.copy(),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col, row, w, h = cv2.boundingRect(maxcnt)

            fill_bin = fill_mark[row:row + h, col:col + w]
            fill_bin = cv2.resize(fill_bin, (100, 100))

            test = np.zeros(fill_bin.shape, np.uint8)

            dst = cv2.cornerHarris(fill_bin, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)                   # 图像膨胀
            test[dst > 0.3*dst.max()] = 255

            # 角点可视化
            # cv2.imshow("test", test)

            img, contours, hierarchy = cv2.findContours(test,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            print("BB2[INFO] 角点个数：", len(contours))
            self.bb2.edge_shape = 10 - my_limit(len(contours) - 4, 0, 10)
            self.bb2.black_size = 5
        self.bb2.sum()
        return

    def score_bb3(self, src_image, fill_mark, other_mark, operation_time):
        if operation_time == '术前':
            return
        elif operation_time == '术中':
            return

        hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)
        img_rows, img_cols = hsv_image.shape[:2]

        fill_count_h = [0 for x in range(256)]
        fill_count_s = [0 for x in range(256)]
        fill_count_v = [0 for x in range(256)]
        fill_count_hsv = 0
        fill_sum_h = 0
        fill_sum_s = 0
        fill_sum_v = 0
        other_count_h = [0 for x in range(256)]
        other_count_s = [0 for x in range(256)]
        other_count_v = [0 for x in range(256)]
        other_count_hsv = 0
        other_sum_h = 0
        other_sum_s = 0
        other_sum_v = 0

        for r in range(0, img_rows):
            for c in range(0, img_cols):
                if fill_mark[r, c] != 0:
                    fill_count_hsv += 1
                    fill_count_h[hsv_image[r, c][0]] += 1
                    fill_count_s[hsv_image[r, c][1]] += 1
                    fill_count_v[hsv_image[r, c][2]] += 1
                    fill_sum_h += hsv_image[r, c][0]
                    fill_sum_s += hsv_image[r, c][1]
                    fill_sum_v += hsv_image[r, c][2]
                if other_mark[r, c] != 0:
                    other_count_hsv += 1
                    other_count_h[hsv_image[r, c][0]] += 1
                    other_count_s[hsv_image[r, c][1]] += 1
                    other_count_v[hsv_image[r, c][2]] += 1
                    other_sum_h += hsv_image[r, c][0]
                    other_sum_s += hsv_image[r, c][1]
                    other_sum_v += hsv_image[r, c][2]

        # 计算均值
        fill_h_avr = fill_sum_h / fill_count_hsv
        fill_s_avr = fill_sum_s / fill_count_hsv
        fill_v_avr = fill_sum_v / fill_count_hsv
        other_h_avr = other_sum_h / other_count_hsv
        other_s_avr = other_sum_s / other_count_hsv
        other_v_avr = other_sum_v / other_count_hsv

        # 计算方差
        fill_h_var = 0
        fill_s_var = 0
        fill_v_var = 0
        other_h_var = 0
        other_s_var = 0
        other_v_var = 0

        for j in range(256):
            fill_h_var += fill_count_h[j] * (j - fill_h_avr) * (j - fill_h_avr)
            fill_s_var += fill_count_s[j] * (j - fill_s_avr) * (j - fill_s_avr)
            fill_v_var += fill_count_v[j] * (j - fill_v_avr) * (j - fill_v_avr)
            other_h_var += other_count_h[j] * (j - other_h_avr) * (j - other_h_avr)
            other_s_var += other_count_s[j] * (j - other_s_avr) * (j - other_s_avr)
            other_v_var += other_count_v[j] * (j - other_v_avr) * (j - other_v_avr)

        fill_h_var = fill_h_var / fill_count_hsv
        fill_s_var = fill_s_var / fill_count_hsv
        fill_v_var = fill_v_var / fill_count_hsv
        other_h_var = other_h_var / other_count_hsv
        other_s_var = other_s_var / other_count_hsv
        other_v_var = other_v_var / other_count_hsv

        # 根据均值方差的差值评分，差值越大分越低
        print('BB3[INFO] 均值差', abs(fill_h_avr-other_h_avr), abs(fill_s_avr-other_s_avr), (fill_v_avr-other_v_avr))
        print('BB3[INFO] 方差差', abs(fill_h_var - other_h_var), abs(fill_s_var - other_s_var), abs(fill_v_var - other_v_var))
        self.bb3.h_avr = my_limit(7*self.bb3.AVR_K-(abs(fill_h_avr-other_h_avr)/self.bb3.MAX_AVR_DIFF_H)*7*self.bb3.AVR_K, 0, 7*self.bb3.AVR_K)
        self.bb3.s_avr = my_limit(7*self.bb3.AVR_K-(abs(fill_s_avr-other_s_avr)/self.bb3.MAX_AVR_DIFF_S)*7*self.bb3.AVR_K, 0, 7*self.bb3.AVR_K)
        self.bb3.v_avr = my_limit(6*self.bb3.AVR_K-(abs(fill_v_avr-other_v_avr)/self.bb3.MAX_AVR_DIFF_V)*6*self.bb3.AVR_K, 0, 6*self.bb3.AVR_K)

        self.bb3.h_var = my_limit(7*(1-self.bb3.AVR_K)-(abs(fill_h_var - other_h_var)/self.bb3.MAX_VAR_DIFF_H)*7*(1-self.bb3.AVR_K), 0, 7*(1-self.bb3.AVR_K))
        self.bb3.s_var = my_limit(7*(1-self.bb3.AVR_K)-(abs(fill_s_var - other_s_var)/self.bb3.MAX_VAR_DIFF_S)*7*(1-self.bb3.AVR_K), 0, 7*(1-self.bb3.AVR_K))
        self.bb3.v_var = my_limit(6*(1-self.bb3.AVR_K)-(abs(fill_v_var - other_v_var)/self.bb3.MAX_VAR_DIFF_V)*6*(1-self.bb3.AVR_K), 0, 6*(1-self.bb3.AVR_K))
        # print(abs(fill_h_var - other_h_var))
        self.bb3.sum()
        return

    def score_bb4(self, src_gray_img, fill_mark, operation_time, operation_name):
        if operation_name == '门牙':
            self.bb4.grade = self.bb4.GAP_SCORE
            return
        elif operation_time == '术中':
            return
        elif operation_time == '术前':
            return

        img, contours, hierarchy = cv2.findContours(fill_mark.copy(),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            maxcnt = max(contours, key=lambda x: cv2.contourArea(x))
            col, row, w, h = cv2.boundingRect(maxcnt)

            fill_gray = src_gray_img[row:row+h, col:col+w]
            fill_bin = fill_mark[row:row + h, col:col + w]
            fill_gray = cv2.resize(fill_gray, (100, 100))
            fill_bin = cv2.resize(fill_bin, (100, 100))
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            fill_bin = cv2.erode(fill_bin, element)

            edge_output = cv2.Canny(fill_gray, 40, 120)
            src_gray_output = cv2.Canny(src_gray_img, 40, 120)

            grain_show = np.zeros(fill_gray.shape[0:2], dtype=np.uint8)
            gap_point_num = 0
            for r in range(fill_gray.shape[0]):
                for c in range(fill_gray.shape[1]):
                    if fill_bin[r][c] == 255 and edge_output[r][c] == 255:
                        grain_show[r][c] = 255
                        gap_point_num += 1
            
            res = cv2.matchTemplate(grain_show, grain_show, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            print("BB4[INFO]:沟壑匹配度", max_val)
            # 纹路可视化
            # cv2.imshow("纹路", grain_show)

            if max_val >= self.bb4.GAP_SUBTRACT_START:
                texture_scores = self.bb4.GAP_SCORE
            else:
                texture_scores = int((self.bb4.GAP_SUBTRACT_START-max_val / self.bb4.GAP_SUBTRACT_RATIO)*self.bb4.GAP_SUBTRACT - 0.5)

            self.bb4.gap = texture_scores
        self.bb4.sum()
        return

    def score_all(self, teeth_pro):
        self.clear()

        self.score_aa1(teeth_pro.dst_all_mark, teeth_pro.site, teeth_pro.radius)
        is_ok = self.score_aa2(teeth_pro.dst_all_mark, teeth_pro.dst_fill_mark, teeth_pro.site,
                               teeth_pro.img_info.operation_time)       
        if is_ok == 0:
            print("！！！！！！！牙齿面积不符合要求直接退出！！！！！！！")
            return 0
        self.score_bb1(teeth_pro.src_gray_image, teeth_pro.dst_fill_mark,
                       teeth_pro.img_info.operation_time)
        self.score_bb2(teeth_pro.dst_fill_mark, teeth_pro.img_info.operation_time)
        self.score_bb3(teeth_pro.src_image, teeth_pro.dst_fill_mark, teeth_pro.dst_other_mark,
                       teeth_pro.img_info.operation_time)
        self.score_bb4(teeth_pro.src_gray_image, teeth_pro.dst_fill_mark,
                       teeth_pro.img_info.operation_time, teeth_pro.img_info.operation_name)

        self.grade = self.aa1.grade + self.aa2.grade + self.aa3.grade
        self.grade += self.bb1.grade + self.bb2.grade + self.bb3.grade + self.bb4.grade

        self.creat_score_txt(teeth_pro.img_info)
        self.aa1.print()
        self.aa2.print()
        self.aa3.print()
        self.bb1.print()
        self.bb2.print()
        self.bb3.print()
        self.bb4.print()
        return 1

    def creat_score_txt(self, img_info):
        if os.access(os.path.join(img_info.pro_path, 'score.txt'), os.F_OK):
            f = open(os.path.join(img_info.pro_path, 'score.txt'), 'a')
            f.write("\n")
            f.write(img_info.patient_name + "-" + img_info.operation_time + "-")
            f.write(img_info.operation_name + "-" + img_info.doctor_name + "-")
            f.write(str(self.aa1.grade) + "-" + str(self.aa2.grade) + "-" + str(self.aa3.grade) + "-")
            f.write(str(self.bb1.grade) + "-" + str(self.bb2.grade) + "-" + str(self.bb3.grade) + "-")
            f.write(str(self.bb4.grade) + "-" + str(self.grade))
        else:
            print("首次，需创建TXT文件")
            f = open(os.path.join(img_info.pro_path, 'score.txt'), 'w')
            f.write("0xaaee0xaaff0x55550xa5a5" + "\n")
            f.write(img_info.patient_name + "-" + img_info.operation_time + "-")
            f.write(img_info.operation_name + "-" + img_info.doctor_name + "-")
            f.write(str(self.aa1.grade) + "-" + str(self.aa2.grade) + "-" + str(self.aa3.grade) + "-")
            f.write(str(self.bb1.grade) + "-" + str(self.bb2.grade) + "-" + str(self.bb3.grade) + "-")
            f.write(str(self.bb4.grade) + "-" + str(self.grade))
        f.close()
        return
