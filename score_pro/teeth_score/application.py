#!/usr/bin/python3.6
# coding=utf-8

import time
import cv2 as cv
from teeth import *
from score import Teeth_Grade
import sys

USE_DEPLOY_FLAG = 1

def main():
    img_order = 0
    time_order = 0
    teeth = Teeth()
    grade = Teeth_Grade()
    
    # dir = "D:/WorkingFolder/Git/teeth_pro/score_pro/JPG_TEST/"
    if USE_DEPLOY_FLAG==0:
        JPG_dir = "D:/WorkingFolder/Git/teeth_pro/score_pro/JPG_TEST/"
    else:
        three_img_dir = ["C:/Users/WaHa/Desktop/TeethScore/project/test_img/201906_301527-患者1-术前-后牙-医生.jpg",
                    "C:/Users/WaHa/Desktop/TeethScore/project/test_img/201906_301527-患者1-术中-后牙-医生.jpg",
                    "C:/Users/WaHa/Desktop/TeethScore/project/test_img/201906_301527-患者1-术后-后牙-医生.jpg"]
        three_img_site = [1,192,179,370,0,187,182,369,2,188,178,364]

    # 参数获取
    if USE_DEPLOY_FLAG == 0:
        for i in range(0, len(sys.argv)):
            print("参数", i, sys.argv[i])
            if i == 1:
                if sys.argv[1] != 'debug':
                    print(sys.argv[1])
                    JPG_dir = sys.argv[1]
            if i == 2:
                img_order = int(sys.argv[2]) - 1

            if i == 3:
                time_order = int(sys.argv[3]) - 1

        filenames = os.listdir(JPG_dir)
        # print(filenames)
        for i in range(img_order, len(filenames)):
            print("num:",i)

            current_path = JPG_dir + filenames[i]
            if os.path.isfile(current_path):
                print(filenames[i], "不是文件夹，不予评分")
                continue

            img_names = os.listdir(current_path)
            is_ok, img_names = pro_require(img_names)
            if is_ok == 0:
                print("此文件夹内文件命名不符合要求，不予评分")
                continue

            # 对三张照片分割评分
            for j in range(time_order, 3):
                teeth.img_info.get_info(img_names[j])
                teeth.img_info.print_info()
                # img_path = os.path.join(current_path, img_names[j])
                img_path = current_path + "/" + img_names[j]

                teeth.clear()
                grade.clear()
                if grade.score_aa3(img_path) == 0:
                    print("照片格式未达到要求，不予评分")
                    break

                # 提取整个牙齿、按个所补牙及剩余牙齿
                teeth.extract_all(0, current_path, img_names[j])
                teeth.img_show()

                # 根据提取的牙齿进行评分
                grade.score_all(teeth)
                print(" ")

                key = 0
                while key != ord("d"):
                    key = cv.waitKey(0)
                    if key == 27:
                        cv.destroyAllWindows()
                        return
            cv.destroyAllWindows()
            time_order = 0
    else:
        # 三个图片地址位置获取
        for i in range(0, len(sys.argv)):
            # print("参数", i, sys.argv[i])
            if i == 1:
                three_img_addr_str = sys.argv[i]
                three_img_dir = three_img_addr_str.split(',')
                print(three_img_dir)
            if i == 2:
                three_img_site_str = sys.argv[i]
                three_img_site = three_img_site_str.split(',')
                for three_img_site_i in range(0,len(three_img_site)):
                    three_img_site[three_img_site_i] = int(three_img_site[three_img_site_i])
                print(three_img_site)

        # 对三张照片分割评分
        for j in range(time_order, 3):
            img_path = three_img_dir[j]
            teeth.img_info.get_info(img_path, use_deploy=1)
            # teeth.img_info.print_info()

            teeth.clear()
            grade.clear()
            if grade.score_aa3(img_path) == 0:
                print("照片格式未达到要求，不予评分")
                break

            # 提取整个牙齿、按个所补牙及剩余牙齿
            # teeth.extract_all(current_path, img_names[j],img_path=img_path)
            teeth.extract_all(1, img_path=img_path, site=three_img_site)
            teeth.img_show()

            # 根据提取的牙齿进行评分
            grade.score_all(teeth, use_deploy=1)
            # print(" ")
            cv.waitKey(0)



if __name__ == '__main__':
    main()

    print("end of main")


