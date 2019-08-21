#!/usr/bin/python3.6
# coding=utf-8
import sys, os
import cv2
import config as myconfig
sys.path.append(myconfig.WORK_FLODER)

from teeth_score.teeth import Teeth, pro_require
from teeth_score.score import Teeth_Grade

USE_DEPLOY_FLAG = 0


def main():
    img_order = 0
    time_order = 0
    find_flag = 'not'
    teeth = Teeth()
    grade = Teeth_Grade()

    if USE_DEPLOY_FLAG == 0:
        JPG_dir = "D:/File/咿呀智能评分/TeethScore/JPG_TEST_History/JPG_TEST20-8-13/"
        for i in range(0, len(sys.argv)):
            print("参数", i, sys.argv[i])
            if i == 1:
                if sys.argv[1] == 'find':
                    find_flag = 'find'
                    break
            if i == 2:
                img_order = int(sys.argv[2])

            if i == 3:
                time_order = int(sys.argv[3])

        filenames = os.listdir(JPG_dir)
        # print(filenames)
        for i in range(img_order, len(filenames)):
            print("num:", i)

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
                img_path = current_path + "/" + img_names[j]
                print(img_path)
                teeth.img_info.get_info(img_path)
                if find_flag == 'find' and teeth.img_info.patient_name != (
                        '患者' + sys.argv[2]):
                    break

                teeth.clear()
                grade.clear()
                if grade.score_aa3(img_path) == 0:
                    print("照片格式未达到要求，不予评分")
                    break

                # 提取整个牙齿、按个所补牙及剩余牙齿
                # img_path = os.path.join(current_path, img_names[j])
                teeth.extract_all(img_path)
                teeth.img_info.print_info()
                teeth.img_show()

                # 根据提取的牙齿进行评分
                grade.score_all(teeth)
                print(" ")

                # key = 0
                # while key != ord("d"):
                #     key = cv2.waitKey(0)
                #     if key == 27:
                #         cv2.destroyAllWindows()
                #         return
            # cv2.destroyAllWindows()
            time_order = 0
    else:
        three_img_dir = [
            "D:/File/咿呀智能评分/TeethScore/JPG_TEST_History/JPG_TEST20-8-13/患者001/201908_132052-患者001-术前-D3-医生.jpg",
            "D:/File/咿呀智能评分/TeethScore/JPG_TEST_History/JPG_TEST20-8-13/患者001/201908_132052-患者001-术中-D3-医生.jpg",
            "D:/File/咿呀智能评分/TeethScore/JPG_TEST_History/JPG_TEST20-8-13/患者001/201908_132052-患者001-术后-D3-医生.jpg"
        ]

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
                for three_img_site_i in range(0, len(three_img_site)):
                    three_img_site[three_img_site_i] = int(
                        three_img_site[three_img_site_i])
                print(three_img_site)

        # 对三张照片分割评分
        for j in range(time_order, 3):
            img_path = three_img_dir[j]
            teeth.img_info.get_info(img_path, use_deploy=1)
            teeth.img_info.print_info()

            grade.clear()
            if grade.score_aa3(img_path) == 0:
                print("照片格式未达到要求，不予评分")
                break

            # 提取整个牙齿、按个所补牙及剩余牙齿
            teeth.extract_all(img_path)
            teeth.img_show()

            # 根据提取的牙齿进行评分
            grade.score_all(teeth, use_deploy=1)
            # print(" ")
            cv2.waitKey(0)


if __name__ == '__main__':
    main()

    print("end of main")
