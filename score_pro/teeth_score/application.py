#!/usr/bin/python3.6
# coding=utf-8
import sys, os
import cv2
import config as myconfig
sys.path.append(myconfig.WORK_FLODER)

from teeth_score.teeth import Teeth, pro_require
from teeth_score.score import Teeth_Grade

USE_DEPLOY_FLAG = 1


def main():
    img_order = 0
    time_order = 0
    find_flag = 'not'
    teeth = Teeth()
    grade = Teeth_Grade()

    JPG_dir = "D:/File/咿呀智能评分/TeethScore/JPG_TEST_History/first_test_112/"
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
            print("error_0#", filenames[i], "#不是文件夹，不予评分")
            continue

        img_names = os.listdir(current_path)
        is_ok, img_names = pro_require(img_names)
        if is_ok == 0:
            print("error_0#", filenames[i], "#此文件夹内文件命名不符合要求，不予评分")
            continue

        # 对三张照片分割评分
        for j in range(time_order, 3):
            img_path = current_path + "/" + img_names[j]
            # print(img_path)
            teeth.img_info.get_info(img_path)
            if find_flag == 'find' and filenames[i] != (sys.argv[2]):
                break

            teeth.clear()
            grade.clear()
            if grade.score_aa3(img_path) == 0:
                print("error_1#",
                      filenames[i] + "_" + teeth.img_info.operation_time,
                      "#照片格式未达到要求，不予评分")
                break

            # 提取整个牙齿、按个所补牙及剩余牙齿
            extract_is_ok = teeth.extract_all(img_path)
            # teeth.img_info.print_info()
            teeth.img_show()

            # 根据提取的牙齿进行评分
            if extract_is_ok == 1:
                grade.score_all(teeth,
                                filenames[i],
                                use_deploy=USE_DEPLOY_FLAG)
            else:
                print("error_1#",
                      filenames[i] + "_" + teeth.img_info.operation_time,
                      "#自动标注未找到所需牙位所补牙")
                grade.score_all(teeth,
                                filenames[i],
                                use_deploy=USE_DEPLOY_FLAG)

        #     key = 0
        #     while key != ord("d"):
        #         key = cv2.waitKey(0)
        #         if key == 27:
        #             cv2.destroyAllWindows()
        #             return
        # cv2.destroyAllWindows()
        time_order = 0


if __name__ == '__main__':
    main()

    print("end of main")
