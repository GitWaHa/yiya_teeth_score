#!/usr/bin/python3.6
# coding=utf-8

# import matplotlib.pyplot as plt
import time
import cv2 as cv
from teeth import *
from score import Teeth_Grade
import sys
 

def main():
    img_order = 0
    time_order = 0
    teeth = Teeth()
    grade = Teeth_Grade()
    dir = "/home/waha/git/pro/JPG_TEST/"

    # 参数获取
    for i in range(0, len(sys.argv)):
        print("参数", i, sys.argv[i])
        if i == 1:
            if sys.argv[1] != 'debug':
                print(sys.argv[1])
                dir = sys.argv[1]
        if i == 2:
            img_order = int(sys.argv[2]) - 1

        if i == 3:
            time_order = int(sys.argv[3]) - 1

    filenames = os.listdir(dir)
    print(filenames)
    for i in range(img_order, len(filenames)):
        current_path = os.path.join(dir, filenames[i])
        if os.path.isfile(current_path):
            print(filenames[i], "不是文件夹，不予评分")
            continue

        img_names = os.listdir(current_path)
        is_ok, img_names = pro_require(img_names)
        if is_ok == 0:
            print("此文件夹内文件命名不符合要求，不予评分")
            continue

        # 对三证照片分割评分
        for j in range(time_order, 3):
            teeth.img_info.get_info(img_names[j], dir)
            teeth.img_info.print_info()
            img_path = os.path.join(current_path, img_names[j])
            print(img_path)

            teeth.clear()
            grade.clear()
            if grade.score_aa3(img_path) == 0:
                print("照片格式未达到要求，不予评分")
                break

            start = time.time()
            # 提取整个牙齿、按个所补牙及剩余牙齿
            teeth.extract_all(current_path, img_names[j])
            teeth.img_show()
            elapsed = (time.time() - start)
            print("提取牙齿Time used:", elapsed, '\n')

            start = time.time()
            # 根据提取的牙齿进行评分
            grade.score_all(teeth)
            elapsed = (time.time() - start)
            print("评分Time used:", elapsed, '\n')

            key = 0
            while key != 84:
                key = cv.waitKey(0)
            
                if key == 27:
                    cv.destroyAllWindows()
                    return
        time_order = 0


if __name__ == '__main__':
    start = time.time()
    main()
    run_time = time.time() - start
    print("评分Time used:", run_time, '\n')
    print("end of main")


