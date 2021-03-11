#!/usr/bin/python3.6
# coding=utf-8
import sys, os
import cv2
import argparse
import config as myconfig
sys.path.append(myconfig.WORK_FLODER)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from teeth import Teeth, pro_require
from score import Teeth_Grade
from output_info import OutputInfo

parser = argparse.ArgumentParser(description="yiya teeth score")
parser.add_argument('--image_dir',
                    type=str,
                    default=os.path.join(myconfig.WORK_FLODER, "test_data"),
                    help='code path')
parser.add_argument('--debug', type=bool, default=False, help='is debug?')


def main():
    cmd_args = parser.parse_args()
    is_debug = cmd_args.debug

    find_flag = 'not'
    teeth = Teeth()
    grade = Teeth_Grade()
    myprint = OutputInfo()

    if is_debug == False:
        image_dir = cmd_args.image_dir
        # exit()

        img_floder = os.listdir(image_dir)
        # print(filenames)
        for i in range(0, len(img_floder)):
            print("num:", i)

            current_path = os.path.join(image_dir, img_floder[i])
            if os.path.isfile(current_path):
                print(img_floder[i], "Not a folder, no scoring")
                continue

            img_names = os.listdir(current_path)
            is_ok, img_names = pro_require(img_names)
            if is_ok == 0:
                myprint.printError(
                    img_floder[i],
                    "number of photos, format, naming format, is not ok")
                continue

            # 对三张照片分割评分
            all_score = []
            debig_info = "is ok"
            for j in range(0, 3):
                img_path = current_path + "/" + img_names[j]
                teeth.img_info.get_info(img_path)
                if find_flag == 'find' and teeth.img_info.patient_name != (
                        '患者' + sys.argv[2]):
                    break

                teeth.clear()
                grade.clear()
                if grade.score_aa3(img_path) == 0:
                    debig_info = "picture format is not jpg"
                    break

                # 提取整个牙齿、按个所补牙及剩余牙齿
                extract_is_ok = teeth.extract_all(img_path)
                # teeth.img_info.print_info()
                # teeth.img_show()

                # 根据提取的牙齿进行评分
                if extract_is_ok == 1:
                    score_is_ok = grade.score_all(teeth, output_to_txt=False)
                    if score_is_ok == 0:
                        debig_info = "Failed to extract the specify teeth"
                        break
                    all_score.append(grade.getScores())
                else:
                    debig_info = "Failed to extract the specify teeth"
                    break
                # print(" ")
            if (len(all_score) == 3):
                myprint.printScore(teeth.img_info.patient_name, all_score,
                                   debig_info)
            else:
                myprint.printError(teeth.img_info.patient_name, debig_info)
            #     key = 0
            #     while key != ord("d"):
            #         key = cv2.waitKey(0)
            #         if key == 27:
            #             cv2.destroyAllWindows()
            #             return
            # cv2.destroyAllWindows()

    # else:
    #     three_img_dir = [
    #         "D:\Download\BaiduNetdiskDownload\score_pro/teeth_score/202012211619万徐香18986439884\\201908_132052-患者001-术前-D3-医生.jpg",
    #         "D:\Download\BaiduNetdiskDownload\score_pro/teeth_score/202012211619万徐香18986439884\\201908_132052-患者001-术中-D3-医生.jpg",
    #         "D:\Download\BaiduNetdiskDownload\score_pro/teeth_score/202012211619万徐香18986439884\\201908_132052-患者001-术后-D3-医生.jpg"
    #     ]

    #     # 三个图片地址位置获取
    #     for i in range(0, len(sys.argv)):
    #         # print("参数", i, sys.argv[i])
    #         if i == 1:
    #             three_img_addr_str = sys.argv[i]
    #             three_img_dir = three_img_addr_str.split(',')
    #             print(three_img_dir)
    #         if i == 2:
    #             three_img_site_str = sys.argv[i]
    #             three_img_site = three_img_site_str.split(',')
    #             for three_img_site_i in range(0, len(three_img_site)):
    #                 three_img_site[three_img_site_i] = int(
    #                     three_img_site[three_img_site_i])
    #             print(three_img_site)

    #     # 对三张照片分割评分
    #     for j in range(time_order, 3):
    #         img_path = three_img_dir[j]
    #         teeth.img_info.get_info(img_path, use_deploy=1)
    #         # teeth.img_info.print_info()

    #         grade.clear()
    #         if grade.score_aa3(img_path) == 0:
    #             print("照片格式未达到要求，不予评分")
    #             break

    #         # 提取整个牙齿、按个所补牙及剩余牙齿
    #         teeth.extract_all(img_path)
    #         # teeth.img_show()

    #         # 根据提取的牙齿进行评分
    #         grade.score_all(teeth, use_deploy=1)
    #         # print(" ")
    #         # cv2.waitKey(0)


if __name__ == '__main__':
    main()

    print("end of main")
