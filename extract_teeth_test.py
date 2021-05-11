#!/usr/bin/python3.6
# coding=utf-8
import sys, os
import cv2
import argparse
import config as myconfig
sys.path.append(myconfig.WORK_FLODER)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from teeth import Teeth, pro_require

parser = argparse.ArgumentParser(description="yiya teeth score")
parser.add_argument('--image_dir',
                    type=str,
                    default=os.path.join(myconfig.WORK_FLODER,
                                         "./data/test/input/补牙"),
                    help='code path')
parser.add_argument('--debug', type=bool, default=False, help='is debug?')


def main():
    cmd_args = parser.parse_args()
    is_debug = cmd_args.debug

    teeth = Teeth()

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
            # is_ok, img_names = pro_require(img_names)
            # if is_ok == 0:
            #     myprint.printError(
            #         img_floder[i],
            #         "number of photos, format, naming format, is not ok")
            #     continue

            # 对三张照片分割评分
            for j in range(0, 3):
                img_path = current_path + "/" + img_names[j]
                print(img_path)
                teeth.img_info.get_info(img_path)

                teeth.clear()

                # 提取整个牙齿、按个所补牙及剩余牙齿
                extract_is_ok = teeth.extract_all(img_path)
                # teeth.img_info.print_info()
                # teeth.img_show()


if __name__ == '__main__':
    main()

    print("end of main")
