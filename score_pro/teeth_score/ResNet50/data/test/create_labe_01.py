import cv2
import os, re, io
import numpy as np


def main():
    img_dir = "./img/"
    zero_dir = "./0/"
    one_dir = "./1/"
    # label_dir = "./img"
    img_filenames = os.listdir(img_dir)
    # label_filenames = os.listdir(label_dir)
    # print(filenames)
    np_label_list_read = np.load("label.npy")
    # print(np_label_list_read.shape, np_label_list_read)
    # label_list = []
    # print(label_list)
    for i in range(0, len(np_label_list_read)):
        print("num:", i)
        current_path = img_dir + img_filenames[i]

        img = cv2.imread(current_path)

        if np_label_list_read[i] == 0:
            cv2.imwrite(zero_dir + img_filenames[i], img)
        elif np_label_list_read[i] == 1:
            cv2.imwrite(one_dir + img_filenames[i], img)
        # cv2.imshow("img", img)
        # key = 3
        # key = cv2.waitKey(0)
        # if key == ord('0'):
        #     label = 0
        # elif key == ord('1'):
        #     label = 1

        # label = input("输入 类别（0：门牙，1：后牙）  ")
        # label_list.append(label)

        # np_label_list = np.array(label_list)
        # np_label_list = np.reshape(np_label_list, (-1, 1))
        # print(np_label_list)
        # np.save("label.npy", np_label_list)
        # np_label_list = np.load("label.npy")

        # print(np_label_list)
        # if os.path.isfile(current_path):
        #     print(filenames[i], "不是文件夹，不予评分")
        #     continue

        # img_names = os.listdir(current_path)
        # is_ok, img_names = pro_require(img_names)
        # if is_ok == 0:
        #     print("此文件夹内文件命名不符合要求，不予评分")
        #     continue
        # 对三张照片分割评分
        # for j in range(0, 3):
        #     old_name = img_names[j]
        #     new_name = get_info(img_names[j])
        #     print(current_path + '/' + new_name)
        #     os.rename(current_path + '/' + old_name,
        #               current_path + '/' + new_name)


if __name__ == '__main__':
    main()

    print("end of main")
