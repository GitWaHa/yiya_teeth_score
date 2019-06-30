import time
import cv2
import sys, os, re

def main():
    JPG_dir = "D:/WorkingFolder/Git/teeth_pro/score_pro/JPG_TEST/"
    filenames = os.listdir(JPG_dir)
    # print(filenames)
    for i in range(0, len(filenames)):
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
        for j in range(0, 3):
            old_name = img_names[j]
            new_name = get_info(img_names[j])
            print(current_path+'/'+new_name)
            os.rename(current_path+'/'+old_name,current_path+'/'+new_name)


def get_info(img_dir):
    pattern = r"(.*)-(.*)-(.*)-(.*)\.(.*)"
    info = list(re.findall(pattern, img_dir)[0])
    upload_time = '201906_301856'
    patient_name = info[0]
    operation_time = info[1]
    operation_name = info[2]
    doctor_name = info[3]
    img_type = info[4]

    return upload_time+'-'+patient_name+'-'+operation_time+'-'+operation_name+'-'+doctor_name+'.jpg'


def pro_require(img_names):
    jpg_num = 0
    first_flag = 0
    second_flag = 0
    third_flag = 0
    correct_img_names = [0 for i in range(3)]
    for i in range(len(img_names)):
        img_str = img_names[i].split(".")
        if img_str[1] == "jpg":
            jpg_num += 1
            img_name_str = img_str[0].split("-")
            operation_time = img_name_str[1]

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


if __name__ == '__main__':
    main()

    print("end of main")