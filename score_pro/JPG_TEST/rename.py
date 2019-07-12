import os, re


def main():
    jpg_dir = "D:/WorkingFolder/Git/teeth_pro/score_pro/JPG_TEST/"
    filenames = os.listdir(jpg_dir)

    print(filenames)
    for i in range(0, len(filenames)):
        print("num:", i)

        current_path = jpg_dir + filenames[i]
        if os.path.isfile(current_path):
            print(filenames[i], "不是文件夹，不予评分")
            continue

        img_names = os.listdir(current_path)
        rename_flag = False
        str = 0
        for img_name in img_names:
            file_type = img_name.split('.')[1]
            if file_type == 'jpg':
                pattern = r"(.*)-(.*)-(.*)-(.*)-(.*)\.(.*)"
                info = list(re.findall(pattern, img_name)[0])
                print(info)
                if rename_flag == False:
                    str = input("请输入牙位信息：")
                    rename_flag = True
                info[3] = str
                new_name = '-'.join(info[0:5])
                new_name = new_name + '.jpg'
                print(new_name)
                os.rename(os.path.join(current_path,img_name),os.path.join(current_path,new_name))
        rename_flag = False


if __name__ == "__main__":
    main()
