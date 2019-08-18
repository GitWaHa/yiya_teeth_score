import os, cv2
from imutils import paths
import numpy as np

label_flag = 0  # 标注动作标志
PointStart = [0, 0]  # 起点坐标
PointEnd = [0, 0]  # 终点坐标
row_format = 0


def main():
    img_path = './original_22'
    img_names = sorted(list(paths.list_images(img_path)))
    print(img_names)
    num = 17
    while True:
        print(num)
        read_img(img_names, num)

        key = 0
        while key != ord('d') or key != ord('a'):
            key = cv2.waitKey(0)
            if key == ord('d'):
                num += 1
                if num == len(img_names):
                    return
                break
            elif key == ord('a'):
                if num > 0:
                    num -= 1
                break
            elif key == 27:
                cv2.destroyAllWindows()
                return


def displap_label():
    with open('../train.txt', 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for line in lines:
            new_line = []
            line_str = line.split(' ')
            new_line.append(line_str[0])
            img_path = '.' + line_str[0]
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            for i in range(1, len(line_str)):
                new_rect =  []
                rect = line_str[i]
                rect_str = rect.split(',')
                print(rect_str)
                st_r = int(rect_str[0])
                st_y = str(st_r)
                st_c = int(rect_str[1])
                st_x = str(st_c)
                ed_r = int(rect_str[2])
                ed_y = str(ed_r)
                ed_c = int(rect_str[3])
                ed_x = str(ed_c)
                class_label = rect_str[4]
                new_rect.append(st_x)
                new_rect.append(st_y)
                new_rect.append(ed_x)
                new_rect.append(ed_y)
                new_rect.append(class_label)
                new_line.append(','.join(new_rect))
                cv2.rectangle(img, (st_c, st_r), (ed_c, ed_r), 255, 1)
            line_to_ok = ' '.join(new_line)
            print(line_to_ok)
            with open('./train_to.txt', 'a', encoding='UTF-8') as file2:
                file2.write(line_to_ok)
            # cv2.imshow('img', img)
            # key = 0
            # while key != ord('d') and key != ord('a'):
            #     key = cv2.waitKey(0)
            #     if key == ord('q'):
            #         cv2.destroyAllWindows()
            #         return


train_img_path = './label_data_creat/original_22'


def read_img(img_names, img_num=0):
    global row_format, label_flag
    image = cv2.imdecode(np.fromfile(img_names[img_num], dtype=np.uint8), -1)
    # print (image.shape[0],image.shape[1])

    # 缩放到屏幕可显示范围内
    image = my_resize(416, 416, image)
    img_name_str = img_names[img_num].split('\\')
    img_name_str[0] = train_img_path
    imgpath = '/'.join(img_name_str)
    print(imgpath)

    cv2.imshow("image", image)
    cv2.setMouseCallback('image', get_label, [image, imgpath])

    key = 0
    while key != ord('s'):
        key = cv2.waitKey(0)
        if key == 27:
            return
        if key == ord('s'):
            print('save row_format', row_format)
            try:
                f = open('./train.txt', 'a')
            except IOError:
                print("缺少必要文件 train.text")
                return
            f.write(row_format + '\n')
            f.close()

            cv2.imencode('.jpg', image)[1].tofile(img_names[img_num])
        else:
            print('错误保存，保存按s,退出按ESC')
    row_format = 0


def my_resize(set_rows, set_cols, src_image):
    img_rows, img_cols = src_image.shape[:2]
    src_copy = src_image.copy()

    if img_rows >= img_cols and img_rows > set_rows:
        resize_k = set_rows / img_rows
        src_copy = cv2.resize(src_copy, (int(resize_k * img_cols), set_rows),
                              interpolation=cv2.INTER_AREA)
    elif img_cols > img_rows and img_cols > set_cols:
        resize_k = set_cols / img_cols
        src_copy = cv2.resize(src_copy, (set_cols, int(resize_k * img_rows)),
                              interpolation=cv2.INTER_AREA)
    img_rows, img_cols, channel = src_copy.shape
    img = np.zeros((set_rows, set_cols, channel), dtype=np.uint8)
    img[0:img_rows, 0:img_cols] = src_copy

    return img


def get_label(event, x, y, flags, param):
    global label_flag, row_format
    if event == cv2.EVENT_LBUTTONDOWN:
        label_flag = 1  # 左键按下
        PointStart[0], PointStart[1] = x, y  # 记录起点位置
    elif event == cv2.EVENT_LBUTTONUP and label_flag == 1:  # 左键按下后检测弹起
        label_flag = 2  # 左键弹起
        PointEnd[0], PointEnd[1] = x, y  # 记录终点位置
        # PointEnd[1] = PointStart[1]+(PointEnd[0]-PointStart[0]) # 形成正方形
        # 提取ROI
        if PointEnd[0] != PointStart[0] and PointEnd[1] != PointStart[
                1]:  # 框出了矩形区域,而非点
            print("SPoint =", (PointStart[0], PointStart[1]))
            print("EPoint =", (PointEnd[0], PointEnd[1]), '\n')

            # 左上角点xy坐标值均较小
            point_x_st = min(PointStart[0], PointEnd[0])
            point_y_st = min(PointStart[1], PointEnd[1])
            # 右下角点xy坐标值均较大
            point_x_ed = max(PointStart[0], PointEnd[0])
            point_y_ed = max(PointStart[1], PointEnd[1])
            # 提取ROI
            image_roi = param[0][point_y_st:point_y_ed, point_x_st:point_x_ed]
            cv2.imshow('roi', image_roi)

            key = 0
            while key < 49 or key > 55:
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyWindow('roi')
                    return

                param[1] = param[1] + ' ' + str(point_x_st) + ',' + str(
                    point_y_st) + ',' + str(point_x_ed) + ',' + str(
                        point_y_ed) + ',' + chr(key)

                row_format = param[1]
                print(row_format)
            cv2.destroyWindow('roi')

    elif event == cv2.EVENT_MOUSEMOVE and label_flag == 1:  # 左键按下后获取当前坐标, 并更新标注框
        PointEnd[0], PointEnd[1] = x, y  # 记录当前位置
        # PointEnd[1] = PointStart[1]+(PointEnd[0]-PointStart[0]) # 形成正方形
        image_copy = param[0].copy()
        cv2.rectangle(image_copy, (PointStart[0], PointStart[1]),
                      (PointEnd[0], PointEnd[1]), (0, 255, 0), 1)  # 根据x坐标画正方形
        cv2.imshow('image', image_copy)


if __name__ == '__main__':
    displap_label()
    # main()