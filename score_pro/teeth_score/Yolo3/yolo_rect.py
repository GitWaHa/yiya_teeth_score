import sys, os
from imutils import paths
import argparse
import numpy as np
sys.path.append('D:/Workspace/Git/yiya/teeth_score/score_pro/')
from teeth_score.Yolo3.yolo import YOLO
from PIL import Image
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

yolo = YOLO()


def detect_img(img_path=0):
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    image = my_resize(416, 416, image)

    # return:[calss score x1 y1 x2 y2] or [calss score col1 raw1 c2 r2]
    result = np.array(yolo.detect_teeth(image))
    # print(result)
    if len(result) == 0:
        return []
    index_score_max2min = my_nms(result, 0.3)
    result_score_max2min = result[index_score_max2min, :]
    final_result = rectify_class_label(result_score_max2min)

    result_show(final_result, image)

    return final_result


def rectify_class_label(rect_data):
    CLASS_ORDER = np.array([
        '0', '0', '0', '0', '0', '0', '7', '6', '5', '4', '3', '2', '1', '1',
        '2', '3', '4', '5', '6', '7', '0', '0', '0', '0', '0', '0'
    ])

    result_rect_left2right = rect_data[np.argsort(rect_data[:, 2].astype(
        np.int))]
    # print(result_rect_left2right)
    result_class = result_rect_left2right[:, 0]
    result_score = result_rect_left2right[:, 1].astype(np.float)
    max_count = 0
    max_score = 0
    idx_rectify = []
    score_rectify = []
    for i in range(len(CLASS_ORDER) - len(result_class) + 1):
        count = 0
        score_count = 0
        for j in range(len(result_class)):
            if CLASS_ORDER[i + j] == result_class[j]:
                count += 1
                score_count += result_score[j]
        if count >= max_count:
            if count > max_count:
                idx_rectify = []
                score_rectify = []
            max_count = count
            score_rectify.append(score_count)
            idx_rectify.append(np.array(CLASS_ORDER[i:i + len(result_class)]))
            # print(count)
    print(idx_rectify)
    max_score_class = idx_rectify[np.argmax(score_rectify)]
    print(max_score_class)
    result_rect_left2right[:, 0] = max_score_class
    print(result_rect_left2right)
    return result_rect_left2right


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


def my_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = np.array(dets[:, 2]).astype(np.int)
    y1 = np.array(dets[:, 3]).astype(np.int)
    x2 = np.array(dets[:, 4]).astype(np.int)
    y2 = np.array(dets[:, 5]).astype(np.int)
    scores = np.array(dets[:, 1]).astype(np.float)

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        # print(ovr)
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


def result_resize(rect_data, set_size=480):
    rect_data_copy = rect_data.copy()
    scale = set_size / 418
    rect_data_copy[:, 2:] = (rect_data[:, 2:].astype(np.int) * scale).astype(
        np.int)
    return rect_data_copy


def result_show(rect_data, image):
    x1 = np.array(rect_data[:, 2]).astype(np.int)
    y1 = np.array(rect_data[:, 3]).astype(np.int)
    x2 = np.array(rect_data[:, 4]).astype(np.int)
    y2 = np.array(rect_data[:, 5]).astype(np.int)
    scores = np.array(rect_data[:, 1]).astype(np.float)
    scores = np.around(scores, decimals=2).astype(np.str)
    class_label = np.array(rect_data[:, 0]).astype(np.str)

    image_copy = image.copy()
    for i in range(len(rect_data)):
        text = class_label[i] + ' ' + scores[i]
        # print(text)
        cv2.putText(image_copy, text, (x1[i], y2[i]), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 255, 0), 1)
        cv2.rectangle(image_copy, (x1[i], y1[i]), (x2[i], y2[i]), 255, 1)
    cv2.imshow('result', image_copy)


def main():
    test_img_path = 'D:\File\咿呀智能评分\TeethScore\JPG_TEST_History\JPG_TEST-28'
    img_names = sorted(list(paths.list_images(test_img_path)))
    print(img_names)
    num = 0
    while True:
        print(num)
        detect_img(img_names[num])

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

if __name__ == '__main__':
    main()
