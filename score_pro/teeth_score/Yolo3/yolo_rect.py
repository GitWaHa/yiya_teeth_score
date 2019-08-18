import sys, os
import argparse
import numpy as np
sys.path.append('D:/Workspace/Git/yiya/teeth_score/score_pro/')
from teeth_score.Yolo3.yolo import YOLO
from PIL import Image
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

yolo = YOLO()
def detect_img(img_path=0):
    
    # img_path = 'D:/File/咿呀智能评分/TeethScore/JPG_TEST_History/JPG_TEST-28/患者1/201906_302052-患者1-术后-后牙-医生.jpg'
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    image = my_resize(416, 416, image)

    # return:[calss score x1 y1 x2 y2] or [calss score col1 raw1 c2 r2]
    result = np.array(yolo.detect_teeth(image))
    print(result)
    if len(result)==0:
        return
    index_score_max2min = my_nms(result, 0.5)
    result = result[index_score_max2min, :]
    print(result)
    resilt_show(result, image)
    # cv2.waitKey(0)


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
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep


def resilt_show(rect_data, image):
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
        print(text)
        cv2.putText(image_copy, text, (x1[i], y1[i]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.rectangle(image_copy, (x1[i], y1[i]), (x2[i], y2[i]), 255, 1)
    cv2.imshow('result', image_copy)


if __name__ == '__main__':
    detect_img(YOLO())
