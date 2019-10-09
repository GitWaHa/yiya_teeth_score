import sys
sys.path.append('D:/Workspace/Git/yiya/teeth_score/score_pro/')

from teeth_score.ResNet50.code_python.model import MyResNet50
from teeth_score.teeth import my_limit
from keras.models import load_model
import cv2, os
import numpy as np

TARGET_INPUT_SIZE = (128, 128)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (3, )

import teeth_score.config as myconfig
work_floder = myconfig.WORK_FLODER

PATH_MODEL_HDF5_BB2_3 = work_floder + "teeth_score/ResNet50/model_hdf5/resnet50_128_bb2_3.hdf5"
PATH_MODEL_HDF5_BB1 = work_floder + "teeth_score/ResNet50/model_hdf5/resnet50_128_bb1.hdf5"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def image_proc(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = cv2.resize(img, target_size)
    # img = np.reshape(img, img.shape + (3, ))
    img = np.reshape(img, (1, ) + img.shape)
    return img


model = MyResNet50(input_size=MODEL_INPUT_SIZE)
model.load_weights(PATH_MODEL_HDF5_BB1)

model_bb2_3 = MyResNet50(input_size=MODEL_INPUT_SIZE)
model_bb2_3.load_weights(PATH_MODEL_HDF5_BB2_3)
# model = load_model(PATH_MODEL_HDF5, compile=False)


def classify_bb1(src_img, fill_rect, flag=0, img=0):
    if flag == 0:
        if len(fill_rect) == 0:
            return 3
        img_rows, img_cols = src_img.shape[:2]
        w = fill_rect[2] - fill_rect[0]
        h = fill_rect[3] - fill_rect[1]
        min_row = int(my_limit(fill_rect[1] - h * 0.1, 0, img_rows))
        max_row = int(my_limit(fill_rect[3] + h * 0.1, 0, img_rows))
        min_col = int(my_limit(fill_rect[0] - w * 0.1, 0, img_cols))
        max_col = int(my_limit(fill_rect[2] + h * 0.1, 0, img_cols))

        roi_img = src_img[min_row:max_row, min_col:max_col]
    else:
        roi_img = img
        cv2.imshow('roi', roi_img)

    # 获得网络的输入图片
    pre_image = image_proc(roi_img, TARGET_INPUT_SIZE)
    # 调用模型
    results = model.predict(pre_image)

    return np.argmax(results)


def classify_bb2_bb3(src_img, fill_rect, flag=0, img=0):
    if flag == 0:
        if len(fill_rect) == 0:
            return 3
        img_rows, img_cols = src_img.shape[:2]
        w = fill_rect[2] - fill_rect[0]
        h = fill_rect[3] - fill_rect[1]
        min_row = int(my_limit(fill_rect[1] - h * 0.1, 0, img_rows))
        max_row = int(my_limit(fill_rect[3] + h * 0.1, 0, img_rows))
        min_col = int(my_limit(fill_rect[0] - w * 0.1, 0, img_cols))
        max_col = int(my_limit(fill_rect[2] + h * 0.1, 0, img_cols))

        roi_img = src_img[min_row:max_row, min_col:max_col]
    else:
        roi_img = img
        cv2.imshow('roi', roi_img)
    # 获得网络的输入图片
    pre_image = image_proc(roi_img.copy(), TARGET_INPUT_SIZE)
    # 调用模型
    results = model_bb2_3.predict(pre_image)

    return np.argmax(results)


if __name__ == "__main__":
    filenames = os.listdir('data/train/2')
    for i in range(len(filenames)):
        print(filenames[i])
        img = cv2.imread('data/train/2/' + filenames[i])
        print(classify_bb2_bb3(0, 0, 1, img))
        key = cv2.waitKey(0)
        if key == ord('q'):
            break