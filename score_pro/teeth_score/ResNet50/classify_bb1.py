from teeth_score.ResNet50.code_python.model import MyResNet50
from keras.models import load_model
import cv2, os
import numpy as np

TARGET_INPUT_SIZE = (256, 256)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (3, )

import teeth_score.config as myconfig
work_floder = myconfig.WORK_FLODER

PATH_MODEL_HDF5 = work_floder + "teeth_score/ResNet50/model_hdf5/resnet50_128.hdf5"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def image_proc(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    img = cv2.resize(img, target_size)
    # img = np.reshape(img, img.shape + (3, ))
    img = np.reshape(img, (1, ) + img.shape)
    return img


model = MyResNet50(input_size=MODEL_INPUT_SIZE)
model.load_weights(PATH_MODEL_HDF5)
# model = load_model(PATH_MODEL_HDF5, compile=False)


def classify_bb1(roi_image):
    # 获得网络的输入图片
    pre_image = image_proc(roi_image, TARGET_INPUT_SIZE)
    # 调用模型
    results = model.predict(pre_image)  

    return np.argmax(results)