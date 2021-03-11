from network.AlexNet.code_python.model import AlexNet
from keras.models import load_model
import cv2, os
import numpy as np

TARGET_INPUT_SIZE = (128, 128)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (1, )

import config as myconfig
work_floder = myconfig.WORK_FLODER

PATH_MODEL_HDF5 = os.path.join(work_floder, "mode/alexnet/alexnet_128.hdf5")
print(PATH_MODEL_HDF5)


def image_proc(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1, ))
    img = np.reshape(img, (1, ) + img.shape)
    return img


model = AlexNet(input_size=MODEL_INPUT_SIZE)
model.load_weights(PATH_MODEL_HDF5)
# model = load_model(PATH_MODEL_HDF5, compile=False)


def classify_teethtype(roi_image):
    # 获得网络的输入图片
    pre_image = image_proc(roi_image, TARGET_INPUT_SIZE)
    # 调用模型
    results = model.predict(pre_image)

    return np.argmax(results)