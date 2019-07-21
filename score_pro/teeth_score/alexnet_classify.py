from AlexNet.code_python.model import AlexNet
import cv2
import numpy as np

TARGET_INPUT_SIZE = (128, 128)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (1, )
PATH_MODEL_HDF5 = "./AlexNet/model_hdf5/alexnet_128.hdf5"
# PATH_MODEL_HDF5 = "D:/WorkingFolder/Git/teeth_pro/score_pro/teeth_score/AlexNet/model_hdf5/alexnet_128.hdf5"


def image_proc(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1, ))
    img = np.reshape(img, (1, ) + img.shape)
    return img


model = AlexNet(input_size=MODEL_INPUT_SIZE)
model.load_weights(PATH_MODEL_HDF5)


def alexnet_classify_fillteeth(roi_image):
    # 获得网络的输入图片
    pre_image = image_proc(roi_image)
    # 调用模型进行分割
    results = model.predict(pre_image)  # shape: (1, 256, 256, 1)

    return np.argmax(results)