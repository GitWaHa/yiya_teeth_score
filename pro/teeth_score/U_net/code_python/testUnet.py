#!/usr/bin/env python
# coding: utf-8

from model import *
from data import *

TARGET_INPUT_SIZE = (128, 128)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (1, )
PATH_MODEL_HDF5 = "../model_hdf5/unet_128.hdf5"

# test your model and save predicted results
print("[INFO] test unet")

# 测试图片列表
test_paths = sorted(glob.glob(os.path.join("../data/test",'*.png')))
# 测试图片数量
imagenum_test = len(test_paths)
# 测试图片生成器
testGene = testGenerator(test_paths, target_size = TARGET_INPUT_SIZE)

model = unet(input_size = MODEL_INPUT_SIZE)
model.load_weights(PATH_MODEL_HDF5)

results = model.predict_generator(testGene,imagenum_test,verbose=1)

# model.evaluate_generator(testGene, steps = 30)
# print("scores", x, y)
print(results.shape)
saveResult("../data/test_output", test_paths, results)

# # 测试单张图片
# model = unet()
# model.load_weights("unet_membrane.hdf5")
# results = model.predict(image_get(as_gray = True)) # shape: (1, 256, 256, 1)
# img = results[0,:,:,0]
# # io.imsave("predict.png",img)
# cv2.imshow('predict',img)
# cv2.waitKey(0)

