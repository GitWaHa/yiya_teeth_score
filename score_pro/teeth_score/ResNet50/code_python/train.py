#!/usr/bin/env python
# coding: utf-8

from model import MyResNet50
from data import trainGenerator
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TARGET_INPUT_SIZE = (128, 128)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (3, )
PATH_MODEL_HDF5 = "../model_hdf5/resnet50_128.hdf5"

# 数据生成器参数

data_gen_args = dict(
    rotation_range=45,  # 随机旋转角度
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,  # 剪切强度（以弧度逆时针方向剪切角度）
    zoom_range=0.1,  # 随机缩放范围。
    horizontal_flip=True,  # 随机水平翻转
    fill_mode='nearest',
    rescale=1.0 / 255)  # 填充模式

myGene = trainGenerator(
    4,
    '../data/train',
    data_gen_args,
    image_color_mode="rgb",
    save_to_dir=None,
    target_size=TARGET_INPUT_SIZE)  # save_to_dir = 'data/trainGenerator'
# print(myGene)

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    '../data/test',
    target_size=TARGET_INPUT_SIZE,
    batch_size=4,
    classes=['0', '1', '2'],
    class_mode='categorical',
    color_mode="rgb")

# print(myGene)
# 模型
model = MyResNet50(input_size=MODEL_INPUT_SIZE)
# 继续训练则加载已存在的模型参数
model.load_weights(PATH_MODEL_HDF5)

# # 当loss 减小则保存，否则跳过
# model_checkpoint = ModelCheckpoint(PATH_MODEL_HDF5,
#                                    monitor='loss',
#                                    verbose=1,
#                                    save_best_only=True)
# # 开始训练
# history = model.fit_generator(myGene,
#                               steps_per_epoch=500,
#                               epochs=2,
#                               callbacks=[model_checkpoint])
# model.save_weights(PATH_MODEL_HDF5)
# err = model.evaluate_generator(validation_generator, steps=500)
# print(err)

results = model.predict_generator(validation_generator, 160, verbose=1)
print(np.argmax(results, axis=1))

# 图形化显示acc
# plt.figure(1)
# plt.plot(history.history['acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # 图形化显示loss
# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')

# plt.show()
