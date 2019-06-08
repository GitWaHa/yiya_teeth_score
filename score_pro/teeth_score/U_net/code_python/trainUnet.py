#!/usr/bin/env python
# coding: utf-8 

# In[1]:


from model import *
from data import *
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TARGET_INPUT_SIZE = (128, 128)
MODEL_INPUT_SIZE = TARGET_INPUT_SIZE + (1, )
PATH_MODEL_HDF5 = "../model_hdf5/unet_128.hdf5"

# 数据生成器参数
data_gen_args = dict(rotation_range=15,			# 随机旋转角度
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,			# 剪切强度（以弧度逆时针方向剪切角度）
                    zoom_range=0.05,			# 随机缩放范围。如果是浮点数，[lower, upper] = [1-zoom_range, 1+zoom_range]。
                    horizontal_flip=True,		# 随机水平翻转
                    fill_mode='nearest')		# 填充模式
myGene = trainGenerator(2,'../data/train','image','label',data_gen_args,save_to_dir = None, target_size = TARGET_INPUT_SIZE) # save_to_dir = 'data/trainGenerator'

# 模型
model = unet(input_size = MODEL_INPUT_SIZE)
# 继续训练则加载已存在的模型参数
model.load_weights(PATH_MODEL_HDF5)

# 当loss 减小则保存，否则跳过
model_checkpoint = ModelCheckpoint(PATH_MODEL_HDF5, monitor='loss',verbose=1, save_best_only=True)
# 开始训练
history = model.fit_generator(myGene, steps_per_epoch=500, epochs=5, callbacks=[model_checkpoint])
# model.save_weights("unet_128.hdf5")


# 图形化显示acc
plt.figure(1)
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# 图形化显示loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.show()



