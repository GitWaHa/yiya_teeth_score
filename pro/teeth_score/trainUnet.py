#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import *
from data import *
import matplotlib.pyplot as plt


# ## Train your Unet with membrane data
# membrane data is in folder membrane/, it is a binary classification task.
#
# The input shape of image and mask are the same :(batch_size,rows,cols,channel = 1)

# ### Train with data generator

# In[2]:

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None) # save_to_dir = 'data/trainGenerator'

# model = unet()
# 继续训练
model = load_model('./unet_membrane.hdf5')

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
print("[INFO] train unet")
# model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])
history = model.fit_generator(myGene,steps_per_epoch=1000,epochs=5,callbacks=[model_checkpoint])


# list all data in history
print(history.history.keys())
# 图形化显示acc
# plt.subplot(2,2,1)
# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# 图形化显示loss
# plt.subplot(2,2,2)
# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Train with npy file

# In[ ]:


#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])


# ### test your model and save predicted results

# In[3]:

# print("\n [INFO] test unet")
# # 测试图片列表
# test_paths = sorted(glob.glob(os.path.join("data/membrane/test",'*.png')))
# # 测试图片数量
# imagenum_test = len(test_paths)
# # 测试图片生成器
# testGene = testGenerator(test_paths)
# model = unet()
# model.load_weights("unet_membrane.hdf5")
# results = model.predict_generator(testGene,imagenum_test,verbose=1)
# print(results.shape)
# saveResult("data/membrane/test_output", test_paths, results)

