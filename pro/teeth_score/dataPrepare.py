#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data import *


# # data augmentation 
# 
# In deep learning tasks, a lot of data is need to train DNN model, when the dataset is not big enough, data augmentation should be applied.
# 
# keras.preprocessing.image.ImageDataGenerator is a data generator, which can feed the DNN with data like : (data,label), it can also do data augmentation at the same time.
# 
# It is very convenient for us to use keras.preprocessing.image.ImageDataGenerator to do data augmentation by implement image rotation, shift, rescale and so on... see [keras documentation](https://keras.io/preprocessing/image/) for detail.
# 
# For image segmentation tasks, the image and mask must be transformed **together!!**

# ## define your data generator
# 
# If you want to visualize your data augmentation result, set save_to_dir = your path

# In[6]:


#if you don't want to do data augmentation, set data_gen_args as an empty dict.
#data_gen_args = dict()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGenerator = trainGenerator(20,'data/membrane/train','image','label',data_gen_args,save_to_dir = "data/membrane/train/aug")


# ## visualize your data augmentation result

# In[8]:


#you will see 60 transformed images and their masks in data/membrane/train/aug
num_batch = 3
for i,batch in enumerate(myGenerator):
    if(i >= num_batch):
        break


# ## create .npy data
# 
# If your computer has enough memory, you can create npy files containing all your images and masks, and feed your DNN with them.

# In[9]:


image_arr,mask_arr = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#np.save("data/image_arr.npy",image_arr)
#np.save("data/mask_arr.npy",mask_arr)

