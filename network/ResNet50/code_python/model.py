#!/usr/bin/python3.6
#coding=utf-8
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam


def MyResNet50(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    base_model = ResNet50(include_top=False,
                          weights='imagenet',
                          input_tensor=inputs,
                          pooling='avg',
                          classes=3)
    # base_model.add()
    dense = Dense(1024, activation='relu')(base_model.layers[-1].output)
    dense = Dropout(0.5)(dense)

    output = Dense(3, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.summary()

    return model

    #     conv1 = Conv2D(96, (11, 11),
    #                     strides=(4, 4),
    #                     input_shape=input_size,
    #                     padding='valid',
    #                     activation='relu',
    #                     kernel_initializer='uniform')(inputs)
    #     pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

    #     conv2 = Conv2D(256, (5, 5),
    #             strides=(1, 1),
    #             padding='same',
    #             activation='relu',
    #             kernel_initializer='uniform')(pool1)
    #     pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)

    #     conv3 = Conv2D(384, (3, 3),
    #             strides=(1, 1),
    #             padding='same',
    #             activation='relu',
    #             kernel_initializer='uniform')(pool2)

    #     conv4 = Conv2D(384, (3, 3),
    #             strides=(1, 1),
    #             padding='same',
    #             activation='relu',
    #             kernel_initializer='uniform')(conv3)

    #     conv5 = Conv2D(256, (3, 3),
    #             strides=(1, 1),
    #             padding='same',
    #             activation='relu',
    #             kernel_initializer='uniform')(conv4)
    #     pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv5)

    #     flatten = Flatten()(pool3)
    #     dense1 = Dense(4096, activation='relu')(flatten)
    #     dense1 = Dropout(0.5)(dense1)

    #     dense2 = Dense(1024, activation='relu')(dense1)
    #     dense2 = Dropout(0.5)(dense2)

    #     output = Dense(2, activation='softmax')(dense2)

    #     model = Model(inputs = inputs, outputs = output)
    #     model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
