import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

INPUT_SHAPE = (128,128,4)

def Conv2D_Input(x, filters):
    x = layers.Conv2D(filters,
                (7,7),
                strides=2,
                input_shape=INPUT_SHAPE)(x)
    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2)(x)
    return x

def Conv2D_transition(x, k):
    x = layers.Conv2D(k,
                (1,1),
                padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2),
                            strides=2,
                            padding='valid')(x)
    return x


def MLBlock(x,k1,k2):
    x_add = layers.BatchNormalization()(x)
    x_add = layers.ReLU()(x_add)
    x_add = layers.Conv2D(k1,
                (1, 1),
                padding='same')(x_add)
    x_add = layers.BatchNormalization()(x_add)
    x_add = layers.ReLU()(x_add)
    x_add = layers.Conv2D(k1,
                (3, 3),
                padding='same')(x_add)


    x_concat = layers.BatchNormalization()(x)
    x_concat = layers.ReLU()(x_concat)
    x_concat = layers.Conv2D(k2,
                (1, 1),
                padding='same')(x_concat)
    x_concat = layers.BatchNormalization()(x_concat)
    x_concat = layers.ReLU()(x_concat)
    x_concat = layers.Conv2D(k2,
                (3, 3),
                padding='same')(x_concat)
    x_add = tf.keras.layers.Add()([x[:,:,:,-k1:], x_add])
    x = layers.Concatenate(axis=3)([x[:,:,:,:-k1], x_add])

    x = layers.Concatenate(axis=3)([x, x_concat])
    return x

def build_MixNet_model(k,MLB1,MLB2,MLB3,MLB4):
    # inputs
    inputs = layers.Input(shape=INPUT_SHAPE)
    x_0 = layers.BatchNormalization()(inputs)
    x_1 = Conv2D_Input(x_0, 2*k)

    for i in range(MLB1):
        x_1 = MLBlock(x_1,k,k)
        x_1 = MLBlock(x_1,k,k)

    x_2 = Conv2D_transition(x_1, k)

    for i in range(MLB2):
        x_2 = MLBlock(x_2,k,k)
        x_2 = MLBlock(x_2,k,k)

    # x_4 = x_1

    x_3 = Conv2D_transition(x_2, k)

    for i in range(MLB3):
        x_3 = MLBlock(x_3,k,k)
        x_3 = MLBlock(x_3,k,k)

    x_4 = Conv2D_transition(x_3, k)

    for i in range(MLB4):
        x_4 = MLBlock(x_4,k,k)
        x_4 = MLBlock(x_4,k,k)

    x_4 = layers.BatchNormalization()(x_4)
    x_4 = layers.ReLU()(x_4)
    x_pooled = layers.GlobalAveragePooling2D(data_format='channels_last')(x_4)

    outputs = layers.Dense(6, activation='softmax')(x_pooled)

    model = tf.keras.Model(inputs, outputs, name="ML-Net")
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt)

    return model


if __name__ == '__main__':
    model = build_MixNet_model(48,6,12)
