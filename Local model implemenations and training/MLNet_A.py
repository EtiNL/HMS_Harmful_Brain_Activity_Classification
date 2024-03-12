import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

def Conv2D_Input(x, filters):
    x = layers.Conv2D(filters,
                (3,3),
                input_shape=(48, 48, 4),
                padding='same')(x)
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

def build_MLNet_model(k):
    # inputs
    inputs = layers.Input(shape=(48, 48, 4))
    x_0 = layers.BatchNormalization()(inputs)
    x_0 = Conv2D_Input(x_0, 2*k)
    x_1 = MLBlock(x_0,k,k)
    x_2 = MLBlock(x_1,k,k)
    x_3 = MLBlock(x_2,k,k)
    x_3 = layers.BatchNormalization()(x_3)
    x_3 = layers.ReLU()(x_3)
    x_pooled = layers.GlobalAveragePooling2D(data_format='channels_last')(x_3)

    outputs = layers.Dense(6, activation='softmax')(x_pooled)

    model = tf.keras.Model(inputs, outputs, name="ML-Net")
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = tf.keras.losses.KLDivergence()

    model.compile(loss=loss, optimizer = opt)

    return model


if __name__ == '__main__':
    model = build_MLNet_model(36)
