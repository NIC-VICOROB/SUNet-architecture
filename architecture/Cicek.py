import numpy as np
import math

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from .Architecture import Architecture

K.set_image_dim_ordering('th')

class Cicek (Architecture):
    @staticmethod
    def get_model(config, crossval_id=None) :
        assert config.arch.num_dimensions in [2, 3]

        num_modalities = len(config.dataset.modalities)
        input_layer_shape = (num_modalities, ) + config.arch.patch_shape[:config.arch.num_dimensions]
        output_layer_shape = (config.train.num_classes, np.prod(config.arch.patch_shape[:config.arch.num_dimensions]))

        model = generate_unet_model(
            config.arch.num_dimensions,
            config.train.num_classes,
            input_layer_shape,
            output_layer_shape,
            config.train.activation,
            downsize_factor=2)

        return model

def generate_unet_model(
    dimension, num_classes, input_shape, output_shape, activation, downsize_factor=2):
    input = Input(shape=input_shape)

    conv1 = get_conv_core(dimension, input, int(math.floor(64/downsize_factor)))
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_conv_core(dimension, pool1, int(math.floor(128/downsize_factor)))
    pool2 = get_max_pooling_layer(dimension, conv2)

    conv3 = get_conv_core(dimension, pool2, int(math.floor(256/downsize_factor)))
    pool3 = get_max_pooling_layer(dimension, conv3)

    conv4 = get_conv_core(dimension, pool3, int(math.floor(512/downsize_factor)))

    up5 = get_deconv_layer(dimension, conv4, int(math.floor(256/downsize_factor)))
    up5 = concatenate([up5, conv3], axis=1)

    conv5 = get_conv_core(dimension, up5, int(math.floor(256/downsize_factor)))

    up6 = get_deconv_layer(dimension, conv5, int(math.floor(128/downsize_factor)))
    up6 = concatenate([up6, conv2], axis=1)

    conv6 = get_conv_core(dimension, up6, int(math.floor(128/downsize_factor)))

    up7 = get_deconv_layer(dimension, conv6, int(math.floor(64/downsize_factor)))
    up7 = concatenate([up7, conv1], axis=1)

    conv7 = get_conv_core(dimension, up7, int(math.floor(64/downsize_factor)))

    pred = get_conv_fc(dimension, conv7, num_classes)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input], outputs=[pred])


def get_conv_core(dimension, input, num_filters) :
    x = None
    kernel_size = (3, 3) if dimension == 2 else (3, 3, 3)

    if dimension == 2 :
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(input)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
    else :
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(input)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)
        x = Conv3D(num_filters, kernel_size=kernel_size, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=1)(x)

    return x

def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(pool_size=pool_size)(input)

def get_deconv_layer(dimension, input, num_filters) :
    strides = (2, 2) if dimension == 2 else (2, 2, 2)
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)

def get_conv_fc(dimension, input, num_filters) :
    fc = None
    kernel_size = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        fc = Conv2D(num_filters, kernel_size=kernel_size)(input)
    else :
        fc = Conv3D(num_filters, kernel_size=kernel_size)(input)

    return Activation('relu')(fc)

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)