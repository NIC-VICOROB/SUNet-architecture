import numpy as np

from keras import backend as K
from keras.layers import Activation, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D
from keras.layers.core import Permute, Reshape
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from .Architecture import Architecture

K.set_image_dim_ordering('th')

class Guerrero (Architecture):
    @staticmethod
    def get_model(config, crossval_id=None):
        assert config.arch.num_dimensions in [2, 3]

        num_modalities = len(config.dataset.modalities)
        input_layer_shape = (num_modalities,) + config.arch.patch_shape[:config.arch.num_dimensions]
        output_layer_shape = (config.train.num_classes, np.prod(config.arch.patch_shape[:config.arch.num_dimensions]))

        model = generate_uresnet_model(
            config.arch.num_dimensions,
            config.train.num_classes,
            input_layer_shape,
            output_layer_shape,
            config.train.activation)

        return model


def generate_uresnet_model(
    dimension, num_classes, input_shape, output_shape, activation):
    input_uresnet = Input(shape=input_shape)

    conv1 = get_res_conv_core(dimension, input_uresnet, 32)
    pool1 = get_max_pooling_layer(dimension, conv1)

    conv2 = get_res_conv_core(dimension, pool1, 64)
    pool2 = get_max_pooling_layer(dimension, conv2)

    conv3 = get_res_conv_core(dimension, pool2, 128)
    pool3 = get_max_pooling_layer(dimension, conv3)

    conv4 = get_res_conv_core(dimension, pool3, 256)
    up1 = get_deconv_layer(dimension, conv4, 128)
    conv5 = get_res_conv_core(dimension, up1, 128)

    add35 = add([conv3, conv5])
    add35 = BatchNormalization(axis=1)(add35)
    add35 = Activation('relu')(add35)
    conv6 = get_res_conv_core(dimension, add35, 128)
    up2 = get_deconv_layer(dimension, conv6, 64)

    add22 = add([conv2, up2])
    add22 = BatchNormalization(axis=1)(add22)
    add22 = Activation('relu')(add22)
    conv7 = get_res_conv_core(dimension, add22, 64)
    up3 = get_deconv_layer(dimension, conv7, 32)

    add13 = add([conv1, up3])
    add13 = BatchNormalization(axis=1)(add13)
    add13 = Activation('relu')(add13)
    conv8 = get_res_conv_core(dimension, add13, 32)

    pred = get_conv_fc(dimension, conv8, num_classes)
    pred = organise_output(pred, output_shape, activation)

    return Model(inputs=[input_uresnet], outputs=[pred])

def get_res_conv_core(dimension, input, num_filters) :
    kernel_size_a = (3, 3) if dimension == 2 else (3, 3, 3)
    kernel_size_b = (1, 1) if dimension == 2 else (1, 1, 1)

    if dimension == 2 :
        a = Conv2D(num_filters, kernel_size=kernel_size_a, padding='same')(input)
        a = BatchNormalization(axis=1)(a)
        b = Conv2D(num_filters, kernel_size=kernel_size_b, padding='same')(input)
        b = BatchNormalization(axis=1)(b)
    else :
        a = Conv3D(num_filters, kernel_size=kernel_size_a, padding='same')(input)
        a = BatchNormalization(axis=1)(a)
        b = Conv3D(num_filters, kernel_size=kernel_size_b, padding='same')(input)
        b = BatchNormalization(axis=1)(b)

    c = add([a, b])
    c = BatchNormalization(axis=1)(c)
    return Activation('relu')(c)

def get_max_pooling_layer(dimension, input) :
    pool_size = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return MaxPooling2D(pool_size=pool_size)(input)
    else :
        return MaxPooling3D(pool_size=pool_size)(input)

def get_deconv_layer(dimension, input, num_filters) :
    kernel_size = (2, 2) if dimension == 2 else (2, 2, 2)
    strides = (2, 2) if dimension == 2 else (2, 2, 2)

    if dimension == 2:
        return Conv2DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)
    else :
        return Conv3DTranspose(num_filters, kernel_size=kernel_size, strides=strides)(input)

def get_conv_fc(dimension, input, num_filters) :
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