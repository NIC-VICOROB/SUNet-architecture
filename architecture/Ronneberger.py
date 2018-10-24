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

class Ronneberger(Architecture):
    @staticmethod
    def get_model(config, crossval_id=None) :
        assert config.arch.num_dimensions is 2
        assert config.arch.patch_shape[0] % 16 == 0, "Invalid patch shape"

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

    L1_out = get_conv_core(input, num_filters=64)

    L2_in = get_max_pooling_layer(L1_out)
    L2_out = get_conv_core(L2_in, num_filters=128)

    L3_in = get_max_pooling_layer(L2_out)
    L3_out = get_conv_core(L3_in, num_filters=256)

    L4_in = get_max_pooling_layer(L3_out)
    L4_out = get_conv_core(L4_in, num_filters=512)

    C5_in = get_max_pooling_layer(L4_out)
    C5_out = get_conv_core(C5_in, num_filters=1024)

    R4_in = get_deconv_layer(C5_out, num_filters=256)
    R4_in = concatenate([R4_in, L4_out], axis=1)
    R4_out = get_conv_core(R4_in, num_filters=512)

    R3_in = get_deconv_layer(R4_out, num_filters=128)
    R3_in = concatenate([R3_in, L3_out], axis=1)
    R3_out = get_conv_core(R3_in, num_filters=256)

    R2_in = get_deconv_layer(R3_out, num_filters=64)
    R2_in = concatenate([R2_in, L2_out], axis=1)
    R2_out = get_conv_core(R2_in, num_filters=128)

    R1_in = get_deconv_layer(R2_out, num_filters=32)
    R1_in = concatenate([R1_in, L1_out], axis=1)
    R1_out = get_conv_core(R1_in, num_filters=64)

    out = get_conv_fc(R1_out, num_classes)
    pred = organise_output(out, output_shape, 'softmax')

    return Model(inputs=[input], outputs=[pred])


def get_conv_core(input, num_filters, batch_norm=False) :
    kernel_size = (3, 3)

    x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(input)
    x = Activation('relu')(x)
    if batch_norm: x = BatchNormalization(axis=1)(x)
    x = Conv2D(num_filters, kernel_size=kernel_size, padding='same')(x)
    x = Activation('relu')(x)
    if batch_norm: x = BatchNormalization(axis=1)(x)

    return x

def get_max_pooling_layer(input) :
    return MaxPooling2D(pool_size=(2, 2))(input)

def get_deconv_layer(input, num_filters) :
    return Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2))(input)

def get_conv_fc(input, num_classes):
    return Conv2D(num_classes, kernel_size=(1, 1))(input)

def organise_output(input, output_shape, activation) :
    pred = Reshape(output_shape)(input)
    pred = Permute((2, 1))(pred)
    return Activation(activation)(pred)