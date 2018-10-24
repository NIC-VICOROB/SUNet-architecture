import numpy as np
import logging as log

from keras import backend as K
from keras.layers import Input, Dropout, Add, Softmax, PReLU, Concatenate, Maximum
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv3D, Conv3DTranspose, MaxPooling3D, UpSampling3D

from keras.layers.core import Permute, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from .Architecture import Architecture

K.set_image_dim_ordering('th')


class SUNETx4(Architecture):
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
            config.arch.dropout_rate,
            config.arch.base_filters
        )

        trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
        log.debug("SUNETx4 model with {} trainable parameters".format(trainable_count))

        return model


def generate_uresnet_model(
    ndims, num_classes, input_shape, output_shape, dropout_rate=0.2, filts=32):

    l1_input = Input(shape=input_shape)
    l1_start = get_conv_input(ndims, l1_input, 1*filts)

    l1_end = get_dual_res_layer(ndims, l1_start, 1*filts)
    l2_start = get_downconvolution_layer(ndims, l1_end, 1*filts)

    l2_end = get_dual_res_layer(ndims, l2_start, 2*filts)
    l3_start = get_downconvolution_layer(ndims, l2_end, 2*filts)

    l3_end = get_dual_res_layer(ndims, l3_start, 4*filts)
    l4_start = get_downconvolution_layer(ndims, l3_end, 4*filts)

    l4_latent = get_dual_res_layer(ndims, l4_start, 8*filts, dropout_rate)
    r4_upped = get_upconvolution_layer(ndims, l4_latent, 4*filts)

    r3_start = Add()([l3_end, r4_upped])
    r3_end = get_mono_res_layer(ndims, r3_start, num_filters=4*filts)
    r3_upped = get_upconvolution_layer(ndims, r3_end, num_filters=2*filts)

    r2_start = Add()([l2_end, r3_upped])
    r2_end = get_mono_res_layer(ndims, r2_start, num_filters=2*filts)
    r2_upped = get_upconvolution_layer(ndims, r2_end, num_filters=1*filts)

    r1_start = Add()([l1_end, r2_upped])
    r1_end = get_mono_res_layer(ndims, r1_start, num_filters=1*filts)
    pred = get_conv_output(ndims, r1_end, num_classes, output_shape)

    return Model(inputs=[l1_input], outputs=[pred])


def get_dual_res_layer(ndims, layer_in, num_filters, dropout_rate=0.2):
    Conv = Conv2D if ndims == 2 else Conv3D
    kernel_size_a = (3, 3) if ndims == 2 else (3, 3, 3)

    a = BatchNormalization(axis=1)(layer_in)
    a = PReLU(shared_axes=(2,3) if ndims is 2 else (2,3,4))(a)
    a = Conv(num_filters, kernel_size=kernel_size_a, padding='same')(a)

    a = Dropout(dropout_rate)(a)

    a = BatchNormalization(axis=1)(a)
    a = PReLU(shared_axes=(2,3) if ndims is 2 else (2,3,4))(a)
    a = Conv(num_filters, kernel_size=kernel_size_a, padding='same')(a)

    layer_out = Add()([a, layer_in])
    return layer_out


def get_mono_res_layer(ndims, layer_in, num_filters, dropout_rate=0.0):
    Conv = Conv2D if ndims == 2 else Conv3D
    kernel_size_a = (3, 3) if ndims == 2 else (3, 3, 3)

    a = BatchNormalization(axis=1)(layer_in)
    a = PReLU(shared_axes=(2,3) if ndims is 2 else (2,3,4))(a)
    a = Conv(num_filters, kernel_size=kernel_size_a, padding='same')(a)
    a = Dropout(dropout_rate)(a)

    layer_out = Add()([a, layer_in])

    return layer_out


def get_upconvolution_layer(ndims, layer_in, num_filters):
    kernel_size = (3, 3) if ndims == 2 else (3, 3, 3)
    strides = (2, 2) if ndims == 2 else (2, 2, 2)
    ConvTranspose = Conv2DTranspose if ndims == 2 else Conv3DTranspose

    layer_out = ConvTranspose(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_in)

    return layer_out


def get_downconvolution_layer(ndims, layer_in, num_filters, dropout_rate=0.0):
    Conv = Conv2D if ndims == 2 else Conv3D
    MaxPooling = MaxPooling2D if ndims == 2 else MaxPooling3D
    kernel_size_a = (3, 3) if ndims == 2 else (3, 3, 3)

    # Low Res Branch
    a_halved = BatchNormalization(axis=1)(layer_in)
    a_halved = PReLU(shared_axes=(2,3) if ndims is 2 else (2,3,4))(a_halved)
    a_halved = Conv(num_filters, kernel_size=kernel_size_a, padding='same', strides=2)(a_halved)
    a_halved = Dropout(dropout_rate)(a_halved)

    # MP branch
    b_mp = MaxPooling(pool_size=(2, 2) if ndims == 2 else (2, 2, 2))(layer_in)

    layer_out_halved = Concatenate(axis=1)([a_halved, b_mp])

    return layer_out_halved


def get_conv_input(ndims, layer_in, num_filters):
    Conv = Conv2D if ndims == 2 else Conv3D
    kernel_size = (3, 3) if ndims == 2 else (3, 3, 3)

    layer_out = Conv(num_filters, kernel_size=kernel_size, padding='same')(layer_in)

    return layer_out


def get_conv_output(ndims, layer_in, num_filters, output_shape):
    Conv = Conv2D if ndims == 2 else Conv3D
    kernel_size = (1, 1) if ndims == 2 else (3, 1, 1)

    pred = Conv(num_filters, kernel_size=kernel_size, padding='same')(layer_in)
    pred = Reshape(output_shape)(pred)
    pred = Permute((2, 1))(pred)  # Put classes in last dimension
    pred = Softmax(axis=-1)(pred)  # Apply softmax on last dimension

    return pred