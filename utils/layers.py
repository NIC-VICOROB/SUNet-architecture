from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class ZeroPaddingChannel(Layer):
    def __init__(self, padding, bs=32, **kwargs):
        self.padding = padding
        self.batch_size = bs
        super(ZeroPaddingChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ZeroPaddingChannel, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        shape_padding = list(x.shape)
        shape_padding[1] = self.padding
        y = K.zeros([self.batch_size] + shape_padding[1:])
        return K.concatenate([x, y], axis=1)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 5
        shape[1] += self.padding
        return tuple(shape)
