import abc #abstract base classes
from utils.loss_functions import *
import keras.optimizers as Koptimizers

import itertools as iter
    

from utils.patch import *
from utils.slices import *
from utils.centers import *

from utils.instructions import *

class Architecture:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def generate_compiled_model(self, config, crossval_id=None):
        """
        Builds and compiles the keras model associated with this architecture
        """
        model = self.get_model(config, crossval_id=crossval_id)

        model.compile(
            loss=Architecture.get_loss_func_object(config),
            optimizer=Architecture.get_optimizer_object(config),
            metrics=config.train.metrics)

        return model

    @staticmethod
    @abc.abstractmethod
    def get_model(config, crossval_id=None):
        """
        Builds the keras model associated with this architecture
        """
        pass

    @staticmethod
    def generate_global_volume(image_vol):
        return None

    @staticmethod
    def get_patch_extraction_instructions(config, sample_idx, centers, image_vol, labels_vol, augment=False):
        """
        Returns patch extraction instructions as required by the architecture
        (Specially useful for 2.5D and multipatch architectures)
        """
        data_slices = get_patch_slices(centers, config.arch.patch_shape)
        label_slices = get_patch_slices(centers, config.arch.output_shape)

        sample_instructions = list()
        for data_slice, label_slice in iter.izip(data_slices, label_slices):
            instruction = PatchExtractInstruction(
                sample_idx=sample_idx, data_patch_slice=data_slice, label_patch_slice=label_slice)
            sample_instructions.append(instruction)

        if augment:
            goal_num_patches = math.ceil((1 - config.train.uniform_ratio) * config.train.num_patches)
            sample_instructions = augment_instructions(sample_instructions, goal_num_patches)

        return sample_instructions

    @staticmethod
    def get_optimizer_object(config):
        """
        Returns configured optimizer object (i.e. with appropiate learning_rate, batch_size...)
        :param config:
        :return: Optimizer object with appropiate configuration
        """

        if config.train.optimizer is 'Adam':
            optimizer = Koptimizers.Adam(lr=config.train.learning_rate,
                                         decay=config.train.lr_decay)
        elif config.train.optimizer is 'Adadelta':
            optimizer = Koptimizers.Adadelta(lr=config.train.learning_rate)
        elif config.train.optimizer is 'Adagrad':
            optimizer = Koptimizers.Adagrad(lr=config.train.learning_rate)
        else:
            # If specified optimizer not implemented return string for default parameters
            optimizer = config.train.optimizer

        return optimizer

    @staticmethod
    def get_loss_func_object(config):
        loss_dictionary = {
            'jaccard' : jaccard,
            'categorical_dice' : dice,
            'sensitivity_specificity' : ss
        }

        if config.train.loss in loss_dictionary:
            loss_func = loss_dictionary[config.train.loss]
        else:
            loss_func = config.train.loss

        return loss_func