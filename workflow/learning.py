from utils.instructions import *

from workflow.generators import TrainPatchGenerator, TestPatchGenerator

def build_learning_generators(config, dataset):
    log.info("Building training generator...")
    train_generator = TrainPatchGenerator(config, dataset.train, augment=True, sampling=config.train.sampling)

    log.info("Building validation generator...")
    val_generator = TrainPatchGenerator(config, dataset.val, is_validation=True, sampling=config.train.sampling)

    return train_generator, val_generator


def build_training_generator(config, sample_in, is_validation=False, augment=False):
    return TrainPatchGenerator(config, sample_in, augment=augment, is_validation=is_validation)


def build_testing_generator(config, sample_in):
    return TestPatchGenerator(config, sample_in)

