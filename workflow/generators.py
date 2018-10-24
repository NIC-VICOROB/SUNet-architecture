import sys
import copy
import math
import numpy as np
import logging as log
import nibabel as nib
import scipy.misc
import itertools as iter

from keras.utils import Sequence
import keras.backend as K

from utils.patch import *
from utils.samples import *
from utils.visualization import *
from utils.centers import *
from utils.instructions import *
from utils.slices import *
from utils.volume import *

from random import shuffle

from workflow.in_out import get_architecture_instance

class TestPatchGenerator(Sequence):
    def __init__(self, config, sample_in):
        # Important parameters
        self.config = config
        self.stats = compute_sample_statistics(sample_in)
        self.batch_size = config.train.batch_size

        # Store volumes and patch extraction instructions
        self.sample = sample_in
        self.centers = sample_uniform_patch_centers(config.arch.patch_shape, config.train.extraction_step_test, self.sample.data)

        self.sample = normalise_sample(sample_in, self.stats[0], self.stats[1])
        self.instructions = get_architecture_instance(config).get_patch_extraction_instructions(
            config, 0, self.centers, self.sample.data, self.sample.labels, augment=False)

        log.debug("Test generator: extracted {} instructions, making {} batches".format(
            len(self.instructions), int(math.ceil(len(self.instructions) / float(self.batch_size)))))

    def __len__(self):
        return int(math.ceil(len(self.instructions) / float(self.batch_size)))

    def __getitem__(self, idx_global):
        batch_start = idx_global * self.batch_size
        batch_end = np.minimum((idx_global + 1) * self.batch_size, len(self.instructions))
        batch_slice = slice(batch_start, batch_end)

        # Get instructions for current batch creation
        batch_instructions = self.instructions[batch_slice]
        batch_length = len(batch_instructions)
        assert batch_length <= self.batch_size, "Returning bigger batch than expected"

        # Allocate space for batch data
        ndims = self.config.arch.num_dimensions
        nmodalities = len(self.config.dataset.modalities)
        x = np.zeros((batch_length, nmodalities) + self.config.arch.patch_shape[:ndims], dtype=np.float32)

        # Extract patches as stated by instructions
        for idx, instruction in enumerate(batch_instructions):
            data_patch, _ = extract_patch_with_instruction(self.sample, instruction)
            x[idx] = data_patch

        return x

class TrainPatchGenerator(Sequence):
    def __init__(self, config, samples_in, augment=False, is_validation=False, sampling='hybrid'):
        # Important parameters
        self.config = config
        self.stats = compute_set_statistics(samples_in)
        self.batch_size = config.train.batch_size
        self.is_validation = is_validation

        # Store volumes and patch extraction instructions
        self.samples = zeropad_set(copy.deepcopy(samples_in), config.arch.patch_shape)
        self.instructions = build_set_extraction_instructions(config, self.samples, augment=augment, sampling=sampling)
        shuffle(self.instructions)

        log.debug("Train generator: extracted {} instructions, making {} batches".format(
            len(self.instructions), int(math.ceil(len(self.instructions) / float(self.batch_size)))))

    def on_epoch_end(self):
        shuffle(self.instructions)

    def __len__(self):
        num_batches = len(self.instructions) / float(self.batch_size)
        return int(math.floor(num_batches)) if self.is_validation else int(math.ceil(num_batches))

    def __getitem__(self, idx_global):
        batch_start = idx_global * self.batch_size
        batch_end = np.minimum((idx_global + 1) * self.batch_size, len(self.instructions))
        batch_slice = slice(batch_start, batch_end)

        # Get instructions for current batch creation
        batch_instructions = self.instructions[batch_slice]
        batch_length = len(batch_instructions)

        if self.is_validation: assert batch_length == self.batch_size, "In validation cannot return partial batches"
        else: assert batch_length <= self.batch_size, "Training generator is returning bigger batch than it should"

        # Allocate space for batch data
        ndims = self.config.arch.num_dimensions
        nmodalities = len(self.config.dataset.modalities)

        x = np.zeros((batch_length, nmodalities) + self.config.arch.patch_shape[:ndims], dtype=np.float32)
        y = np.zeros((batch_length, np.prod(self.config.arch.output_shape[:ndims]), self.config.train.num_classes))

        # Extract patches as stated by instructions
        for idx, instruction in enumerate(batch_instructions):
            data_patch, label_patch = extract_patch_with_instruction(self.samples, instruction)

            sample_mean = self.stats[0][instruction.sample_idx]
            sample_std = self.stats[1][instruction.sample_idx]

            # Store in batch
            try:
                x[idx] = normalise_patch(data_patch, sample_mean, sample_std)
                y[idx] = np_utils.to_categorical(label_patch.flatten(), self.config.train.num_classes)
            except Exception as e:
                print(e)
                print(instruction.sample_idx, instruction.data_patch_slice, instruction.label_patch_slice)


        return x, y


def build_set_extraction_instructions(config, samples_in, augment=False, sampling='hybrid'):
    set_instructions = list()
    for idx, sample in enumerate(samples_in):
        printProgressBar(idx, len(samples_in), suffix='samples processed')
        set_instructions += build_sample_patch_extraction_instructions(config, idx, sample, augment=augment, sampling=sampling)
    printProgressBar(len(samples_in), len(samples_in), suffix='samples processed')

    return set_instructions


def build_sample_patch_extraction_instructions(config, sample_idx, sample, augment=False, sampling='hybrid'):
    assert sampling in {'hybrid', 'Guerrero', 'Kamnitsas'}

    #log.debug("Using {} sampling strategy".format(sampling))

    if sampling is 'hybrid':
        ### --------------------------
        ### 1. Centers
        ### --------------------------

        # Positive
        pos_centers = sample_positive_patch_centers(sample.labels)
        pos_centers = resample_centers(
            pos_centers, min_samples=config.train.min_pos_patches, max_samples=config.train.max_pos_patches)

        offset_patch_shape = config.arch.patch_shape if not config.arch.offset_shape else config.arch.offset_shape

        pos_centers = randomly_offset_centers(
            pos_centers, offset_shape=offset_patch_shape, patch_shape=config.arch.patch_shape, original_vol=sample.data)

        # Uniform
        extraction_step = config.train.extraction_step
        while True:
            unif_centers = sample_uniform_patch_centers(config.arch.patch_shape, extraction_step, sample.data)
            unif_centers = resample_centers(unif_centers, max_samples=config.train.max_unif_patches)

            if unif_centers.shape[0] >= config.train.max_unif_patches - 1:
                break

            extraction_step = tuple([np.maximum(1, dim_step - 1) for dim_step in extraction_step])
            if np.array_equal(extraction_step, (1, 1, 1)):
                raise ValueError, "Cannot extract enough uniform patches, please decrease number of patches"

        ### -----------------------------------
        ### 2. Patch extraction instructions
        ### -----------------------------------

        arch = get_architecture_instance(config)
        pos_instructions = arch.get_patch_extraction_instructions(
            config, sample_idx, pos_centers, sample.data, sample.labels, augment=augment)
        unif_instructions = arch.get_patch_extraction_instructions(
            config, sample_idx, unif_centers, sample.data, sample.labels)

        sample_instructions = pos_instructions + unif_instructions

    elif sampling is 'Guerrero':
        ### --------------------------
        ### 1. Centers
        ### --------------------------

        # Positive
        pos_centers = sample_positive_patch_centers(sample.labels)
        pos_centers = resample_centers(
            pos_centers, min_samples=config.train.num_patches, max_samples=config.train.num_patches)

        offset_patch_shape = config.arch.patch_shape if not config.arch.offset_shape else config.arch.offset_shape
        pos_centers = randomly_offset_centers(
            pos_centers, offset_shape=offset_patch_shape, patch_shape=config.arch.patch_shape, original_vol=sample.data)

        ### -----------------------------------
        ### 2. Patch extraction instructions
        ### -----------------------------------

        arch = get_architecture_instance(config)
        pos_instructions = arch.get_patch_extraction_instructions(
            config, sample_idx, pos_centers, sample.data, sample.labels)

        sample_instructions = pos_instructions
    elif sampling is 'Kamnitsas':
        ### --------------------------
        ### 1. Centers
        ### --------------------------

        # Positive
        pos_centers = sample_positive_patch_centers(sample.labels)
        pos_centers = resample_centers(
            pos_centers, max_samples=config.train.max_pos_patches)

        # Uniform
        extraction_step = config.train.extraction_step
        while True:
            unif_centers = sample_uniform_patch_centers(config.arch.patch_shape, extraction_step, sample.data)
            unif_centers = resample_centers(unif_centers, max_samples=config.train.max_unif_patches)

            if unif_centers.shape[0] >= config.train.max_unif_patches - 1:
                break

            extraction_step = tuple([np.maximum(1, dim_step - 1) for dim_step in extraction_step])
            if np.array_equal(extraction_step, (1, 1, 1)):
                raise ValueError, "Cannot extract enough uniform patches, please decrease number of patches"

        ### -----------------------------------
        ### 2. Patch extraction instructions
        ### -----------------------------------

        arch = get_architecture_instance(config)
        pos_instructions = arch.get_patch_extraction_instructions(
            config, sample_idx, pos_centers, sample.data, sample.labels, augment=augment)
        unif_instructions = arch.get_patch_extraction_instructions(
            config, sample_idx, unif_centers, sample.data, sample.labels)

        sample_instructions = pos_instructions + unif_instructions
    else:
        raise ValueError


    return sample_instructions