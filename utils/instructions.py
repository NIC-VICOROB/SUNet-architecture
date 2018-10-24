import sys
import copy
import math
import numpy as np
import logging as log
import scipy.misc
import itertools as iter

from keras.utils import np_utils, Sequence
import keras.backend as K

from utils.patch import *
from utils.samples import *
from utils.visualization import *
from utils.misc import *

class PatchExtractInstruction:
    def __init__(self, sample_idx=-1, data_patch_slice=None, label_patch_slice=None, augment_func=None):
        self.sample_idx = sample_idx

        self.data_patch_slice = data_patch_slice
        self.label_patch_slice = label_patch_slice

        self.augment_func = augment_func


def extract_patch_with_instruction(samples, instruction):
    assert isinstance(instruction, PatchExtractInstruction)
    sample = samples[instruction.sample_idx] if isinstance(samples, list) else samples

    extract_label = instruction.label_patch_slice is not None and sample.labels is not None

    # Extract patches
    data_patch = extract_patch_at(sample.data, instruction.data_patch_slice)
    label_patch = extract_patch_at(sample.labels, instruction.label_patch_slice) if extract_label else None

    # Augment patches
    if instruction.augment_func is not None:
        augment_func = get_augment_functions(x_axis=1, y_axis=2)[instruction.augment_func]
        data_patch = augment_func(data_patch)
        label_patch = augment_func(label_patch) if extract_label else None

    return data_patch, label_patch


def augment_instructions(original_instructions, goal_num_instructions):
    augment_funcs = get_augment_functions(x_axis=1, y_axis=2)  # (modality, x, y, z)

    num_patches_in = len(original_instructions)
    num_augments_per_patch = np.minimum( int(math.ceil(goal_num_instructions / num_patches_in)), len(augment_funcs))
    goal_num_augmented_patches = int(goal_num_instructions - num_patches_in)

    # Augment and add remaining copies
    sampling_idxs = get_resampling_indexes(num_patches_in, goal_num_augmented_patches // num_augments_per_patch)
    func_idxs = get_resampling_indexes(len(augment_funcs), num_augments_per_patch)

    augmented_instructions = list()
    for sampling_idx, func_idx in iter.product(sampling_idxs, func_idxs):
        aug_instr = copy.copy(original_instructions[sampling_idx])
        aug_instr.augment_func = func_idx
        augmented_instructions.append(aug_instr)

    final_instructions = original_instructions + augmented_instructions

    return final_instructions