import logging as log
import numpy as np
import math
import itertools as iter
import scipy.ndimage as ndimage
import copy
import sys

from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
from utils.visualization import normalize_and_save_patch
from utils.misc import *

from utils.slices import get_in_bounds_slices_index, get_patch_slices

def sample_uniform_patch_centers(patch_shape, extraction_step, volume_foreground):
    """
    Generate patch center indexes for 3D volume uniform sampling
    patch_shape, extraction step and expected shape should be in 3D!

    volume_foreground = sample.data, is used to avoid adding centers for which the center of the patch is empty
    """
    expected_shape = volume_foreground.shape[1:]

    assert len(patch_shape) == len(extraction_step) == len(expected_shape) == 3

    # Get patch half size
    half_sizes = [[dim // 2, dim // 2] for dim in patch_shape]
    for i in range(len(half_sizes)):  # If even dimension, subtract 1 to account for assymetry
        if patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    idxs = []
    for dim in range(len(expected_shape)):
        idxs.append(range(half_sizes[dim][0], expected_shape[dim] - half_sizes[dim][1], extraction_step[dim]))

    # Put in ndarray format
    centers = np.zeros((np.prod([len(dim_idxs) for dim_idxs in idxs]), len(idxs)), dtype=int)

    # Make sure added center is foreground
    added_centers = 0

    for count, center_coords in enumerate(iter.product(*idxs)):
        if volume_foreground[0, center_coords[0], center_coords[1], center_coords[2]] != 0:
            centers[added_centers] = np.asarray(center_coords)
            added_centers += 1

    centers = centers[:added_centers] # Get filled centers

    centers = clip_centers_out_of_bounds(centers, patch_shape, volume_foreground)

    return centers


def sample_positive_patch_centers(labels_volume):
    if len(labels_volume.shape) > 3:
        labels_volume = labels_volume[0] # Select modality 0

    pos_centers = np.where(labels_volume == 1)

    # Put in ndarray format
    centers = np.zeros((len(pos_centers[0]), len(pos_centers)), dtype=int)
    for j, dim_coords in enumerate(pos_centers):
        for i, dim_coord in enumerate(dim_coords):
            centers[i][j] = dim_coord

    return centers


def randomly_offset_centers(centers, offset_shape, patch_shape, original_vol):
    ### Offset patches to avoid location bias

    # Create offset matrix
    center_offsets = 2.0 * (np.random.rand(centers.shape[0], centers.shape[1]) - 0.5)  # generate uniform sampling between -1 and 1
    offset_ranges = np.stack([(offset_shape[0] // 2) * np.ones((centers.shape[0],)),
                              (offset_shape[1] // 2) * np.ones((centers.shape[0],)),
                              (offset_shape[2] // 2) * np.ones((centers.shape[0],))], axis=1)
    center_offsets = np.multiply(center_offsets, offset_ranges).astype(int)

    # Apply offset
    centers += center_offsets

    # Check centers in-bounds
    centers = clip_centers_out_of_bounds(centers, patch_shape, original_vol)

    return centers



def resample_centers(centers, min_samples=None, max_samples=None):
    resampled_idxs = None
    if min_samples is not None:
        if centers.shape[0] < min_samples:
            resampled_idxs = get_resampling_indexes(centers.shape[0], min_samples)

    if max_samples is not None:
        if centers.shape[0] > max_samples:
            resampled_idxs = get_resampling_indexes(centers.shape[0], max_samples)

    if resampled_idxs is not None:  # Perform resampling
        resampled_idxs = np.asarray(resampled_idxs, dtype=np.int)
        centers = centers[resampled_idxs]

    return centers

def clip_centers_out_of_bounds(centers, patch_shape, original_vol):
    vol_shape = original_vol.shape[1:] # Omit modality dimension
    center_range = [[(patch_shape[0] // 2) + 1, vol_shape[0] - (patch_shape[0] // 2) - 1],
                    [(patch_shape[1] // 2) + 1, vol_shape[1] - (patch_shape[1] // 2) - 1],
                    [(patch_shape[2] // 2) + 1, vol_shape[2] - (patch_shape[2] // 2) - 1]]
    center_range = np.asarray(center_range, dtype='int')

    for i in [0,1,2]:
        centers[:, i] = np.clip(centers[:, i], a_min=center_range[i][0], a_max=center_range[i][1])

    return centers