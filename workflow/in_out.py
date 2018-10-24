import nibabel as nib
import logging as log
import numpy as np
import string
import os
import datetime

import cv2

from workflow.filenames import *

from dataset.isles2017 import Isles2017
from dataset.Isles15_SISS import Isles15_SISS
from dataset.Isles15_SPES import Isles15_SPES

from architecture.Cicek import Cicek
from architecture.Guerrero import Guerrero
from architecture.Ronneberger import Ronneberger
from architecture.SUNETx4 import SUNETx4

def get_architecture_instance(config):
    arch_name = config.train.architecture_name

    arch = get_architecture_class(arch_name)

    return arch()

def get_architecture_class(arch):
    # DONT FORGET TO IMPORT THE CLASS
    arch_dict = {
        '3Dunet': Cicek,
        '3Duresnet': Guerrero,
        '2Dunet': Ronneberger,
        'SUNETx4': SUNETx4,
        'SUNETx4_f32_25k_bs16': SUNETx4
    }

    if arch in arch_dict:
        return arch_dict[arch]
    raise NotImplementedError("Architecture name not linked to object, add entry to dictionary.")


def get_dataset_instance(config, dataset=None) :
    dataset_name = config.train.dataset_name if dataset is None else dataset

    # DONT FORGET TO IMPORT THE CLASS
    dataset_dict = {
        'ISLES15_SISS': Isles15_SISS,
        'ISLES15_SPES': Isles15_SPES,
        'ISLES2017': Isles2017,
    }

    assert dataset_name in dataset_dict
    dataset_instance = dataset_dict[dataset_name]
    return dataset_instance()


# Load pre-trained model
def read_model(config, model_def, crossval_id=None) :
    model_filename = generate_model_filename(config, crossval_id)
    log.debug("Reading model weights from {}".format(os.path.basename(model_filename)))
    model_def.load_weights(model_filename)
    return model_def

def save_result_sample(config, sample, result_vol, params=(), file_format='nii.gz', asuint16=False) :
    out_filename = generate_result_filename(config, (sample.id, ) + params, file_format)
    log.debug("Saving volume {}: {}".format(result_vol.shape, out_filename))

    # Remove originally empty voxels by multiplying by brain mask
    result_vol = np.multiply(result_vol, sample.mask)
    if asuint16:
        result_vol = result_vol.astype('uint16')

    img = nib.Nifti1Image(result_vol, sample.nib.affine, sample.nib.header)
    nib.save(img, out_filename)