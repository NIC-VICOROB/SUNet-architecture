import nibabel as nib
import logging as log
import numpy as np
import string
import os
import datetime

import cv2

def generate_output_filename(path, dataset, params, extension) :
    base_path = os.path.join(path, dataset) if dataset is not None else os.path.join(path)
    params_filename = string.join(['{}'.format(param) for param in params], '_')
    extension_name = '.{}'.format(extension) if extension[0] is not '.' else extension
    params_filename = params_filename.replace('.', ',') # Avoid decimal points that could be confounded with extension .
    params_filename = params_filename.replace(' ', '')  # Avoid spaces
    return os.path.join(base_path, params_filename + extension_name)

def generate_model_filename_old(config, crossval_id=None):
    model_params = (
        config.train.architecture_name,
        "{}D".format(config.arch.num_dimensions),
        config.arch.patch_shape,
        config.train.optimizer,
        "{:.0e}".format(config.train.learning_rate),
        "{:.0e}".format(config.train.lr_decay))

    if crossval_id is not None:
        model_params += (crossval_id,)

    return generate_output_filename(config.model_path, config.train.dataset_name, model_params, 'h5')

def generate_model_filename(config, crossval_id=None):
    model_params = (
        config.train.architecture_name,
        "{}D".format(config.arch.num_dimensions),
        config.arch.patch_shape,
        config.train.optimizer)

    if crossval_id is not None:
        model_params += (crossval_id,)

    return generate_output_filename(config.model_path, config.train.dataset_name, model_params, 'h5')

def generate_result_filename(config, additional_params, extension):
    result_params = (
        config.dataset.evaluation_set,
        config.train.architecture_name,
        "{}D".format(config.arch.num_dimensions),
        config.arch.patch_shape,
        config.train.extraction_step_test)
    result_params = additional_params + result_params

    return generate_output_filename(config.results_path, config.train.dataset_name, result_params, extension)

def generate_log_filename(config):
    log_params = (
        datetime.datetime.now().isoformat('_').replace(':', '-').split(".")[0],
        config.train.architecture_name,
        config.arch.num_dimensions,
        config.train.optimizer,
        config.train.batch_size,
        "{:.0e}".format(config.train.learning_rate),
        "{:.0e}".format(config.train.lr_decay),
        config.train.num_patches,
        config.train.uniform_ratio)
    return generate_output_filename(config.log_path, config.train.dataset_name, log_params, 'csv')

def generate_metrics_filename(config):
    metrics_params = ()

    if config.eval.evaluation_type is 'crossval':
        metrics_params += (
            str(config.eval.crossval_fold) + 'fold',)
    elif config.eval.evaluation_type in ['val', 'test']:
        pass
    elif config.eval.evaluation_type is 'grid':
        metrics_params += (
            config.eval.metrics_eval_set,
            config.eval.metrics_thresh_values,
            config.eval.metrics_lesion_sizes)

    metrics_params += (
            datetime.datetime.now().isoformat('_').replace(':', '-').split(".")[0],
            config.train.dataset_name,
            config.train.architecture_name,
            "{}D".format(config.arch.num_dimensions),
            config.train.optimizer,
            config.train.batch_size,
            "{:.0e}".format(config.train.learning_rate),
            "{:.0e}".format(config.train.lr_decay),
            config.train.num_patches,
            config.dataset.min_lesion_voxels,
            config.dataset.lesion_threshold)

    if config.eval.evaluation_type is 'grid':
        datetime_string = datetime.datetime.now().isoformat('_').replace(':', '-').split(".")[0]
        metrics_params = (datetime_string,) + metrics_params

        temp_filename = generate_output_filename(config.metrics_path, None, ('grid_temp',) + metrics_params, 'csv')
        tables_filename = generate_output_filename(config.metrics_path, None, ('grid_tables',) + metrics_params, 'csv')
        return temp_filename, tables_filename
    else:
        return generate_output_filename(config.metrics_path, None, (config.eval.evaluation_type, ) + metrics_params, 'csv')
