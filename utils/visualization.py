
from __future__ import print_function
import sys
import time
import logging as log
import csv
import itertools as iter
import string
import numpy as np
import cv2

import copy
import nibabel as nib

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 25, fill = '='):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """

    total = total

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)

    bar = fill * filledLength + '>' * min(length - filledLength, 1) + '.' * (length - filledLength - 1)

    print('\r{} [{}] {}% {}'.format(prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration == total: print()

def print_metrics_list(metrics_list, case_names=None):
    if type(metrics_list) is not list:
        metrics_list = [metrics_list]

    if case_names is None:
        case_names = range(len(metrics_list))

    #log.debug("{:^40}".format("Metrics"))
    print("{:<16}".format('Sample'), end='')
    for metric_name, metric_value in sorted(metrics_list[0].items()):
        print("  {:>8}".format(metric_name), end='')
    print("")

    for i, metrics in enumerate(metrics_list):
        print("{:<16}".format(case_names[i]), end='')
        for metric_name, metric_value in sorted(metrics.items()):
            print("  {:<08.3}".format(metric_value), end='')
        print("")
    print("")

def save_metrics_csv(config, metrics_list, filename=None, case_names=None, print_config=False):
    if type(metrics_list) is not list:
        metrics_list = [metrics_list]
    if case_names is None:
        case_names = range(len(metrics_list))

    with open(filename, 'wb') as csvfile:
        row = "{},".format("#sample")
        for metric_name, metric_value in sorted(metrics_list[0].items()):
            row += "{},".format(metric_name)
        csvfile.write(row + '\n')

        for i, metrics in enumerate(metrics_list):
            row = "{},".format(case_names[i])
            for metric_name, metric_value in sorted(metrics.items()):
                row += "{},".format(metric_value)

            csvfile.write(row + '\n')

        csvfile.write(parse_configuration(config, return_csv=True, num_columns=1))

def append_grid_search_result(metrics_avg, param_names, param_values, filename):
    with open(filename, 'ab') as csvfile:
        row = " , ," # Add two empty entries to compensate for param names
        for metric_name, metric_value in sorted(metrics_avg.items()):
            row += "{},".format(metric_name)
        csvfile.write(row + '\n')

        for param_name, param_value in iter.izip(param_names, param_values):
            row += "{}={},".format(param_name, param_value)
        csvfile.write(row)

        row = ""
        for metric_name, metric_value in sorted(metrics_avg.items()):
            row += "{},".format(metric_value)
        csvfile.write(row + '\n')

def save_grid_search_tables(config, metrics_avg_list, param_names, param_values, filename):
    assert len(metrics_avg_list) == np.prod([len(p) for p in param_values])

    empty_table = [[None] * (len(param_values[0]) + 1) for i in range(len(param_values[1]) + 1)]

    metric_names = metrics_avg_list[0].keys()
    for metric_name in sorted(metric_names):
        # Make a copy of empty table
        metric_table = copy.deepcopy(empty_table)

        # Metric name
        metric_table[0][0] = "{},".format(metric_name)
        # Labels 1st column - param0
        for idx in range(len(param_values[0])):
            metric_table[idx+1][0] = "{},".format(param_names[0] + "=" + str(param_values[0][idx]))
        # Labels 1st row - param1
        for idx in range(len(param_values[1])):
            metric_table[0][idx+1] = "{},".format(param_names[1] + "=" + str(param_values[1][idx]))
        # Average metric values
        for count, idxs in enumerate(iter.product(range(len(param_values[0])), range(len(param_values[1])))):
            metric_table[idxs[0]+1][idxs[1]+1] = "{},".format(metrics_avg_list[count][metric_name])

        with open(filename, 'ab') as csvfile:
            for row in metric_table:
                csvfile.write(string.join(row) + "\n")
            csvfile.write("\n")
            csvfile.write(parse_configuration(config, return_csv=True, num_columns=1))

def parse_configuration(config, return_csv=False, print_terminal=False, num_columns=1, exclude=list()):
    width_name, width_value = 20, 13
    total_width = num_columns*(width_name + width_value + 6)

    template_supertitle = '{:^' + str(total_width-6) + '}'
    template_title = '{}'
    template_name = '    {:.<' + str(width_name) + '}'
    template_value = ' {:<' + str(width_value) + '}'

    if exclude is not None:
        exclude += ['keras_verbose', 'num_classes', 'extraction_step', 'bg_discard_percentage', 'verbose', 'general_pattern',
            'min_pos_patches', 'max_pos_patches', 'max_unif_patches', 'loss', 'train_monitor', 'metrics', 'format', 'path']
    else:
        exclude = []

    settings = list()

    settings += ['', 'Evaluation configuration']
    settings.append(['evaluation_type', config.eval.evaluation_type])
    settings.append(['verbose', config.eval.verbose])
    settings.append(['save_probabilities', config.eval.save_probabilities])
    settings.append(['save_segmentations', config.eval.save_segmentations])
    if config.eval.evaluation_type is 'crossval':
        settings.append(['crossval_fold', str(config.eval.crossval_fold) + 'fold'])
    elif config.eval.evaluation_type in ['val', 'test']:
        pass
    elif config.eval.evaluation_type is 'grid':
        pass

    settings += ['', 'Dataset']
    settings.append(['name', config.train.dataset_name])
    for name, value in sorted(vars(config.dataset).items()):
        if name not in exclude: settings.append([name, value])

    settings += ['', 'Architecture']
    settings.append(['name', config.train.architecture_name])
    for name, value in sorted(vars(config.arch).items()):
        if name not in exclude: settings.append([name, value])

    settings += ['','Training']
    for name, value in sorted(vars(config.train).items()):
        if name not in exclude + ['dataset_name', 'architecture_name']: settings.append([name, value])

    if print_terminal:
        log.info(
            "\n\n" + "-" * total_width + "\n" + template_supertitle.format('Configuration') + "\n" + "-" * total_width)

        row = ''
        num_settings_section = 0
        for i, setting in enumerate(settings):
            if isinstance(setting, list):
                row += template_name.format(setting[0]) + template_value.format(setting[1])
                num_settings_section += 1

                if num_settings_section > 0 and num_settings_section % num_columns == 0: # Normal column break
                    print(row)
                    num_settings_section, row = 0, ''
            else:
                if num_settings_section > 0:
                    print(row)
                num_settings_section, row = 0, ''
                print(template_title.format(setting))
        print('')

    csv_string = ''
    if return_csv:
        row = ''
        num_settings_section = 0
        for i, setting in enumerate(settings):
            if isinstance(setting, list):
                row += str(setting[0]) + ',' + str(setting[1]) + ','
                num_settings_section += 1

                if num_settings_section > 0 and num_settings_section % num_columns == 0:  # Normal column break
                    csv_string += row + '\n'
                    num_settings_section, row = 0, ''
            else:
                if num_settings_section > 0:
                    csv_string += row + '\n'
                num_settings_section, row = 0, ''
                csv_string += str(setting) + '\n'

    return csv_string

def store_batch(filename, x, y_in):
    if y_in.ndim != x[0].ndim:
        log.debug("Detected categorical y {}".format(y_in.shape))
        y = np.reshape(y_in[:, :, 1], (len(y_in), 1, 24, 24, 24))
    else:
        y = y_in

    cell_shape = (6*30, 6*30, 24)

    x_save = np.zeros(cell_shape)
    y_save = np.zeros(cell_shape)

    patch_shape = (24, 24, 24)
    modality_idx = 0

    # 6*6
    for i, j in iter.product(range(0, 6), range(6)):
        idx = np.ravel_multi_index((i,j), (6, 6))

        selection = [slice(i*30, i*30 + patch_shape[0]), slice(j*30, j*30 + patch_shape[1]), slice(None)]
        x_save[selection] = x[0][idx, modality_idx]

        selection = [slice(i* 30, i * 30 + patch_shape[0]), slice(j * 30, j * 30 + patch_shape[1]), slice(None)]
        y_save[selection] = y[idx, 0]

    x_save = x_save - np.min(x_save)

    log.debug("Storing batch {}".format(filename.format('x')))
    img_x = nib.Nifti1Image(x_save, np.eye(4))
    nib.save(img_x, filename.format('x'))

    log.debug("Storing batch {}".format(filename.format('y')))
    img_y = nib.Nifti1Image(y_save, np.eye(4))
    nib.save(img_y, filename.format('y'))

def store_sample_patches(filename, x, y_in):
    if y_in.ndim != x.ndim:
        log.debug("Detected categorical y {}".format(y_in.shape))
        y = np.reshape(y_in[:, 1], (1, 24, 24, 8))
    else:
        y = y_in

    num_patches = x.shape[0] + 1
    cell_shape = (num_patches*30, 24, 8)
    patch_shape = (24, 24, 8)

    x_save = np.zeros(cell_shape)

    for i in range(x.shape[0] + 1):
        selection = [slice(i*30, i*30 + patch_shape[0]), slice(None), slice(None)]

        if i < x.shape[0]:
            x_save[selection] = normalize_image(x[i])
        else:
            x_save[selection] = normalize_image(y[0])

    log.debug("Storing batch {}".format(filename.format('x')))
    img_x = nib.Nifti1Image(x_save, np.eye(4))
    nib.save(img_x, filename.format('x'))

def normalize_image(img):
    image = np.round(255.0*((img - np.min(img)) / (np.max(img) - np.min(img))), decimals=0)
    return image.astype('uint8')

def normalize_and_save_patch(image_patch, label_patch, filename):
    save_size = (128, 128)

    img = normalize_image(image_patch)
    img_big = cv2.resize(img, save_size, interpolation=cv2.INTER_NEAREST)

    if label_patch is not None:
        lbl = normalize_image(label_patch)
        lbl_big = cv2.resize(lbl, save_size, interpolation=cv2.INTER_NEAREST)
        patch = np.concatenate((img_big, lbl_big), axis=1)
    else:
        patch = img_big

    log.debug("Saving {} with max:{}, min{}".format(filename, np.max(patch), np.min(patch)))
    cv2.imwrite(filename, patch)

def normalize_and_save_multires_patch(global_patch, local_patch, label_patch, filename):

    local_patch = np.pad(local_patch, (16,), 'constant', constant_values=0)
    label_patch = np.pad(label_patch, (24,), 'constant', constant_values=0)

    save_size = (128, 128)
    img_g_big = cv2.resize(global_patch, save_size, interpolation=cv2.INTER_NEAREST)
    img_l_big = cv2.resize(local_patch, save_size, interpolation=cv2.INTER_NEAREST)
    lbl_big = cv2.resize(label_patch, save_size, interpolation=cv2.INTER_NEAREST)

    patch = np.concatenate((img_g_big, img_l_big), axis=1)
    patch = normalize_image(patch)
    patch = np.concatenate((patch, normalize_image(lbl_big)), axis=1).astype('uint8')


    log.debug("Saving {}".format(filename))
    cv2.imwrite(filename, patch)

def normalize_and_save_patches(patches, filename):
    save_size = (128, 128)

    patch_out = None
    for patch in patches:
        p = normalize_image(patch)
        p_big = cv2.resize(np.copy(p), save_size, interpolation=cv2.INTER_NEAREST)
        patch_out = p_big if patch_out is None else np.concatenate((patch_out, p_big), axis=1)

    log.debug("Saving {}".format(filename))
    cv2.imwrite(filename, patch_out)
