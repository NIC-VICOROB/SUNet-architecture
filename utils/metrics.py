from keras import backend as K
import numpy as np
import logging as log
import time
import math

import itertools as it

from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from medpy.metric.binary import hd as haussdorf_dist

import copy
import nibabel as nib


def class_from_probabilities(y_prob_in, thresh, min_lesion_vox):
    """
    Generates final class prediction by thresholding according to threshold and filtering by minimum lesion size
    """

    # Apply threshold
    y_prob = y_prob_in > thresh

    # Get connected components information
    y_prob_labelled, nlesions = ndimage.label(y_prob)
    if nlesions > 0:
        label_list = np.arange(1, nlesions + 1)
        lesion_volumes = ndimage.labeled_comprehension(y_prob, y_prob_labelled, label_list, np.sum, float, 0)

        # Set to 0 invalid lesions
        lesions_to_ignore = [idx + 1 for idx, lesion_vol in enumerate(lesion_volumes) if lesion_vol < min_lesion_vox]
        y_prob_labelled[np.isin(y_prob_labelled, lesions_to_ignore)] = 0

    # Generate binary mask and return
    y_pred = (y_prob_labelled > 0).astype('uint8')

    return y_pred


def compute_confusion_matrix(y_true, y_pred):
    """
    Returns tuple tp, tn, fp, fn
    """

    assert y_true.size == y_pred.size

    true_pos = np.sum(np.logical_and(y_true, y_pred))
    true_neg = np.sum(np.logical_and(y_true == 0, y_pred == 0))

    false_pos = np.sum(np.logical_and(y_true == 0, y_pred))
    false_neg = np.sum(np.logical_and(y_true, y_pred == 0))

    return true_pos, true_neg, false_pos, false_neg

def compute_segmentation_metrics(y_true, y_pred, lesion=False, exclude=None):
    metrics = {}
    eps = K.epsilon()

    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)

    #Sensitivity and specificity
    metrics['sens'] = tp / (tp + fn + eps)
    metrics['spec'] = tn / (tn + fp + eps)

    # Voxel Fractions
    #metrics['tpf'] = metrics['sens']
    metrics['fpf'] = 1 - metrics['spec']

    # Lesion metrics
    if lesion:
        tpl, fpl, num_lesions_true, num_lesions_pred = compute_lesion_confusion_matrix(y_true, y_pred)
        metrics['l_tpf'] = tpl / num_lesions_true if num_lesions_true > 0 else np.nan
        metrics['l_fpf'] = fpl / num_lesions_pred if num_lesions_pred > 0 else np.nan

        metrics['l_ppv'] = tpl / (tpl + fpl + eps)
        metrics['l_f1'] = (2.0 * metrics['l_ppv'] * metrics['l_tpf']) / (metrics['l_ppv'] + metrics['l_tpf'] + eps)

    #Dice coefficient
    metrics['dsc'] = dice_coef(y_true, y_pred)
    # RELATIVE volume difference
    metrics['avd'] = 2.0 * np.abs(np.sum(y_pred) - np.sum(y_true))/(np.sum(y_pred) + np.sum(y_true) + eps)

    # Haussdorf distance
    try:
        metrics['hd'] = haussdorf_dist(y_pred, y_true, connectivity=3)  # Why connectivity 3?
    except Exception:
        metrics['hd'] = np.nan

    if exclude is not None:
        [metrics.pop(metric, None) for metric in exclude]

    return metrics

def add_metrics_avg_std(metrics_in):
    assert type(metrics_in) is list
    metrics_list = copy.deepcopy(metrics_in)

    metrics_all = copy.deepcopy(metrics_list[0])
    for k, v in sorted(metrics_all.items()):
        metrics_all[k] = list()

    for metrics in metrics_list:
        for k, v in sorted(metrics.items()):
            metrics_all[k].append(v)

    metrics_avg = copy.deepcopy(metrics_list[0])
    metrics_std = copy.deepcopy(metrics_list[0])
    for k, v in sorted(metrics_all.items()):
        metrics_avg[k] = np.nanmean(metrics_all[k])
        metrics_std[k] = np.nanstd(metrics_all[k])

    # Avoid nan's corrupting average
    metrics_list.append(metrics_avg)
    metrics_list.append(metrics_std)

    return metrics_list

# TODO avoid duplicated code!
def get_metrics_avg_std(metrics_in):
    assert type(metrics_in) is list
    metrics_list = copy.deepcopy(metrics_in)

    metrics_all = copy.deepcopy(metrics_list[0])
    for k, v in sorted(metrics_all.items()):
        metrics_all[k] = list()

    for metrics in metrics_list:
        for k, v in sorted(metrics.items()):
            metrics_all[k].append(v)

    metrics_avg = copy.deepcopy(metrics_list[0])
    metrics_std = copy.deepcopy(metrics_list[0])
    for k, v in sorted(metrics_all.items()):
        metrics_avg[k] = np.nanmean(metrics_all[k])
        metrics_std[k] = np.nanstd(metrics_all[k])

    return [metrics_avg, metrics_std]


def compute_lesion_confusion_matrix(y_true, y_pred):
    # True positives
    lesions_true, num_lesions_true = ndimage.label(y_true)
    lesions_pred, num_lesions_pred = ndimage.label(y_pred)

    true_pos = 0.0
    for i in range(num_lesions_true):
        lesion_detected = np.logical_and(y_pred, lesions_true == (i + 1)).any()
        if lesion_detected: true_pos += 1
    true_pos = np.min([true_pos, num_lesions_pred])

    # False positives
    tp_labels = np.unique(y_true * lesions_pred)
    fp_labels = np.unique(np.logical_not(y_true) * lesions_pred)

    # [label for label in fp_labels if label not in tp_labels]
    false_pos = 0.0
    for fp_label in fp_labels:
        if fp_label not in tp_labels: false_pos += 1

    return true_pos, false_pos, num_lesions_true, num_lesions_pred


def dice_coef(y_true, y_pred, smooth = 0.01):
    intersection = np.sum(np.logical_and(y_true, y_pred))

    if intersection > 0:
        return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    else:
        return 0.0





















