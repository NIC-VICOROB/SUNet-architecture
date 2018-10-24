from __future__ import print_function

import scipy.misc
import copy
import time
import datetime
import numpy as np
import logging as log
import itertools as iter

from workflow.learning import *
from workflow.network import *
from workflow.filenames import *

import multiprocessing


def run_multi(config):
    log.info("\n\n" + "="*75 + "\n  Running multieval on {} \n".format(config.dataset.evaluation_set) + "="*75 + "\n")

    multieval_params = config.eval.multieval_params
    multieval_update_config = config.eval.multieval_update_config #function
    assert callable(multieval_update_config)

    # Set evaluation type as eval for naming consistency
    for eval_num, params in enumerate(multieval_params):
        log.info("\n Running multieval {}/{} - {}\n".format(
            eval_num, len(multieval_params), params))

        config_param = multieval_update_config(copy.deepcopy(config), params)

        if config.eval.verbose in [1, 2]:
            parse_configuration(config_param, return_csv=False, print_terminal=False, num_columns=2, exclude=None)

        evaluation_type = config_param.eval.evaluation_type
        if evaluation_type == 'crossval':
            run_crossvalidation(config_param)
        elif evaluation_type == 'eval':
            run_evaluation(config_param)
        elif evaluation_type == 'metrics':
            run_metrics(config_param)
        elif evaluation_type == 'metrics_search':
            run_metrics_search(config_param)
        else:
            raise NotImplementedError("\'{}\' evaluation type is not valid".format(evaluation_type))


def run_evaluation(config):
    log.info("\n\n" + "="*75 + "\n  Running evaluation on {} set \n".format(config.dataset.evaluation_set) + "="*75 + "\n")

    # Get instances of specified architecture and dataset in config
    arch = get_architecture_instance(config)
    _ = arch.get_model(config) # Test model before loading dataset

    # Load database samples and compile keras model
    dataset = get_dataset_instance(config)
    dataset.load(config)

    # Normalise and split train into train and val set (val will be normalised since train is already normalised)
    dataset, crossval_id = split_train_val(dataset, validation_split=config.dataset.validation_split)

    # Train model
    model_def = arch.generate_compiled_model(config, crossval_id)
    model = train_model_on_dataset(config, model_def, dataset, crossval_id, load_trained=True)
    log.info("Finished training")

    # Test model
    samples_eval = dataset.val if config.dataset.evaluation_set is 'val' else dataset.test
    metrics_sample = evaluate_samples(
        config, model, samples_eval, config.eval.save_probabilities, config.eval.save_segmentations)

    if len(metrics_sample) > 0:
        # Print results (only when there are labels to compare)
        metrics_final = add_metrics_avg_std(metrics_sample)
        case_ids = [s.id for s in samples_eval] + ['avg', 'std']

        print_metrics_list(metrics_final, case_names=case_ids)
        save_metrics_csv(config, metrics_final, case_names=case_ids, filename=generate_metrics_filename(config))


def run_testing(config):
    log.info("\n\n" + "="*75 + "\n  Running evaluation on test set \n" + "="*75 + "\n")

    # Get instances of specified architecture and dataset in config
    arch = get_architecture_instance(config)
    #_ = arch.get_model(config) # Test model before loading dataset

    # Load database samples and compile keras model
    dataset = get_dataset_instance(config)
    dataset.load(config)
    assert dataset.test, "Testing set not loaded"

    # Normalise and split train into train and val set (val will be normalised since train is already normalised)
    dataset, crossval_id = split_train_val(dataset, validation_split=config.dataset.validation_split)

    # Train model
    model_def = arch.generate_compiled_model(config, crossval_id)
    model = train_model_on_dataset(config, model_def, dataset, crossval_id, load_trained=False) # WONT DO ANYTHING -> No trainable weights
    log.info("Finished training")

    # Test model
    test_samples(config, model, dataset.test, save_segmentations=True)


def run_crossvalidation(config):
    fold_factor = config.eval.crossval_fold

    log.info("\n\n" + "="*75 + "\n  Running {}-fold crossvalidation on {} set \n".format(
        fold_factor, config.dataset.evaluation_set) + "="*75 + "\n")

    # Get instances of specified architecture and dataset in config
    arch = get_architecture_instance(config)
    #_ = arch.get_model(config) # Test model is correct before loading dataset # BROKEN if loading models inside arch

    # Load database samples and compile keras model
    dataset = get_dataset_instance(config)
    dataset.load(config)

    # Prepare variables
    dataset_original = copy.deepcopy(dataset)
    csv_filename = generate_metrics_filename(config)
    metrics_all, case_ids_all = list(), list()

    # BEGIN CROSS VALIDATION
    for i in range(config.eval.crossval_start, len(dataset_original.train), fold_factor):
        if config.eval.crossval_stop is not None:
            if i >= config.eval.crossval_stop:
                log.info("Reached crossval stop index, finishing...")
                break

        log.info("\n\n  Running crossval iteration {} to {}\n\n".format(i, i + fold_factor))

        dataset, crossval_id = split_train_val(copy.deepcopy(dataset_original), idxs=range(i, i + fold_factor))

        # Train model
        model_def = arch.generate_compiled_model(config, crossval_id)
        model = train_model_on_dataset(config, model_def, dataset, crossval_id)
        log.info("Finished training")

        samples_eval = dataset.val if config.dataset.evaluation_set is 'val' else dataset.test
        metrics_fold = evaluate_samples(
            config, model, samples_eval, save_probabilities=config.eval.save_probabilities)

        #Save metrics_fold
        case_ids_all += [s.id for s in samples_eval]
        metrics_all += metrics_fold

        # Print stuff
        metrics_final = add_metrics_avg_std(metrics_all)
        case_ids_final = case_ids_all + ['avg', 'std']

        print_metrics_list(metrics_final, case_names=case_ids_final)
        save_metrics_csv(config, metrics_final, case_names=case_ids_final, filename=csv_filename)


def run_metrics(config):
    # Get instances of specified architecture and dataset in config
    arch = get_architecture_instance(config)

    # Load database samples and compile keras model
    dataset = get_dataset_instance(config)
    dataset.load(config)
    dataset, crossval_id = split_train_val(dataset, validation_split=config.dataset.validation_split)

    # Pick evaluation samples
    evaluation_set = \
        config.dataset.evaluation_set if config.eval.metrics_eval_set in [None, ''] else config.eval.metrics_eval_set
    if evaluation_set is 'all':
        samples_eval = dataset.train + dataset.val
    else:
        samples_eval = dataset.val if config.dataset.evaluation_set is 'val' else dataset.test

    # Generate result filename and try to load_samples results
    metrics_list = list()
    for i, sample in enumerate(samples_eval):
        if '031950' in str(sample.id):
            continue

        result_filename = generate_result_filename(config, (sample.id, 'probs'), 'nii.gz')
        log.info("Loading {}".format(result_filename))
        lesion_probs = nib.load(result_filename).get_data()

        true_vol = sample.labels[0]
        rec_vol = class_from_probabilities(lesion_probs, config.dataset.lesion_threshold, config.dataset.min_lesion_voxels)

        metrics_list.append(compute_segmentation_metrics(true_vol, rec_vol, lesion=config.dataset.lesion_metrics))

    metrics_final = add_metrics_avg_std(metrics_list)
    case_ids_final = [s.id for s in samples_eval] + ['avg', 'std']

    #print_metrics_list(metrics_final, case_names=case_ids_final)
    print("Saving metrics...")
    save_metrics_csv(config, metrics_final, case_names=case_ids_final, filename=generate_metrics_filename(config))

def get_metrics(params):
    metrics_list_shared, job_num, thresh, lesion_size, prob_vols, true_vols, lesion_metrics = params #tuple unpacking

    metrics_iter = list()
    for lesion_probs, true_vol in iter.izip(prob_vols, true_vols):
        rec_vol = class_from_probabilities(lesion_probs, thresh, lesion_size)
        metrics_iter.append(compute_segmentation_metrics(true_vol, rec_vol, lesion=lesion_metrics))

    m_avg, m_std = get_metrics_avg_std(metrics_iter)
    for k, v in m_std.items():
        m_avg['{}_std'.format(k)] = v

    metrics_list_shared[job_num] = m_avg
    log.debug("Evaluated tresh={}, lesion_size={}".format(thresh, lesion_size))

def run_metrics_search(config):
    thresh_values = config.eval.metrics_thresh_values
    lesion_sizes = config.eval.metrics_lesion_sizes

    # Get instances of specified architecture and dataset in config
    #arch = get_architecture_instance(config)

    # Load database samples and compile keras model
    dataset = get_dataset_instance(config)
    dataset.load(config)
    dataset, crossval_id = split_train_val(dataset, validation_split=config.dataset.validation_split)

    evaluation_set = \
        config.dataset.evaluation_set if config.eval.metrics_eval_set in [None, ''] else config.eval.metrics_eval_set
    if evaluation_set is 'all':
        samples_eval = dataset.train + dataset.val
        print("Getting ALL samples")
    else:
        print("Getting specific samples", config.dataset.evaluation_set)
        samples_eval = dataset.val if config.dataset.evaluation_set is 'val' else dataset.test


    true_vols, prob_vols = list(), list()
    for i, sample in enumerate(samples_eval):
        result_filename = generate_result_filename(config, (sample.id, 'probs'), 'nii.gz')
        log.info("Loading {}".format(result_filename))
        prob_vols.append(nib.load(result_filename).get_data())
        true_vols.append(sample.labels[0])

    # Generate result filename and try to load_samples results
    metrics_list = list()
    metrics_names = list()

    for thresh, lesion_size in iter.product(thresh_values, lesion_sizes):
        log.debug("Evaluating tresh={}, lesion_size={}".format(thresh, lesion_size))

        metrics_iter = list()
        for lesion_probs, true_vol in iter.izip(prob_vols, true_vols):
            rec_vol = class_from_probabilities(lesion_probs, thresh, lesion_size)
            metrics_iter.append(compute_segmentation_metrics(true_vol, rec_vol, lesion=config.dataset.lesion_metrics))

        m_avg, m_std = get_metrics_avg_std(metrics_iter)
        for k, v in m_std.items():
            m_avg['{}_std'.format(k)] = v

        metrics_list.append(m_avg)
        metrics_names.append("th={}_ls={}".format(thresh, lesion_size))

        #metrics_list.append(get_metrics_avg_std(metrics_iter)[0])
        #metrics_names.append("th={}_ls={}".format(thresh, lesion_size))

    config.eval.evaluation_type = 'grid'
    result_filename, tables_filename = generate_metrics_filename(config)
    print_metrics_list(metrics_list, case_names=metrics_names)
    save_metrics_csv(config, metrics_list, case_names=metrics_names, filename=result_filename)
