#!/usr/bin/env python
import os
from utils.loss_functions import *
from math import floor


class TrainConfiguration:
    def __init__(self):
        # GPU management
        self.cuda_device_id = 0
        self.dynamic_gpu_memory = False

        # Basic
        self.num_classes = 2
        self.dataset_name = 'ISLES2017'  # Overrided in multi evaluation
        self.architecture_name = 'SUNETx4'  # Overrided in multi evaluation

        # Patch sampling config
        self.extraction_step = (6, 6, 3)
        self.extraction_step_test = (6, 6, 1)

        self.sampling = 'hybrid'  # hybrid, Kamnitsas, Guerrero
        self.num_patches = 10000  # Max number of patches extracted PER CASE (pos + unif)
        self.uniform_ratio = 0.5  # Percentage of num_patches that will be uniformly sampled
        self.min_fg_percentage = 0.0

        # Filled from Configuration.on_change()
        self.min_pos_patches, self.max_pos_patches, self.max_unif_patches = None, None, None

        # Learning parameters
        self.activation = 'softmax'
        self.loss = 'categorical_crossentropy'  # 'categorical_crossentropy'
        self.metrics = ['acc', dice]
        self.train_monitor, self.train_monitor_mode = 'val_dice', 'min'

        self.optimizer = 'Adadelta'  # Check Arquitecture.py to add more optimizers
        self.learning_rate, self.lr_decay = 1.0, 1e-3  # Only used for 'Adam' optimizer # 0.03

        self.batch_size = 32  # default: 32
        self.patience = 6
        self.num_epochs = 0


class EvaluationConfiguration:
    def __init__(self):
        self.verbose = 1  # 0 warn, 1 info, 2 debug
        self.keras_verbose = 1
        self.evaluation_type = 'multi'  # multi, crossval, eval, metrics, metrics_search, debug, test

        # Val or test
        self.save_probabilities = False
        self.save_segmentations = True

        # Crossval
        self.crossval_fold = 9
        self.crossval_start = 0  # INDEX OF SAMPLE to start with, useful for resuming crossvalidations
        self.crossval_stop = None

        # Metrics
        self.metrics_eval_set = 'all'  # all, val, test,
        self.metrics_thresh_values = [.1, .2, .3, .4, .5, .6, .7, .8]
        self.metrics_lesion_sizes = [1, 10, 20, 50, 100, 200, 300, 500, 750, 1000]

        # Multievaluation
        self.multieval_params = [
            ('crossval', 'SUNETx4', 'ISLES15_SISS', 10000, 6, (24, 24, 8), (0.6, 200)),
            ('crossval', 'SUNETx4', 'ISLES15_SPES', 10000, 6, (24, 24, 8), (0.4, 500)),
            ('crossval', 'SUNETx4', 'ISLES2017', 7500, 9, (24, 24, 8), (0.2, 100)),

            ('crossval', '2Dunet', 'ISLES15_SISS', 10000, 6, (48, 48, 1), (0.8, 200)),
            ('crossval', '2Dunet', 'ISLES15_SPES', 10000, 6, (48, 48, 1), (0.1, 750)),
            ('crossval', '2Dunet', 'ISLES2017', 7500, 9, (48, 48, 1), (0.1, 100)),

            ('crossval', '3Dunet', 'ISLES15_SISS', 10000, 6, (24, 24, 8), (0.6, 50)),
            ('crossval', '3Dunet', 'ISLES15_SPES', 10000, 6, (24, 24, 8), (0.2, 1000)),
            ('crossval', '3Dunet', 'ISLES2017', 7500, 9, (24, 24, 8), (0.1, 200)),

            ('crossval', '3Duresnet', 'ISLES15_SISS', 10000, 6, (24, 24, 8), (0.7, 50)),
            ('crossval', '3Duresnet', 'ISLES15_SPES', 10000, 6, (24, 24, 8), (0.3, 1000)),
            ('crossval', '3Duresnet', 'ISLES2017', 7500, 9, (24, 24, 8), (0.2, 100)),
        ]

        def multieval_update_config(config, params):
            assert isinstance(config, Configuration)
            config.eval.evaluation_type = params[0]
            config.change_architecture(params[1])
            config.change_dataset(params[2])
            config.change_num_patches(params[3])
            config.eval.crossval_fold = params[4]
            config.arch.patch_shape = params[5] if params[5] is not None else config.arch.patch_shape
            config.arch.output_shape = params[5] if params[5] is not None else config.arch.output_shape
            config.dataset.lesion_threshold = params[6][0]
            config.dataset.min_lesion_voxels = params[6][1]
            return config
        self.multieval_update_config = multieval_update_config


"""
Mandatory fields for architecture entry:
-> 'multires' : Bool # If architecture is multiple patch input
-> 'patch_shape' : 3-element tuple extraction shape (normally the biggest of em all)
-> 'num_dimensions'
"""
architecture_dict = {
    '2Dunet': {
        'num_dimensions': 2,
        'patch_shape': (48, 48, 1),
        'output_shape': (48, 48, 1)
    },
    '3Dunet': {
        'num_dimensions' : 3,
        'patch_shape': (24, 24, 8),
        'output_shape': (24, 24, 8)
    },
    '3Duresnet': {
        'num_dimensions' : 3,
        'patch_shape': (24, 24, 8),
        'output_shape': (24, 24, 8)
    },
    'SUNETx4': {
        'num_dimensions' : 3,
        'patch_shape': (24, 24, 8),
        'output_shape': (24, 24, 8),
        'dropout_rate': 0.2,
        'base_filters': 32
    },

}

"""
Mandatory fields!
'lesion_metrics'
'modalities': ['', ...]
'evaluation_set'
"""

dataset_dict = {  # Each dataset uses its own configuration keys, no need to preserve consistency
    'ISLES15_SISS': {
        'path': '/path/to/SISS/dataset/',
        'lesion_metrics': False,
        'evaluation_set': 'val',
        'validation_split': 0.2,
        'lesion_threshold': 0.6,
        'min_lesion_voxels': 200,
        'general_pattern': ['training/{}/', 'testing/{}/'],
        'num_volumes': [28, 36],
        'modalities': ['Flair', 'T1', 'T2', 'DWI'],
    },
    'ISLES15_SPES': {
        'path': '/path/to/SPES/dataset/',
        'lesion_metrics': False,
        'evaluation_set': 'val',
        'validation_split': 0.2,
        'lesion_threshold': 0.4,
        'min_lesion_voxels': 500,
        'general_pattern': ['training/{}/', 'testing/Nr{}/'],
        'num_volumes': [30, 20],
        'modalities': ['DWI', 'CBF', 'CBV', 'T1c', 'T2', 'Tmax', 'TTP'],
    },
    'ISLES2017': {
        'path': '/path/to/ISLES2017/dataset/',
        'lesion_metrics': False,
        'validation_split': 0.2,
        'lesion_threshold': 0.2,
        'min_lesion_voxels': 100,
        'general_pattern': ['training/training_{}/', 'testing/test_{}/'],
        'num_volumes': [48, 40], #[43, 32],
        'modalities': ['ADC', 'CBF', 'CBV', 'MTT', 'Tmax', 'TTP'],
        'evaluation_set': 'val',
    },
}

"""
Assertions and main config class
"""
def check_dictionaries(dataset, architecture):
    for dataset_name, dataset_config in dataset.items():
        assert 'lesion_metrics' in dataset_config, 'Missing field \'lesion_metrics\' for database entry'
        assert 'modalities' in dataset_config, 'Missing field \'lesion_metrics\' for database entry'
        assert 'evaluation_set' in dataset_config, 'Missing field \'evaluation_set\' for database entry'

    for arch_name, arch_config in architecture.items():
        assert all([len(patch_shape) == 3 for key, patch_shape in arch_config.items() if 'shape' in key]), arch_name

check_dictionaries(dataset_dict, architecture_dict)

import copy
class DictionaryDot(dict):
    """
    Wrapper for dictionary that allows for dot notation access to entries

    Example:
    m = DictDot({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    print(m.first_name)
    """

    def __init__(self, *args, **kwargs):
        super(DictionaryDot, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DictionaryDot, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DictionaryDot, self).__delitem__(key)
        del self.__dict__[key]

    def __deepcopy__(self, memo):
        return DictionaryDot([(copy.deepcopy(k, memo), copy.deepcopy(v, memo)) for k, v in self.items()])

class Configuration:
    def __init__(self):
        self.log_path = 'log/'
        self.model_path = 'checkpoints/'
        self.results_path = 'results/'
        self.metrics_path = 'metrics/'

        self.eval = EvaluationConfiguration()
        self.train = TrainConfiguration()
        self.dataset = DictionaryDot(dataset_dict[self.train.dataset_name])
        self.arch = DictionaryDot(architecture_dict[self.train.architecture_name])

        self.on_change()

    def on_change(self):
        self.train.min_pos_patches = int(((1.0 - self.train.uniform_ratio) * self.train.num_patches) // 6)
        self.train.max_pos_patches = int(floor((1.0 - self.train.uniform_ratio) * self.train.num_patches))
        self.train.max_unif_patches = int(floor(self.train.uniform_ratio * self.train.num_patches))

        # Other operations
        self.dataset.path = os.path.expanduser(self.dataset.path)
        if self.dataset.path[-1] is not '/':
            self.dataset.path += '/'

        if self.arch.num_dimensions is 2:
            assert all([shape[-1] == 1 for key, shape in self.arch.items() if 'shape' in key])

    def change_num_patches(self, new_num_patches):
        self.train.num_patches = new_num_patches
        self.on_change()

    def change_architecture(self, new_arch_name):
        self.train.architecture_name = new_arch_name
        self.arch = DictionaryDot(architecture_dict[self.train.architecture_name])
        self.on_change()

    def change_dataset(self, new_dataset_name):
        self.train.dataset_name = new_dataset_name
        self.dataset = DictionaryDot(dataset_dict[self.train.dataset_name])
        self.on_change()
