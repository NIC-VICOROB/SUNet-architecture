import os
import numpy as np
import logging as log

from .Dataset import Dataset, Sample
from utils.volume import get_brain_mask, read_volume, load_nib_file


class Isles15_SISS(Dataset):
    def load_samples(self, config_dataset):
        log.info("Loading Isles15 SISS dataset...")

        num_volumes = config_dataset.num_volumes
        dataset_path = config_dataset.path
        pattern = config_dataset.general_pattern
        modalities = config_dataset.modalities

        # Training loading
        for case_num in range(1, num_volumes[0] + 1):
            filepaths = [None] * (len(modalities) + 1)

            for root, subdirs, files in os.walk(os.path.join(dataset_path, pattern[0].format(case_num))):
                for idx_mod, modality in enumerate(modalities):

                    modality_file = [file_idx for file_idx, filename in enumerate(files) if modality in filename]
                    assert len(modality_file) in [0,1], "Found more than one file for the same modality"
                    if modality_file:
                        filepaths[idx_mod] = os.path.join(root, files[modality_file[0]])

                    ground_truth_file = [file_idx for file_idx, filename in enumerate(files) if 'OT' in filename]
                    if ground_truth_file:
                        filepaths[-1] = os.path.join(root, files[ground_truth_file[0]])


            # Check folder exists (some samples missing)
            if any([filepath is None for filepath in filepaths]):
                raise ValueError, "Didn't find all expected modalities"

            sample = Sample(id=case_num)

            # Load volume to check dimensions (not the same for all train samples)
            sample.nib = load_nib_file(filepaths[0])
            vol = sample.nib.get_data()

            sample.data = np.zeros((len(modalities),) + vol.shape)
            sample.labels = np.zeros((1,) + vol.shape)

            # Load all modalities (except last which is gt segmentation) into last appended ndarray
            sample.data[0] = vol
            for i, filepath in enumerate(filepaths[:-1]):
                sample.data[i] = read_volume(filepath)

            sample.mask = get_brain_mask(sample.data)

            # Ground truth loading
            sample.labels[0] = read_volume(filepaths[-1])

            self.add_train(sample)

        # Testing loading
        for case_num in range(1, num_volumes[1] + 1):
            filepaths = [None] * (len(modalities))

            for root, subdirs, files in os.walk(os.path.join(dataset_path, pattern[1].format(case_num))):
                for idx_mod, modality in enumerate(modalities):

                    modality_file = [file_idx for file_idx, filename in enumerate(files) if modality in filename]
                    assert len(modality_file) in [0,1], "Found more than one file for the same modality"
                    if modality_file:
                        filepaths[idx_mod] = os.path.join(root, files[modality_file[0]])

            # Check folder exists (some samples missing)
            if any([filepath is None for filepath in filepaths]):
                raise ValueError, "Didn't find all expected modalities"

            sample = Sample(id=case_num)

            # Load volume to check dimensions (not the same for all train samples)
            sample.nib = load_nib_file(filepaths[0])
            vol = sample.nib.get_data()

            sample.data = np.zeros((len(modalities),) + vol.shape)

            # Load all modalities (except last which is gt segmentation) into last appended ndarray
            sample.data[0] = vol
            for i, filepath in enumerate(filepaths):
                sample.data[i] = read_volume(filepath)

            sample.mask = get_brain_mask(sample.data)

            self.add_test(sample)


