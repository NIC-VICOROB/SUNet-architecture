import copy
import abc # abstract base classes
import logging as log

class Sample:
    def __init__(self, id=None, nib=None, mask=None, data=None, labels=None):
        self.id = id
        self.nib = nib
        self.mask = mask # Brain mask (NOT lesion mask)
        self.data = data
        self.labels = labels

class Dataset:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.train = []
        self.val = []
        self.test = []

    def load(self, config):
        self.load_samples(config.dataset)
        self.as_float32() # Convert loaded data to float32

        log.debug("Loaded {} training, {} validation, {} testing".format(
            len(self.train), len(self.val), len(self.test)))

    @abc.abstractmethod
    def load_samples(self, dataset_info):
        pass

    def __is_sample(self, sample_in):
        assert isinstance(sample_in, Sample), "Elements added to dataset must be instances of Sample"

    def add_train(self, sample_in):
        self.__is_sample(sample_in)
        self.train.append(sample_in)

    def add_val(self, sample_in):
        self.__is_sample(sample_in)
        self.val.append(sample_in)

    def add_test(self, sample_in):
        self.__is_sample(sample_in)
        self.test.append(sample_in)

    def as_float32(self):
        for idx in range(len(self.train)):
            self.train[idx].data = self.train[idx].data.astype('float32')

        for idx in range(len(self.val)):
            self.val[idx].data = self.val[idx].data.astype('float32')

        for idx in range(len(self.test)):
            self.test[idx].data = self.test[idx].data.astype('float32')



