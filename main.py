import os
import socket

import logging as log
from configuration import Configuration

config = Configuration()

#print("Using GPU {} on {}".format(config.train.cuda_device_id[socket.gethostname()], socket.gethostname()))
# os.environ["CUDA_VISIBLE_DEVICES"] = str(config.train.cuda_device_id[socket.gethostname()])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.train.cuda_device_id) #str(config.train.cuda_device_id[socket.gethostname()])
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # Silence tensorflow initialization messages

from workflow.evaluate import *

if config.eval.verbose == 0:
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.WARN)
    log.warn("logging level WARN")
elif config.eval.verbose == 2:
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    log.debug("logging level DEBUG")
else:
    log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    log.info("logging level INFO")

if config.eval.verbose in [1, 2]:
    parse_configuration(config, return_csv=True, print_terminal=True, num_columns=2, exclude=None)

if config.train.dynamic_gpu_memory:
    from keras import backend as K
    import tensorflow as tf
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth=True
    K.set_session(tf.Session(config=config_tf))

evaluation_type = config.eval.evaluation_type
if evaluation_type is 'crossval':
    run_crossvalidation(config)
elif evaluation_type is 'eval':
    run_evaluation(config)
elif evaluation_type is 'test':
    run_testing(config)
elif evaluation_type is 'multi':
    run_multi(config)
elif evaluation_type is 'metrics':
    run_metrics(config)
elif evaluation_type is 'metrics_search':
    run_metrics_search(config)
else:
    raise NotImplementedError("\'{}\' evaluation type is not valid".format(evaluation_type))


