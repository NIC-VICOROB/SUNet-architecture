from __future__ import print_function

from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint

from utils.metrics import *
from utils.patch import *
from utils.samples import *
from utils.visualization import *
from utils.volume import *
from workflow.in_out import *
from workflow.learning import build_testing_generator, build_learning_generators


def generate_callbacks(config, crossval_id=None):
    model_filename = generate_model_filename(config, crossval_id=crossval_id)
    csv_filename = generate_log_filename(config)

    stopper = EarlyStopping(
        mode=config.train.train_monitor_mode,
        monitor=config.train.train_monitor,
        min_delta=1e-5,
        patience=config.train.patience)

    checkpointer = ModelCheckpoint(
        filepath=model_filename,
        verbose=0,
        monitor=config.train.train_monitor,
        save_best_only=True,
        save_weights_only=True)

    csv_logger = CSVLogger(
        csv_filename,
        separator=',')

    return [stopper, checkpointer, csv_logger]

def train_model_on_dataset(config, model_def, dataset, crossval_id=None, load_trained=False):
    trainable_count = int(np.sum([K.count_params(p) for p in set(model_def.trainable_weights)]))

    if load_trained:
        try:
            model_def = read_model(config, model_def, crossval_id)
            log.debug("Loaded weights BEFORE training")
        except IOError:
            log.debug("Failed to load weights BEFORE training")

    if trainable_count == 0:
        weight_savefile = generate_model_filename(config, crossval_id)
        log.warn("Detected untrainable network, storing weights...\n{}".format(weight_savefile))
        model_def.save_weights(weight_savefile)
    elif config.train.num_epochs > 0:
        train_gen, val_gen = build_learning_generators(config, dataset)
        train_model(config, model_def, train_gen, val_gen, crossval_id)
        del train_gen, val_gen

    model = read_model(config, model_def, crossval_id)
    return model


def train_model(config, model_def, train_data, validation_data, crossval_id=None):
    train_generator, validation_generator = train_data, validation_data

    model_def.fit_generator(
        train_generator,
        epochs=config.train.num_epochs,
        validation_data=validation_generator,
        max_queue_size=10,
        shuffle=True,
        verbose=config.eval.keras_verbose,
        callbacks=generate_callbacks(config, crossval_id))


def test_samples(config, model, samples_in, save_segmentations=True):
    log.info("Testing {} samples...".format(len(samples_in)))

    for i, sample_test in enumerate(samples_in):
        lesion_probs = predict_sample(config, model, sample_test)

        rec_vol = class_from_probabilities(lesion_probs, config.dataset.lesion_threshold, config.dataset.min_lesion_voxels)

        if save_segmentations:
            save_result_sample(config, samples_in[i], rec_vol, params=('seg',), file_format='nii', asuint16=True)


def evaluate_samples(config, model, samples_in, save_probabilities=True, save_segmentations=False):
    # TODO add thresh and ls search

    log.info("Evaluating with {} samples...".format(len(samples_in)))

    metrics_list = list()
    for i, sample_test in enumerate(samples_in):
        lesion_probs = predict_sample(config, model, sample_test)

        if save_probabilities:
            save_result_sample(config, samples_in[i], lesion_probs, params=('probs',))


        rec_vol = class_from_probabilities(lesion_probs, config.dataset.lesion_threshold, config.dataset.min_lesion_voxels)

        if save_segmentations:
            save_result_sample(config, samples_in[i], rec_vol, params=('seg',))

        if sample_test.labels is not None:
            true_vol = samples_in[i].labels[0]
            metrics_list += [compute_segmentation_metrics(true_vol, rec_vol, lesion=config.dataset.lesion_metrics)]
            print_metrics_list(metrics_list[-1], case_names=[sample_test.id])


    return metrics_list if len(metrics_list) > 0 else None

def predict_sample(config, model, sample_in):
    """
    Given the volumes and the trained model -> Outputs positive lesion probabilities
    """

    assert isinstance(sample_in, Sample)
    sample_seg = copy.deepcopy(sample_in)

    log.info("Segmenting case {} {}".format(sample_seg.id, sample_seg.data.shape))
    sample_seg = zeropad_sample(sample_seg, config.arch.patch_shape)

    test_generator = build_testing_generator(config, sample_seg)
    pred = predict_model_generator(config, model, test_generator)

    lesion_probs = reconstruct_volume(
        patches=pred,
        patch_shape = config.arch.patch_shape,
        original_vol = sample_seg.data,
        original_centers=test_generator.centers,
        extraction_step=config.train.extraction_step_test,
        num_classes=config.train.num_classes)

    lesion_probs = remove_zeropad_volume(lesion_probs, config.arch.patch_shape)

    assert lesion_probs.shape == sample_in.data.shape[1:], str(lesion_probs.shape) + ", " + str(sample_in.data.shape[1:])
    return lesion_probs

def predict_model_generator(config, model, test_generator):
    """
    Performs prediction on a set of patches and returns the reshaped prediction
    """

    num_classes = config.train.num_classes
    output_shape = config.arch.output_shape[:config.arch.num_dimensions]

    pred = model.predict_generator(test_generator, verbose=config.eval.keras_verbose, workers=5)
    pred = pred.reshape((len(pred),) + output_shape + (num_classes,))

    if len(pred.shape) != 5:
        pred = np.expand_dims(pred, axis=3)  # 2D/3D compatibility (0 None, 1 x, 2 y, 3 new:z, 4 classes)
        log.debug("predict_model 2D detected, expanding to {}".format(pred.shape))

    return pred

