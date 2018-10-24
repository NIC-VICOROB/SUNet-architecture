from keras import backend as K

def jaccard(y_true, y_pred, smooth=100.0):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.githuAdadeltab.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice(ground_truth, prediction):
    """
    Computing mean-class Dice similarity.
    This function assumes one-hot encoded ground truth (CATEGORICAL :D)

    :param prediction: last dimension should have ``num_classes``
    :param ground_truth: segmentation ground truth (encoded as a binary matrix)
        last dimension should be ``num_classes``
    :param weight_map:
    :return: ``1.0 - mean(Dice similarity per class)``
    """

    prediction = K.cast(prediction, dtype='float32')
    ground_truth = K.cast(ground_truth, dtype='float32')

    # computing Dice over the spatial dimensions
    reduce_axes = list(range(len(prediction.shape) - 1))
    dice_numerator = 2.0 * K.sum(prediction * ground_truth, axis=reduce_axes)
    dice_denominator = K.sum(prediction, axis=reduce_axes) + K.sum(ground_truth, axis=reduce_axes)

    epsilon_denominator = 0.0001
    dice_score = dice_numerator / (dice_denominator + epsilon_denominator)
    return 1.0 - K.mean(dice_score)

def ss(y_true, y_pred,weight_map=None,r=0.05):
    """
    Function to calculate a multiple-ground_truth version of
    the sensitivity-specificity loss defined in "Deep Convolutional
    Encoder Networks for Multiple Sclerosis Lesion Segmentation",
    Brosch et al, MICCAI 2015,
    https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1

    error is the sum of r(specificity part) and (1-r)(sensitivity part)

    :param prediction: the logits
    :param ground_truth: segmentation ground_truth.
    :param r: the 'sensitivity ratio'
        (authors suggest values from 0.01-0.10 will have similar effects)
    :return: the loss
    """

    # chosen region may contain no voxels of a given label. Prevents nans.
    eps = 1e-5

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    sq_error = K.square(y_true_f - y_pred_f)

    spec_part = K.sum(sq_error * y_true_f) / (K.sum(y_true_f) + eps)
    sens_part =  K.sum(sq_error * (1 - y_true_f)) / (K.sum(1 - y_true_f) + eps)

    return r*spec_part + (1.0 - r)*sens_part