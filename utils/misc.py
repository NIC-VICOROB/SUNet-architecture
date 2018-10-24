import math

def get_resampling_indexes(num_patches, num_resampled):
    assert num_patches > 0

    resampled_idxs = list()
    sampling_left = num_resampled

    # Repeat all patches until sampling_left is smaller than num_patches
    if num_patches < num_resampled:
        while sampling_left >= num_patches:
            resampled_idxs += range(0, num_patches)
            sampling_left -= num_patches

    # Fill rest of indexes with uniform undersampling
    if sampling_left > 0:
        sampling_step = float(num_patches) / sampling_left
        sampling_point = 0.0
        for i in range(sampling_left):
            resampled_idxs.append(int(math.floor(sampling_point)))
            sampling_point += sampling_step

    assert len(resampled_idxs) == num_resampled
    return resampled_idxs
