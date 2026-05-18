import numpy as np

def find_best_slice(volume):
    """
    Select the best slice from MRI volume using variance.
    """
    max_var = 0
    best_index = volume.shape[2] // 2

    for i in range(volume.shape[2]):
        slice_img = volume[:,:,i]
        var = np.var(slice_img)

        if var > max_var:
            max_var = var
            best_index = i

    return best_index
