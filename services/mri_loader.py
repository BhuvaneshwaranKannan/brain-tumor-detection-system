import nibabel as nib
import numpy as np

def load_mri(filepath):
    """
    Load MRI (.nii/.nii.gz) using nibabel and return volume + header.
    """
    img_obj = nib.load(filepath)
    volume = np.array(img_obj.get_fdata())
    header = img_obj.header
    img_obj.uncache()
    return volume, header
