import numpy as np
import nibabel as nib
import os
from manuel.old.naming import naming


def transform_classification(class_file: str, keys: [[int, int]] = [[1, 2], [2, 3], [3, 1]], out_dir: str = None,
                             out_name: str = None, overwrite: bool = False, mask: str = None):
    """
    Transforms tissue classification file with regard to tissue keys and file type (output is .nii)
    Parameters
    ----------
    class_file: str
        full path to input file. File must be 2D or 3D
    keys: [[int,int]] (Optional)
        List of int pairs that you want to exchange. First value of tuple in input file will be passed to second value in tuple in output file:
        Default is [[1, 2], [2, 3], [3, 1]], the keys to transform from nighres.fuzzy_means to laynii convention
    out_dir: str (Optional)
        Path of output file. May contain .nii-file-name already (Will override out_name in this case).
        If empty, path will be taken from input file.
    out_name: str (Optional)
        Name of output file (mus include .nii or similar extension). Overwritten if out_path contains .nii extension.
        If empty, base name from input and .nii-extension will be used wit a _transformed addition
    overwrite: bool (optional)
        If False, existing files will not be overwritten. If a file matching output parameters already exists, transformation will be aborted.
        Default is False
    mask: str (Optional)
        Additional Mask to multiply the result with (ideally mask of 1 and 0)
        Default is None

    Returns path of output file as string.
    """

    # Figure out the output file path from different input options
    out_path = naming(class_file, out_name, out_dir, suffix='_transformed')

    # check if such a file already exists
    if os.path.isfile(out_path):
        if overwrite:
            print('Transformed classification file exists already. Overriding it.')
        else:
            print('Transformed classification file exists already. Skipping transformation')
            return

    # read file and check dimensions
    class_img = nib.load(class_file)
    class_data = class_img.get_fdata()
    shape = np.shape(class_data)

    # check dimensions and create new array
    if len(shape) == 3:
        out_data = np.zeros(shape)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    val_in = class_data[x, y, z]
                    val_out = val_in
                    for i in keys:
                        if val_in == i[0]:
                            val_out = i[1]
                    out_data[x, y, z] = val_out
    elif len(shape) == 2:
        out_data = np.zeros(shape)
        for x in range(shape[0]):
            for y in range(shape[1]):
                val_in = class_data[x, y]
                val_out = val_in
                for i in keys:
                    if val_in == i[0]:
                        val_out = i[1]
                out_data[x, y] = val_out
    else:
        print('Error transforming classification: File must have 2 or 3 dimensions')
        return

    # apply mask
    if mask is not None:
        mask_img = nib.load(mask)
        mask_data = mask_img.get_fdata()
        if np.shape(mask_data) == shape:
            out_data = out_data * mask_data
        else:
            print('Shape missmatch. Skipping mask.')

    # save data to file
    class_out = nib.Nifti1Image(out_data, class_img.affine, class_img.header)
    nib.save(class_out, out_path)
    return out_path

