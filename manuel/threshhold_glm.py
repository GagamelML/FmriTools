import numpy as np
import nibabel as nib

from manuel.old.naming import  naming


def threshhold_glm(files: [str], threshhold: float, threshtype: str = "one", suffix: str = None, out_dir: str = None):
    num = len(files)
    """
    Threshholds input files together. "Together" means that any voxel will either pass in all images or in none, but never in one image and not another.
    Saves threshholded files and returns list of paths to those files.
    Parameters
    ---------------
    files: [str]
        List of strings with filenames
    threshhold: float
        Threshhold value
    threshtype: "one" or "all"
        Type of threshhold that a voxel needs to pass. 
        "one": Voxel needs to be above threshhold value in one single image to pass in all images.
        "all": Voxel needs to be above threshhold value in every single image to pass in any image.
        Default is "one"
    suffix: str [optional]
        will be added at the end of the filenames when output data is saved
        Default is '_thresh<threshhold>-<threshtype>
    """
    # import data
    images = []
    data = []
    for i in range(num):
        images.append(nib.load(files[i]))
        data.append(images[i].get_fdata())

    # check shapes
    shape = np.shape(data[0])
    for k in range(num):
        if shape != np.shape(data[i]):
            print('Shape missmatch in threshhold_data: data must have identical shapes')
            return

    # flatten data for easier processing. Also create empty array with same structure to save threshholded data in.
    data_flat = []
    data_flat_threshholded = []
    for k in range(num):
        data_flat.append(np.ndarray.flatten(data[k]))
        data_flat_threshholded.append(np.zeros(len(data_flat[0])))

    # do threshholding with regard to type
    if threshtype == 'one':
        for l in range(len(data_flat[0])):
            passer = False
            for k in range(num):
                if data_flat[k][l] >= threshhold:
                    passer = True
            if passer:
                for k in range(num):
                    data_flat_threshholded[k][l] = data_flat[k][l]
    elif threshtype == 'all':
        for l in range(len(data_flat[0])):
            passer = True
            for k in range(num):
                if data_flat[k][l] < threshhold:
                    passer = False
            if passer:
                for k in range(num):
                    data_flat_threshholded[k][l] = data_flat[k][l]
    else:
        print('Parameter error in threshhold_data: threshtype must be "one" or "all"')
        return

    # reshape data:
    data_threshholded = []
    for k in range(num):
        data_threshholded.append(np.ndarray.reshape(data_flat_threshholded[k], shape))

    # figure out output file naming
    out_path = []
    if suffix is None:
        suffix = '_thresh' + str(threshhold) + '-' + threshtype
    for k in range(num):
        out_path.append(naming(files[k], None, out_dir, suffix=suffix))


    # save threshholded data
    for k in range(num):
        this_data = data_threshholded[k]
        this_header = images[k].header
        this_affine = images[k].affine
        this_path = out_path[k]
        this_rawdata = data[k]

        this_out = nib.Nifti1Image(this_data, this_affine, this_header)
        nib.save(this_out, this_path)

    return out_path



