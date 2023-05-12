import os
import numpy as np
import nibabel as nib
from basics import naming
from threshhold_glm import threshhold_glm


def apply_mask(files, mask: str = None, out_names=None, out_dir: str = None,
               overwrite: bool = False):
    """
    Applies mask to all images in 'files'. Saves output to new images (nifti, unless specified otehrwise in name).
    Parameters
    ----------
    files: str or [str]
        Single file or list of files that the mask needs to be applied to
    mask: str
        Mask that should be applied
    out_names: str or [str] [Optional]
        Names of output files. Must have same dimension as 'files'.
        Default is name of original file (without extension) pus '_masked.nii'-suffix
    out_dir: str [Optional]
        Path where output will be saved.
        Default is path of first file in 'files'
    overwrite: bool [Optional]
        Whether an existing file matching the output will be overwritten. Default is False.

    Returns list of output files.
    -------

    """

    # filter single input
    if type(files) == str:
        files = [files]
    elif type(files) != list:
        print("Input error in 'apply_mask': 'files' must be string or list of strings")
    if out_names is not None:
        if type(out_names) == str:
            out_names = [out_names]

    # check names
    if len(files) != len(out_names):
        print("Name error in 'apply_mask': names must be of same length as files")
        return

    # figure out naming
    out_path = []
    for k in range(len(files)):
        if out_names is None:
            thisname = None
        else:
            thisname = out_names[k]
        this_out = naming(files[k], thisname, out_dir, suffix='_masked')
        out_path.append(this_out)

    # actual data processing now
    #read mask
    mask_img = nib.load(mask)
    mask_data = mask_img.get_fdata()
    mask_shape = np.shape(mask_data)

    # go through files
    for k in range(len(files)):
        this_img = nib.load(files[k])
        this_data = this_img.get_fdata()
        if np.shape(this_data) == mask_shape:
            this_data_masked = this_data * mask_data
            this_img_masked = nib.Nifti1Image(this_data_masked, this_img.affine, this_img.header)
            nib.save(this_img_masked, out_path[k])
        else:
            print("Shape error in 'apply_mask': Input needs to have same shape as mask. Skipping this input")

    if type(files) == str:
        return out_path[0]
    else:
        return out_path



def mask_from_glms(glms: [str], out_name: str, threshhold: int, threshtype: str = 'one', out_dir: str = None,
                   overwrite: bool = False):
    # sort naming
    out_path = naming(glms[0], out_name, out_dir)

    # check overwrite
    if os.path.isfile(out_path):
        if overwrite:
            print("mask '" + out_name + "' exists already. Overwriting.")
        else:
            print("mask '" + out_name + "' exists already. Skipping mask_from_glms.")
            return out_path

    # create temporary threshholded glm files
    # sort dir for those:
    if out_dir is not None:
        temp_dir = os.path.join(out_dir, 'temp/')
    else:
        first_dir, first_name = os.path.split(glms[0])
        temp_dir = os.path.join(first_dir, 'temp/')
    # now actually calculate those:
    thresh_glms = threshhold_glm(glms, threshhold=threshhold, threshtype=threshtype,
                                 suffix='_threshholded-' + str(threshhold) + '-' + threshtype,
                                 out_dir=temp_dir)

    # read in new files
    thresh_glms_img = []
    thresh_glms_data = []
    for g in thresh_glms:
        this_img = nib.load(g)
        thresh_glms_img.append(this_img)
        thresh_glms_data.append(this_img.get_fdata())

    # add all the masks. it is highly unlikely that elements will add up do exacelty 0, I am taking that chance
    thresh_glm_sum = sum(thresh_glms_data)

    # make new mask
    new_mask = np.zeros(np.shape(thresh_glms_data[0]))
    new_mask[thresh_glm_sum != 0] = 1

    # save data
    new_mask_img = nib.Nifti1Image(new_mask, thresh_glms_img[0].affine, thresh_glms_img[0].header)
    nib.save(new_mask_img, out_path)

    return out_path


def mask_multiplyer(masks: str, out_name: str, out_dir: str = None, overwrite: bool = False):
    # sort naming
    out_path = naming(masks[0], out_name, out_dir)

    # check overwrite
    if os.path.isfile(out_path):
        if overwrite:
            print("Mask '" + out_name + "' exists already. Overwriting.")
        else:
            print("Mask '" + out_name + "' exists already. Skipping.")
            return out_path

    # read data
    masks_img = []
    masks_data = []
    for k in range(len(masks)):
        masks_img.append(nib.load(masks[k]))
        masks_data.append(masks_img[k].get_fdata())

    # check shapes:
    shape = np.shape(masks_data[0])
    for k in range(len(masks)):
        if np.shape(masks_data[k]) != shape:
            print("Shape missmatch in mask_substractor: mask #" + str(k) + " does not have same size as first mask")
            return

    # multiply data
    new_mask_data = np.ones(shape)
    for md in masks_data:
        new_mask_data = new_mask_data * md

    # safe and return output
    new_mask_img = nib.Nifti1Image(new_mask_data, masks_img[0].affine, masks_img[0].header)
    nib.save(new_mask_img, out_path)

    return out_path

