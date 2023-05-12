import os
import numpy as np
import nibabel as nib

from basics import naming


def epi_avg_img(epi: [str], out_name: str, out_dir: str = None, overwrite: bool = None):
    # sort naming
    out_avg = naming(epi[0], out_name + '-avg', out_dir, subfolder='avgs')
    out_std = naming(epi[0], out_name + '-std', out_dir, subfolder='avgs')
    out_sem = naming(epi[0], out_name + '-sem', out_dir, subfolder='avgs')
    out_paths = [out_avg, out_std, out_sem]

    # check overwrite:
    exists = True
    for out in out_paths:
        if not os.path.isfile(out):
            exists = False
    if exists:
        if overwrite:
            print("Average image exists already. Overwriting.")
        else:
            print("Average image exists already. Skipping.")
            return out_paths

    # read in all the files
    epi_img = []
    epi_data = []
    for k in range(len(epi)):
        epi_img.append(nib.load(epi[k]))
        epi_data.append(epi_img[k].get_fdata())

    # check shapes
    shape = np.shape(epi_data[0])
    for k in range(len(epi)):
        if shape(epi_data[k]) != shape: # todo: raise exception statt print und return
            print("Shape missmatch in 'epi_avg_img'. All epi images must have same shape.")
            return

    if len(shape) == 4:
        time = True
    elif len(shape) == 3:
        time = False
    else:
        print("Dimension error in 'epi_avg_img': epis must be 3D or 4D")

    # make empty avg, std and sem images
    avg_data = np.zeros(shape)
    std_data = np.zeros(shape)
    sem_data = np.zeros(shape)

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if time:
                    for t in range(shape[3]):
                        values = []
                        for k in range(len(epi)):
                            values.append(epi_data[k][x,y,z,t])
                        avg_data[x, y, z, t] = np.mean[values]
                        std = np.std(values)
                        std_data[x, y, z, t] = std
                        sem_data[x, y, z, t] = std / np.sqrt(len(epi))

    # safe output
    avg_img = nib.Nifti1Image(avg_data, epi_img[0].affine, epi_img[0].header)
    nib.safe(avg_img, out_avg)
    std_img = nib.Nifti1Image(std_data, epi_img[0].affine, epi_img[0].header)
    nib.safe(std_img, out_std)
    sem_img = nib.Nifti1Image(sem_data, epi_img[0].affine, epi_img[0].header)
    nib.safe(sem_img, out_sem)

    return out_paths


def epi_average_images(files: [str], out_name: str, out_dir: str = None, overwrite: bool = False):
    """
    Returns average image of input files, all dimensions preserved.
    Parameters:
    ------------
    files: [str]
        List of images that will be combined. All images must be of identical shape.
    name: str
        Name of output file, excluding path but including file extension.
    out_path: str [Optional]
        Path where the output will be saved. Default ist path of fist image.
    overwrite: bool [Optional]
        Whether an existing file matching the output will be overwritten. Default is False.

    Returns path to output image as string.
    """

    num = len(files)
    # figure out file naming
    out_path = naming(files[0], out_name, out_dir)

    # check overwrite
    if os.path.isfile(out_path):
        if overwrite:
            print("File '" + out_path + "' already exists. Overwriting.")
        else:
            print("File '" + out_path + "' already exists. Skipping epi_combine_images()")
            return out_path

    # read images
    images = []
    data = []
    for i in range(num):
        images.append(nib.load(files[i]))
        data.append(images[i].get_fdata())

    # get shapes and check shape match
    shape = np.shape(data[0])
    for k in range(num):
        if np.shape(data[k]) != shape:
            print('Shape missmatch in "epi_combine_images": Input must all have the same shape')
            return

    # new data array:
    data_combined = np.zeros(shape)
    for k in range(num):
        data_combined += data[k]
    data_combined = data_combined / num

    # save result, return path
    out_image = nib.Nifti1Image(data_combined, images[0].affine, images[0].header)
    nib.save(out_image, out_path)
    return out_path


def epi_average_blocks(file: str, blocks: [int], skip_first: bool = False, out_name: str = None, out_dir: str = None, overwrite: bool = False):
    """
    Calculates the average images over multiple similar time periods. Intended for use with epi time series, where the paradigm repeats periodically.
    Input length in time dimension must be a multiple of the period length, although the very first block can be skipped.
    Parameters
    ----------
    file: str
        File that will be processed. Only 4D-images are supported, with time as the 4th dimension (nifti default).
    blocks: [int]
        List of block sizes that periodically repeat. Will be used to calculate total period lenth.
    skip_first: bool [Optional]
        Whether to skip the very first block. Default is False
    out_name: str [Optional]
        Name of output file. Default is name of input file plus a '_blockavg-<block[0]>-<block[1]>-...'-suffix
    out_dir: str [Optional]
        Path of ouput file. Default is path of input file.
    overwrite: bool [Optional]
        Whether an existing file matching the output will be overwritten. Default is False.

    Returns .nii with time-lenght of one period. Images are average over all periodes of images that sheare inner-period position
    -------
    """
    # figure out file naming
    suffix = '_blockavg'
    for k in blocks:
        suffix += '-' + str(k)
    out_path = naming(file, out_name, out_dir, suffix=suffix)

    # check overwrite
    if os.path.isfile(out_path):
        if overwrite:
            print("File '" + out_path + "' already exists. Overwriting.")
        else:
            print("File '" + out_path + "' already exists. Skipping epi_combine_images()")
            return out_path

    # read in file
    image = nib.load(file)
    data = image.get_fdata()
    shape = np.shape(data)
    time = shape[3]
    bigblock = sum(blocks)

    if skip_first:
        data = data[:, :, :, blocks[0]:]
        time = time - blocks[0]

    if time % bigblock != 0:
        print('Image number missmatch in epi_combine_blocks: Could not fit block design into number of images')
        return
    else:
        num_blocks = int(time / bigblock)

    newshape = shape[0:3] + (bigblock,)

    data_combined = np.zeros(newshape)
    pos = 0
    for i in range(num_blocks):
        data_combined += data[:, :, :, pos:pos + bigblock]
        pos = i + bigblock
    data_combined = data_combined / num_blocks

    out_image = nib.Nifti1Image(data_combined, image.affine, image.header)
    nib.save(out_image, out_path)
    return out_path


def snipp_blocks(file: str, blocks: [int], skip_first: bool = False, out_name: str = None, out_dir: str = None, overwrite: bool = False, suffix: str = '_block'):
    """
    Calculates the average images over multiple similar time periods. Intended for use with epi time series, where the paradigm repeats periodically.
    Input length in time dimension must be a multiple of the period length, although the very first block can be skipped.
    Parameters
    ----------
    file: str
        File that will be processed. Only 4D-images are supported, with time as the 4th dimension (nifti default).
    blocks: [int]
        List of block sizes that periodically repeat. Will be used to calculate total period lenth.
    skip_first: bool [Optional]
        Whether to skip the very first block. Default is False
    out_name: str [Optional]
        Name of output file. Default is name of input file plus a '_blockavg-<block[0]>-<block[1]>-...'-suffix
    out_dir: str [Optional]
        Path of ouput file. Default is path of input file.
    overwrite: bool [Optional]
        Whether an existing file matching the output will be overwritten. Default is False.

    Returns .nii with time-lenght of one period. Images are average over all periodes of images that sheare inner-period position
    -------
    """

    # read in file
    image = nib.load(file)
    data = image.get_fdata()

    shape = np.shape(data)
    time = shape[3]
    bigblock = sum(blocks)

    if skip_first:
        data = data[:, :, :, blocks[0]:]
        time = time - blocks[0]

    if time % bigblock != 0:
        print('Image number missmatch in epi_combine_blocks: Could not fit block design into number of images')
        return
    else:
        num_blocks = int(time / bigblock)

    # figure out file naming
    if out_dir is None:
        infile, indir = os.path.split(file)
        inbase, inext1 = os.path.splitext(infile)
        inbase, inext2 = os.path.splitext(inbase)
        out_dir = os.path.join(indir, inbase + '_blocks')
    out_paths = []
    for b in range(num_blocks):
        if out_name is not None:
            thisname = out_name + suffix + str(b+1)
        out_paths.append(naming(file, thisname, out_dir, suffix=suffix + str(b)))

    # check overwrite
    exists = False
    for thispath in out_paths:
        if os.path.isfile(thispath):
            exists = True
    if exists:
        if overwrite:
            print("Some Block files already exists. Overwriting.")
        else:
            print("Some Block files already exists. Skipping Block separation.")
            return out_paths

    # save separate blocks as niftis
    pos = 0
    for i in range(num_blocks):
        thisblock = data[:, :, :, pos:pos + bigblock]
        pos = i + bigblock
        out_image = nib.Nifti1Image(thisblock, image.affine, image.header)
        nib.save(out_image, out_paths[i])
    return out_paths

