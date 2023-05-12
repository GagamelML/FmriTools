import numpy as np
import nibabel as nib
import os
from basics import naming


def grow_layers(rim: str, num_layers: int = 20, out_name: str = None, out_dir: str = None,
                overwrite: bool = False,
                path_laynii: str = '/data/tu_mlohoff/programme/LayNii_v2.3.0_Linux64/'):
    """
    Simple Laynii wrapper. Executes LN_GROW_LAYERS with some custom input.
    Parameters
    ----------
    rim: str
        rim file for tissue borders. ('-rim' in Laynii)
            Values of 0 are to be ignored;
            values of 1 denote CSF;
            values of 2 denote WM;
            values of 3 denote GM
    num_layers: int [Optional]
        Number of layers ('-N' in Laynii)
        Default is 20
    out_name: str [Optional]
        Output file name. Default is original name with '_layers'-suffix
    out_dir: str [Optional]
        Output path. Default is path of input rim-file
    overwrite: bool [Optional]
        Whether an existing file matching the output will be overwritten. Default is False.
    path_laynii: str [Optional]
        Path to laynii. It is recommended to edit the default path in this script, rather than passing it as a parameter.

    Returns output path
    -------

    """
    # get input names
    tissue_path, tissue_name = os.path.split(rim)
    tissue_basename, tissue_ext = os.path.splitext(tissue_name)

    # input must be .nii
    if tissue_ext != '.nii':
        print("Input error in calc_layers: 'tissue_class' must be '.nii'-file")
        return

    # figure out file naming
    out_path = naming(rim, out_name, out_dir, suffix='_layers')

    # check overwrite
    if os.path.isfile(out_path):
        if overwrite:
            print("File '" + out_path + "' already exists. Overwriting.")
        else:
            print("File '" + out_path + "' already exists. Skipping 'calc_layers'")
            return out_path


    os.environ['PATH'] += os.path.pathsep + path_laynii
    grow_layers_command = 'LN_GROW_LAYERS' + ' -rim ' + rim + ' -N ' + str(num_layers) + ' -output ' + out_path
    print(grow_layers_command)
    os.system(grow_layers_command)

    return out_path




def layer_sort_coordinates(layers: str, mask: str = None, num_layers: int = None):
    """
    Sorts voxel coordinates by layer.
    Parameters
    ----------
    layers: str
        Layers file, intended for use with output of layniis 'LN_GROW_LAYERS'
    mask: str [Optional]
        Mask file. Only voxels with non-zero in mask will be included in output
    num_layers: int [Optional]
        Specify number of layers. Must be smaller than or equal to N used in laynii.
        Default: Max. value found in layers-file

    Returns list of lists (1 per Layer) of tuples (coordinates that layer)
    -------

    """
    # read files
    layers_img = nib.load(layers)
    layers_data = layers_img.get_fdata()

    # check shape missmatch
    shape = np.shape(layers_data)

    # get mask sorted
    if mask is not None:
        mask_img = nib.load(mask)
        mask_data = mask_img.get_fdata()
        if shape != np.shape(mask_data):
            print('Shape missmatch in layer_sort_coordinates: layers and mask must have same shape')
            return
    else:
        mask_data = np.ones(shape)

    # get number of layers
    if num_layers is None:
        num_layers = int(np.max(layers_data))

    # create nested list of lists to sort all coordinates
    layer_sorted_coordinates = []
    for l in range(num_layers):
        layer_sorted_coordinates.append([])

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if mask_data[x, y, z] != 0:
                    this_layer = int(layers_data[x, y, z])
                    layer_sorted_coordinates[this_layer - 1].append((x, y, z))

    return layer_sorted_coordinates


def epi_layeravgs_vs_time(epi: [str], layers: str, mask: str = None, num_layers: int = None,
                          layer_blocks: [(int, int)] = None, time_blocks: [(int, int)] = None,
                          baseline: (int, int) = None) -> str:
    """
    Calculates mean, std and sem of voxel values, split by layer and time.
    Parameters
    ----------
    epi: str
        epi image
    layers
        Layer file, intended for use with LN_GROW_LAYERS
    mask: str [Optional]
        Only voxels with non-zero in mask will be included
    num_layers: int [Optional]
        Specify number of layers. Must be smaller than or equal to N used in laynii.
        Default: Max. value found in layers-file
    baseline: tuple of int
        Average of the values in a layer at baseline times will be subtracted from all values in that layer.
        Input should be tuple of baseline indices with [first, last]. Counting starts at 0
    Returns list (layers) of list (times) of dictionarys ('mean', 'std', 'sem', 'time'(time index), 'laminar'(layer index), 'num'(number of elements in laminar))
    -------

    """
    # read in all the files
    epi_img = []
    epi_data = []
    for k in range(len(epi)):
        epi_img.append(nib.load(epi[k]))
        epi_data.append(epi_img[k].get_fdata())

    # check shapes
    shape = np.shape(epi_data[0])
    for k in range(len(epi)):
        if np.shape(epi_data[k]) != shape:
            print("Shape missmatch in 'epi_layeravg_vs_time'. All epi images must have same shape.")
            return

    # get time length of epis
    times = shape[3]

    # sort num_layers
    if num_layers is None:
        layer_img = nib.load(layers)
        layers_data = layer_img.get_fdata()
        num_layers = int(np.max(layers_data))

    # sort combined blocks
    if layer_blocks is None:
        layer_blocks = [(x, x) for x in range(num_layers)]
    if time_blocks is None:
        time_blocks = [(t, t) for t in range(times)]

    # voxels sorted as members of layers
    members = layer_sort_coordinates(layers, mask)

    # lb = layerblock, tb = timeblock
    results_alllb_alltb_allepi = []
    for lb in layer_blocks:
        results_thislb_alltb_allepi = []
        values_thislb_alltimes_allepi = []
        members_lb = []
        for l in range(lb[0], lb[1] + 1):
            members_lb += members[l]
        for t in range(times):
            values_thislb_thistime_allepi = []
            for b in range(len(epi)):
                for coord in members_lb:
                    thisfullcoord = coord + (t,)
                    thisval = epi_data[b][thisfullcoord]
                    values_thislb_thistime_allepi.append(thisval)
            values_thislb_alltimes_allepi.append(values_thislb_thistime_allepi)

        # calc baseline for layer
        values_baseline_thislb = []
        for t in range(baseline[0], baseline[1] + 1):
            values_baseline_thislb += values_thislb_alltimes_allepi[t]
        baseline_thislb = np.mean(values_baseline_thislb)

        # collapse time into blocks
        for tb in time_blocks:
            values_thislb_thistb = []
            for time in range(tb[0], tb[1] + 1):
                values_thislb_thistb += values_thislb_alltimes_allepi[time]

            mean_nobs = np.mean(values_thislb_thistb)
            mean = mean_nobs - baseline_thislb
            std = np.std(values_thislb_thistb)
            num = len(values_thislb_thistb)
            sem = std / np.sqrt(num)
            results_thislb_thistb_allepi = {"mean": mean, "std": std, "sem": sem, 'times': tb, 'laminars': lb, 'num': num}
            results_thislb_alltb_allepi.append(results_thislb_thistb_allepi)

        results_alllb_alltb_allepi.append(results_thislb_alltb_allepi)

    return results_alllb_alltb_allepi


def analysis_combinelayers(data, layers: list[tuple] = None, times: list[tuple] = None):
    """

    Parameters
    ----------
    data: list of list of dict (see epi_layeravgs_vs_time output)
        Inout data
    layers: list(tuple)
        List of blocks that will be combined in layer domain. Syntax (start, stop). Indices start at 1.
        Indices not included in any touple will be discarded. To include single layers, use (index, index)
    times: list(tuple)
        List of blocks that will be combined in time domain. Same syntax as layers. Either layers or times must be passed.
    Returns data in same architecture, but with intended blocks combined.
    -------

    """
    if layers is None and times is None:
        print("analysis_combinelayers needs either layers or times to combine")
        return

    layers_ges = len(data)
    times_ges = len(data[0])

    if layers is not None:
        layer_blocked_data_alltimes_allblocks = []
        for t in times_ges:
            layer_blocked_data_thistime_allblocks = []
            for layer_block in layers:
                means = []
                nums = []
                stds = []
                blocklength = layer_block[1] - layer_block[0] + 1
                for l in range(layer_block[0]-1, layer_block[1]):
                    means.append(data[l][t]['mean'])
                    nums.append(data[l][t]['num'])
                    stds.append(data[l][t]['std'])

                ges_num = sum(nums)
                fracs = [num / ges_num for num in nums]
                mean_ges = sum(np.multiply(means, fracs))
                std_dummy_squared = [stds[i]**2 * fracs[i] for i in range(blocklength)]
                std_ges_squared = np.std(means)**2 + sum(std_dummy_squared)
                std_ges = np.sqrt(std_ges_squared)
                sem_ges = std_ges / np.sqrt(ges_num)

                layer_blocked_data_thistime_thisblock = {"mean": mean_ges, "std": std_ges, "sem": sem_ges, 'time': t, 'laminar': layer_block + 1, 'num': ges_num}
                layer_blocked_data_thistime_allblocks. append(layer_blocked_data_thistime_thisblock)

            layer_blocked_data_alltimes_allblocks.append(layer_blocked_data_thistime_allblocks)





