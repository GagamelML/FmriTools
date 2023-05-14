#!/usr/bin/python3

import os.path
from typing import Tuple

import numpy as np
import nibabel as nib
import logging
from basics import naming
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def get_fit_TE(val1, TE1, val2, TE2):
    if val1 == 0 or val2 == 0:
        return (0, 1)
    fit = np.polyfit((TE1, TE2), (np.log(val1), np.log(val2)), deg=1)
    S0 = np.exp(fit[0])
    T2s = fit[1]
    out = (S0, T2s)
    return out


def calc_inner_fit(epi1_data_x, epi2_data_x, epi1_TE, epi2_TE):
    S0_map_data_x = np.zeros(np.shape(epi2_data_x))
    T2s_map_data_x = np.zeros(np.shape(epi2_data_x))
    shape = np.shape(epi2_data_x)
    for y in range(shape[0]):
        for z in range(shape[1]):
            for t in range(shape[2]):
                val1 = epi1_data_x[y, z, t]
                val2 = epi2_data_x[y, z, t]
                fit = get_fit_TE(val1, epi1_TE, val2, epi2_TE)
                S0_map_data_x[y, z, t] = fit[0]
                T2s_map_data_x[y, z, t] = fit[1]
    return S0_map_data_x, T2s_map_data_x


def calc_echotimes(epi1: str, epi1_TE: float, epi2: str, epi2_TE: float,
                   out_name: str, out_dir: str = None, overwrite: bool = False,
                   TE_wishlist: [int] = (), calc_T2sw_S0w: bool = False, calc_T2sw_noS0w: bool = False):
    epi1_img = nib.load(epi1)
    epi1_data = epi1_img.get_fdata()
    epi2_img = nib.load(epi2)
    epi2_data = epi2_img.get_fdata()

    out_paths_S0w = []
    for i in range(len(TE_wishlist)):
        TE = TE_wishlist[i]
        suffix = "TE" + str(TE) + "_S0w"
        thispath = naming(epi1, out_name, out_dir, suffix=suffix, subfolder="fitted_TE")
        out_paths_S0w.append(thispath)

    out_paths_noS0w = []
    for i in range(len(TE_wishlist)):
        TE = TE_wishlist[i]
        suffix = "TE" + str(TE) + "_noS0w"
        thispath = naming(epi1, out_name, out_dir, suffix=suffix, subfolder="fitted_TE")
        out_paths_S0w.append(thispath)

    out_paths_S0_map = naming(epi1, out_name, out_dir, suffix="S0-map", subfolder="fitted_TE")
    out_paths_T2s_map = naming(epi1, out_name, out_dir, suffix="T2s-map", subfolder="fitted_TE")

    # construct dict to return later
    output = {'S0_map': out_paths_S0_map,
              'T2s_map': out_paths_T2s_map,
              'S0w_at_TE': out_paths_S0w,
              'noS0w_at_TE': out_paths_noS0w
              }

    # check overwrite for maps
    missing = False
    if not os.path.isfile(out_paths_S0_map) or not os.path.isfile(out_paths_T2s_map):
        missing = True

    if calc_T2sw_S0w:
        for i in range(len(TE_wishlist)):
            if not os.path.isfile(out_paths_S0w[i]):
                missing = True
    if calc_T2sw_noS0w:
        for i in range(len(TE_wishlist)):
            if not os.path.isfile(out_paths_noS0w[i]):
                missing = True

    if not missing:
        if overwrite:
            print('TE fit files exist already. Overwriting.')
        else:
            print('TE fit files exist already. Skipping.')
            return output

    # initialize maps
    shape = np.shape(epi1_data)
    S0_map_data = np.zeros(shape)
    T2s_map_data = np.zeros(shape)

    with ProcessPoolExecutor() as tp:
        threads = dict()
        for x in range(shape[0]):
            epi1_data_x = epi1_data[x, :, :, :]
            epi2_data_x = epi2_data[x, :, :, :]
            threads[x] = tp.submit(calc_inner_fit, epi1_data_x, epi2_data_x, epi1_TE, epi2_TE)
        for x in range(shape[0]):
            logging.info(f'volume {x} done')
            volumes = threads[x].result()
            S0_map_data[x, :, :, :] = volumes[0]
            T2s_map_data[x, :, :, :] = volumes[1]

    S0_map_img = nib.Nifti1Image(S0_map_data, epi1_img.affine, epi1_img.header)
    nib.save(S0_map_img, out_paths_S0_map)
    T2s_map_img = nib.Nifti1Image(T2s_map_data, epi1_img.affine, epi1_img.header)
    nib.save(T2s_map_img, out_paths_T2s_map)

    if calc_T2sw_noS0w:
        T2sw_noS0w_data = []
        for i in range(len(TE_wishlist)):
            this_TE = TE_wishlist[i]
            this_TE_data = np.zeros(shape)
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        for t in range(shape[3]):
                            this_TE_data[x, y, z, t] = np.exp(-this_TE/T2s_map_data[x, y, z, t])
            T2sw_noS0w_data.append(this_TE_data)
            this_TE_img = nib.Nifti1Image(this_TE_data, epi1_img.affine, epi1_img.header)
            nib.save(this_TE_img, out_paths_noS0w[i])

    if calc_T2sw_S0w:
        T2sw_S0w_data = []
        for i in range(len(TE_wishlist)):
            this_TE = TE_wishlist[i]
            this_TE_data = np.zeros(shape)
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        for t in range(shape[3]):
                            this_TE_data[x, y, z, t] = T2sw_noS0w_data[i][x, y, z, t] * S0_map_data[x, y, z, t]
            T2sw_S0w_data.append(this_TE_data)
            this_TE_img = nib.Nifti1Image(this_TE_data, epi1_img.affine, epi1_img.header)
            nib.save(this_TE_img, out_paths_S0w[i])
            logging.info(f'image written in {out_paths_S0w[i]}')

    return output



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description='whatever')
    parser.add_argument('epi1', help='file epi1')
    parser.add_argument('epi1_TE', help='TE epi1', type=float)
    parser.add_argument('epi2', help='file epi2')
    parser.add_argument('epi2_TE', help='TE epi2', type=float)
    parser.add_argument('outname', help='output name')
    parser.add_argument('outpath', help='output path', default=os.curdir)
    parser.add_argument('--overwrite', help="overwrite", action='store_true', default=False)
    parser.add_argument('--log-level', help='logging level', default='warning')

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    calc_echotimes(args.epi1,args.epi1_TE,args.epi2,args.epi2_TE,args.outname,overwrite=args.overwrite,out_dir=args.outpath)


# "/data/p_02808/Sessions/2023-04-18_S06_15636.fa/imagedata/epi/uS12_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii", 11.8,
# "/data/p_02808/Sessions/2023-04-18_S06_15636.fa/imagedata/epi/uS12_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii", 27.8,
# "test")
