"""
ANA <-> ORIG registration

The purpose of the following script is to compute the deformation field for the registration 
between antomy in conformed freesurfer space and native anatomical space. The transformation can be
computed from the headers of source and target images.

created by Daniel Haenelt
Date created: 06-01-2020
Last modified: 06-01-2020
"""
import os
import shutil as sh
from lib.io.get_filename import get_filename
from lib.io.mgh2nii import mgh2nii
from lib.registration.get_scanner_transform import get_scanner_transform

# input data
file_t1 = "/data/pt_01880/Experiment1_ODC/p3/anatomy/S22_MP2RAGE_0p7_T1_Images_2.45.nii"
file_orig = "/data/pt_01880/Experiment1_ODC/p3/anatomy/freesurfer/mri/orig.mgz"
path_output = "/data/pt_01880/odc_temp/deformation/header"
cleanup = False

""" do not edit below """

"""
set folder structure
"""
path_temp = os.path.join(path_output,"temp")

if not os.path.exists(path_output):
    os.makedirs(path_output)

if not os.path.exists(path_temp):
    os.makedirs(path_temp)

# get filenames
_, _, ext_orig = get_filename(file_orig)
_, _, ext_t1 = get_filename(file_t1)

# copy input files
sh.copyfile(file_orig, os.path.join(path_temp,"orig"+ext_orig))
sh.copyfile(file_t1, os.path.join(path_temp,"T1"+ext_t1))

"""
convert to nifti
"""
if ext_orig != ".nii":
    mgh2nii(os.path.join(path_temp,"orig"+ext_orig), path_temp)

if ext_t1 != ".nii":
    mgh2nii(os.path.join(path_temp,"orig"+ext_orig), path_temp)

"""
scanner transformation
"""
get_scanner_transform(os.path.join(path_temp,"orig.nii"),os.path.join(path_temp,"T1.nii"),path_temp)
get_scanner_transform(os.path.join(path_temp,"T1.nii"),os.path.join(path_temp,"orig.nii"),path_temp)

"""
get output
"""
os.rename(os.path.join(path_temp,"orig_2_T1_scanner.nii"),
          os.path.join(path_output,"orig_2_T1_scanner.nii"))
os.rename(os.path.join(path_temp,"T1_2_orig_scanner.nii"),
          os.path.join(path_output,"T1_2_orig_scanner.nii"))

# clean intermediate files
if cleanup:
    sh.rmtree(path_temp, ignore_errors=True)