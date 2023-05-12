# -*- coding: utf-8 -*-
"""
ANA <-> EPI registration
Segmentation

The purpose of the following script is to compute the deformation field for the
registration between antomy and EPI in native space. The mask should be in ana
space. Optionally, a second mask can be applied which must be in orig space. The
script consists of the following steps:
    1. enhance brainmask if second mask is given (brain.finalsurf.mgz)
    2. set output folder structure
    3. scanner transform t1 <-> epi
    4. n4 correction epi
    5. clean ana (remove ceiling and normalise)
    6. mask t1 and epi
    7. antsreg
    8. clean deformations
    9. expand deformations
    10. apply deformation
    11. start segmentation

The script needs an installation of freesurfer and ants.

"""

# python standard library inputs
import os
import shutil as sh

# external inputs
import nibabel as nb
from nipype.interfaces.ants import N4BiasFieldCorrection
from nighres.registration import embedded_antsreg, apply_coordinate_mappings
from nighres.segmentation import fuzzy_cmeans

# local inputs
from fmri_tools.cmap.clean_coordinate_mapping import clean_coordinate_mapping
from fmri_tools.cmap.expand_coordinate_mapping import expand_coordinate_mapping
from fmri_tools.skullstrip.skullstrip_refined import skullstrip_refined
from fmri_tools.registration.mask_ana import mask_ana
from fmri_tools.registration.mask_epi import mask_epi
from fmri_tools.registration.clean_ana import clean_ana

# input data
file_mean_epi = "/data/p_02808/Sessions/2023-04-18_S06_15636.fa/imagedata/epi/meanS12_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii"
file_t1 = "/data/p_02808/Sessions/2023-04-18_S06_15636.fa/imagedata/mp2rage/S4_mp2rage_0p7_iPAT2_phPF68_FA5-3_TI900-2750_BW250_T1_Images_2.01.nii"
file_mask1 = "/data/p_02808/Sessions/2023-04-18_S06_15636.fa/fmritools/skull/skullstrip_mask.nii"  # skullstrip_mask
file_mask2 = None  # brain.finalsurfs
path_output = "/data/p_02808/Sessions/2023-04-18_S06_15636.fa/fmritools"
clean_cmap = True
expand_cmap = True
cleanup = False

# parameters for epi skullstrip
niter_mask = 3
sigma_mask = 3

# parameters for syn 
run_rigid = True
rigid_iterations = 1000
run_affine = False
affine_iterations = 1000
run_syn = True
coarse_iterations = 50
medium_iterations = 150
fine_iterations = 100
cost_function = 'CrossCorrelation'
interpolation = 'Linear'

# do not edit below

# enhance brainmask
if file_mask2 is not None:
    file_mask1 = skullstrip_refined(file_mask1, file_mask2)

# set folder structure
path_temp = os.path.join(path_output, "temp")
path_epi = os.path.join(path_temp, "epi")
path_t1 = os.path.join(path_temp, "t1")
path_syn = os.path.join(path_temp, "syn")

if not os.path.exists(path_output):
    os.makedirs(path_output)

if not os.path.exists(path_temp):
    os.makedirs(path_temp)

if not os.path.exists(path_epi):
    os.makedirs(path_epi)

if not os.path.exists(path_t1):
    os.makedirs(path_t1)

if not os.path.exists(path_syn):
    os.makedirs(path_syn)

# copy input files
sh.copyfile(file_mean_epi, os.path.join(path_epi, "epi.nii"))
sh.copyfile(file_t1, os.path.join(path_t1, "T1.nii"))
sh.copyfile(file_mask1, os.path.join(path_t1, "mask.nii"))

# bias field correction to epi
n4 = N4BiasFieldCorrection()
n4.inputs.dimension = 3
n4.inputs.input_image = os.path.join(path_epi, "epi.nii")
n4.inputs.bias_image = os.path.join(path_epi, 'n4bias.nii')
n4.inputs.output_image = os.path.join(path_epi, "bepi.nii")
n4.run()

# clean ana
clean_ana(os.path.join(path_t1, "T1.nii"), 1000.0, 4095.0, overwrite=True)

# mask t1 and epi
mask_ana(os.path.join(path_t1, "T1.nii"), os.path.join(path_t1, "mask.nii"),
         background_bright=False)
mask_epi(os.path.join(path_epi, "bepi.nii"),
         os.path.join(path_t1, "pT1.nii"),
         os.path.join(path_t1, "mask.nii"),
         niter_mask, sigma_mask)

# syn
embedded_antsreg(os.path.join(path_t1, "pT1.nii"),  # source image
                 os.path.join(path_epi, "pbepi.nii"),  # target image
                 run_rigid,  # whether or not to run a rigid registration first
                 rigid_iterations,  # number of iterations in the rigid step
                 run_affine,  # whether or not to run an affine registration first
                 affine_iterations,  # number of iterations in the affine step
                 run_syn,  # whether or not to run a SyN registration
                 coarse_iterations,  # number of iterations at the coarse level
                 medium_iterations,  # number of iterations at the medium level
                 fine_iterations,  # number of iterations at the fine level
                 8,
                 cost_function,  # CrossCorrelation or MutualInformation
                 interpolation,  # interpolation for registration result (NeareastNeighbor or Linear)
                 convergence=1e-6,  # threshold for convergence (can make algorithm very slow)
                 ignore_affine=False,  # ignore the affine matrix information extracted from the image header
                 ignore_header=False,  # ignore the orientation information and affine matrix information extracted from the image header
                 save_data=True,  # save output data to file
                 overwrite=False,  # overwrite existing results
                 output_dir=path_syn,  # output directory
                 file_name="syn",  # output basename
                 )

# rename final deformations
os.rename(os.path.join(path_syn, "syn_ants-map.nii.gz"),
          os.path.join(path_output, "ana2epi.nii.gz"))
os.rename(os.path.join(path_syn, "syn_ants-invmap.nii.gz"),
          os.path.join(path_output, "epi2ana.nii.gz"))

# clean deformation
if clean_cmap:
    epi2ana_cleaned = clean_coordinate_mapping(
        os.path.join(path_output, "ana2epi.nii.gz"),
        os.path.join(path_output, "epi2ana.nii.gz"),
        overwrite_file=True,
        save_mask=False)

    # write mask
    nb.save(epi2ana_cleaned["mask"],
            os.path.join(path_output, "epi2ana_mask.nii.gz"))

# expand deformation
if expand_cmap:
    _ = expand_coordinate_mapping(os.path.join(path_output, "ana2epi.nii.gz"),
                                  path_output,
                                  name_output="ana2epi",
                                  write_output=True)

    _ = expand_coordinate_mapping(os.path.join(path_output, "epi2ana.nii.gz"),
                                  path_output,
                                  name_output="epi2ana",
                                  write_output=True)

# apply deformation
# ana -> epi
apply_coordinate_mappings(file_t1,  # input
                          os.path.join(path_output, "ana2epi.nii.gz"),  # cmap
                          interpolation="linear",  # nearest or linear
                          padding="zero",  # closest, zero or max
                          save_data=True,  # save output data to file (boolean)
                          overwrite=False,  # overwrite existing results (boolean)
                          output_dir=path_output,  # output directory
                          file_name="ana2epi_example"  # base name with file extension for output
                          )


# rename final deformation examples
os.rename(os.path.join(path_output, "ana2epi_example_def-img.nii.gz"),
          os.path.join(path_output, "ana2epi_t1w.nii.gz"))

# clean intermediate files
if cleanup:
    sh.rmtree(path_temp, ignore_errors=True)



