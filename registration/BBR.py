"""
BBR

The script calls the freesurfer bbr method. Inputs are not checked if they are valid. Two white
surfaces are expected corresponding to left and right hemisphere. Therefore, the input should 
contain a list with two entries. The target volume is used as freesurfer orig file. A freesurfer
brainmask file is generated by skullstripping the target file using a mask from a separate 
anatomical scan. The mask is transformed to orig space by registering ana and target. The resulting 
transformation matrix is expressed as coordinate mapping and saved as nifti volume. Example files
are created by applying the resulting coordinate mappings to source and target images.

Before running the script, login to queen via ssh and set the freesurfer and ANTS environments by 
calling FREESURFER and ANTSENV in the terminal.

created by Daniel Haenelt
Date created: 19-06-2020
Last modified: 23-06-2020
"""
import os
import numpy as np
import nibabel as nb
import shutil as sh
from nibabel.affines import apply_affine
from nighres.registration import apply_coordinate_mappings
from nipype.interfaces.ants import N4BiasFieldCorrection
from lib.io.get_filename import get_filename
from lib.io.mgh2nii import mgh2nii
from lib.io.read_vox2vox import read_vox2vox
from lib.registration.clean_ana import clean_ana
from lib.registration.mask_ana import mask_ana
from lib.registration.mask_epi import mask_epi
from lib.cmap.generate_coordinate_mapping import generate_coordinate_mapping
from lib.utils.remove_nans import remove_nans
from lib.utils.multiply_images import multiply_images

# input
input_white = ["/data/pt_01880/Experiment1_ODC/p1/anatomy/dense_refined/lh.white_def2_final",
               "/data/pt_01880/Experiment1_ODC/p1/anatomy/dense_refined/rh.white_def2_final",
               ]
input_ana = "/data/pt_01880/Experiment1_ODC/p1/anatomy/S5_MP2RAGE_0p7_T1_Images_2.45.nii"
input_mask = "/data/pt_01880/Experiment1_ODC/p1/anatomy/skull/skullstrip_mask.nii"
input_target = "/data/pt_01880/Experiment1_ODC/p1/resting_state/mean_udata.nii"
input_source = "/data/pt_01880/Experiment1_ODC/p1/retinotopy/diagnosis/mean_uadata.nii"
path_output = "/data/pt_01880/thisisthefinaltest2"
init_reg = "freesurfer" # header, freesurfer, fsl
cleanup = False

# parameters for orig skullstrip
niter_mask = 3
sigma_mask = 3

# maximum number of interations in bbr
nmax = 1000

""" do not edit below """

# set freesurfer path environment
os.environ["SUBJECTS_DIR"] = path_output

# freesurfer subject
sub = "temp"

# make output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

# mimic freesurfer folder structure (with some additional folders for intermediate files)
path_sub = os.path.join(path_output, sub)
path_mri = os.path.join(path_sub, "mri")
path_surf = os.path.join(path_sub, "surf")
path_t1 = os.path.join(path_sub, "t1")
path_bbr = os.path.join(path_sub, "bbr")

os.makedirs(path_sub)
os.makedirs(path_mri)
os.makedirs(path_surf)
os.makedirs(path_t1)
os.makedirs(path_bbr)

# change path to output folder
os.chdir(path_output)

# copy surfaces
for i in range(len(input_white)):
    
    # white surface
    _, hemi, _ = get_filename(input_white[i])
    sh.copyfile(input_white[i], os.path.join(path_surf,hemi+".white"))

# copy volumes
sh.copy(input_target, os.path.join(path_mri,"orig.nii"))
sh.copy(input_source, os.path.join(path_bbr,"source.nii"))
sh.copy(input_ana, os.path.join(path_t1,"T1.nii"))
sh.copy(input_mask, os.path.join(path_t1, "mask.nii"))

# remove nans
remove_nans(os.path.join(path_mri,"orig.nii"), os.path.join(path_mri,"orig.nii"))
remove_nans(os.path.join(path_bbr,"source.nii"),os.path.join(path_bbr,"source.nii"))
remove_nans(os.path.join(path_t1,"T1.nii"),os.path.join(path_t1,"T1.nii"))
remove_nans(os.path.join(path_t1, "mask.nii"), os.path.join(path_t1, "mask.nii"))

# get brainmask
clean_ana(os.path.join(path_t1,"T1.nii"), 1000.0, 4095.0, overwrite=True)
mask_ana(os.path.join(path_t1,"T1.nii"), os.path.join(path_t1,"mask.nii"), background_bright=False)

# bias field correction
n4 = N4BiasFieldCorrection()
n4.inputs.dimension = 3
n4.inputs.input_image = os.path.join(path_mri, "orig.nii")
n4.inputs.bias_image = os.path.join(path_mri, "n4bias.nii")
n4.inputs.output_image = os.path.join(path_mri, "borig.nii")
n4.run()

mask_epi(os.path.join(path_mri,"borig.nii"),
         os.path.join(path_t1,"pT1.nii"),
         os.path.join(path_t1,"mask.nii"),
         niter_mask, sigma_mask)

multiply_images(os.path.join(path_mri, "orig.nii"), 
                os.path.join(path_t1, "mask_def-img3.nii.gz"), 
                os.path.join(path_mri, "brainmask.nii"))

# convert orig and brainmask to mgz
mgh2nii(os.path.join(path_mri,"orig.nii"), path_mri, out_type="mgz")
mgh2nii(os.path.join(path_mri,"brainmask.nii"), path_mri, out_type="mgz")

# choose initialization method
if init_reg == "header":
    bbr_var = "--init-header"
elif init_reg == "freesurfer":
    bbr_var = "--init-coreg"
elif init_reg == "fsl":
    bbr_var = "--init-fsl"

# run freesurfer bbr
os.chdir(path_bbr)
os.system("bbregister" + \
          " --s " + sub + \
          " --mov " + os.path.join(path_bbr, "source.nii") + \
          " --bold" + \
          " --reg regheader" + \
          " --gm-proj-abs 1" + \
          " --wm-proj-abs 1" + \
          " --nmax " + str(nmax) + \
          " --o " + os.path.join(path_bbr, "registered.nii") + \
          " --lta " + os.path.join(path_bbr, "transformation.lta") + \
          " --no-cortex-label" + \
          " --6" + \
          bbr_var + \
          " --nocleanup" + \
          " --tmp " + path_bbr)

# get transformation matrix from freesurfer lta file
M, Minv = read_vox2vox(os.path.join(path_bbr, "transformation.lta"))

# target to source coordinate mapping
cmap_source = generate_coordinate_mapping(input_source, pad=0)
arr_cmap_source = cmap_source.get_fdata()

xdim = cmap_source.header["dim"][1]
ydim = cmap_source.header["dim"][2]
zdim = cmap_source.header["dim"][3]

x = arr_cmap_source[:,:,:,0].flatten()
y = arr_cmap_source[:,:,:,1].flatten()
z = arr_cmap_source[:,:,:,2].flatten()

source_listed = np.array([x,y,z]).T
source_transformed = apply_affine(M, source_listed)

x_new = np.reshape(source_transformed[:,0], (xdim,ydim,zdim))
y_new = np.reshape(source_transformed[:,1], (xdim,ydim,zdim))
z_new = np.reshape(source_transformed[:,2], (xdim,ydim,zdim))

arr_cmap_transformed = np.zeros_like(arr_cmap_source)
arr_cmap_transformed[:,:,:,0] = x_new
arr_cmap_transformed[:,:,:,1] = y_new
arr_cmap_transformed[:,:,:,2] = z_new

# nibabel instance of final cmap
t2s = nb.Nifti1Image(arr_cmap_transformed, cmap_source.affine, cmap_source.header)

# apply cmap to target
t2s_example = apply_coordinate_mappings(input_target, 
                                        mapping1=t2s, 
                                        mapping2=None, 
                                        mapping3=None, 
                                        mapping4=None, 
                                        interpolation="linear", 
                                        padding="closest", 
                                        save_data=False, 
                                        overwrite=False, 
                                        output_dir=None, 
                                        file_name=None)
   
# source to target transformation
cmap_target = generate_coordinate_mapping(input_target, pad=0)
arr_cmap_target = cmap_target.get_fdata()

xdim = cmap_target.header["dim"][1]
ydim = cmap_target.header["dim"][2]
zdim = cmap_target.header["dim"][3]

# transform source volume
x = arr_cmap_target[:,:,:,0].flatten()
y = arr_cmap_target[:,:,:,1].flatten()
z = arr_cmap_target[:,:,:,2].flatten()

target_listed = np.array([x,y,z]).T
target_transformed = apply_affine(Minv, target_listed)
    
x_new = np.reshape(target_transformed[:,0], (xdim,ydim,zdim))
y_new = np.reshape(target_transformed[:,1], (xdim,ydim,zdim))
z_new = np.reshape(target_transformed[:,2], (xdim,ydim,zdim))

arr_cmap_transformed = np.zeros_like(arr_cmap_target)
arr_cmap_transformed[:,:,:,0] = x_new
arr_cmap_transformed[:,:,:,1] = y_new
arr_cmap_transformed[:,:,:,2] = z_new

# nibabel instance of final cmap
s2t = nb.Nifti1Image(arr_cmap_transformed, cmap_target.affine, cmap_target.header)

# apply cmap to source
s2t_example = apply_coordinate_mappings(input_source, 
                                        mapping1=s2t, 
                                        mapping2=None, 
                                        mapping3=None, 
                                        mapping4=None, 
                                        interpolation="linear", 
                                        padding="closest", 
                                        save_data=False, 
                                        overwrite=False, 
                                        output_dir=None, 
                                        file_name=None)

# write output
nb.save(t2s, os.path.join(path_output,"target2source.nii.gz"))
nb.save(t2s_example["result"], os.path.join(path_output, "target2source_example.nii.gz"))
nb.save(s2t, os.path.join(path_output, "source2target.nii.gz"))
nb.save(s2t_example["result"], os.path.join(path_output, "source2target_example.nii.gz"))

# clean intermediate files
if cleanup:
    sh.rmtree(path_sub, ignore_errors=True)