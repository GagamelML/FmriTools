"""
EPI <-> ANA registration

The purpose of the following script is to compute the deformation field for the registration 
between antomy and EPI in native space. The script consists of the following steps:
    1. set output folder structure
    2. scanner transform t1 <-> epi
    3. n4 correction epi
    4. mask t1 and epi
    5. antsreg
    6. apply deformations

Before running the script, login to queen via ssh and set the freesurfer and ANTS environments by 
calling FREESURFER and ANTSENV in the terminal.

created by Daniel Haenelt
Date created: 02-05-2019
Last modified: 04-05-2019
"""
import os
import shutil as sh
from nipype.interfaces.ants import N4BiasFieldCorrection
from nighres.registration import embedded_antsreg, apply_coordinate_mappings
from lib.registration.mask_ana import mask_ana
from lib.registration.mask_epi import mask_epi

# input data
file_mean_epi = "/nobackup/actinium2/haenelt/VasoTest/flicker/diagnosis/mean_bold_basis.nii"
file_t1 = "/nobackup/actinium2/haenelt/VasoTest/anatomy/T1_0p7.nii"
file_mask = "/nobackup/actinium2/haenelt/VasoTest/anatomy/skull/skullstrip_mask.nii"
path_output = "/nobackup/actinium2/haenelt/VasoTest/deformation/flicker"
cleanup = True

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

""" do not edit below """

"""
set folder structure
"""
path_temp = os.path.join(path_output,"temp")
path_epi = os.path.join(path_temp,"epi")
path_t1 = os.path.join(path_temp,"t1")

if not os.path.exists(path_output):
    os.makedirs(path_output)

if not os.path.exists(path_temp):
    os.makedirs(path_temp)

if not os.path.exists(path_epi):
    os.makedirs(path_epi)

if not os.path.exists(path_t1):
    os.makedirs(path_t1)

# copy input files
sh.copyfile(file_mean_epi, os.path.join(path_epi,"epi.nii"))
sh.copyfile(file_t1, os.path.join(path_t1,"T1.nii"))
sh.copyfile(file_mask, os.path.join(path_t1,"mask.nii"))

"""
bias field correction to epi
"""
n4 = N4BiasFieldCorrection()
n4.inputs.dimension = 3
n4.inputs.input_image = os.path.join(path_epi,"epi.nii")
n4.inputs.bias_image = os.path.join(path_epi,'n4bias.nii')
n4.inputs.output_image = os.path.join(path_epi,"bepi.nii")
n4.run()

"""
mask t1 and epi
"""
mask_ana(os.path.join(path_t1,"T1.nii"),os.path.join(path_t1,"mask.nii"))
mask_epi(os.path.join(path_epi,"bepi.nii"), 
         os.path.join(path_t1,"pT1.nii"), 
         os.path.join(path_t1,"mask.nii"), 
         niter_mask, sigma_mask)

"""
syn
"""
embedded_antsreg(os.path.join(path_t1,"pT1.nii"), # source image
                 os.path.join(path_epi,"pbepi.nii"), # target image 
                 run_rigid, # whether or not to run a rigid registration first 
                 rigid_iterations, # number of iterations in the rigid step
                 run_affine, # whether or not to run an affine registration first
                 affine_iterations, # number of iterations in the affine step
                 run_syn, # whether or not to run a SyN registration
                 coarse_iterations, # number of iterations at the coarse level
                 medium_iterations, # number of iterations at the medium level
                 fine_iterations, # number of iterations at the fine level
                 cost_function, # CrossCorrelation or MutualInformation
                 interpolation, # interpolation for registration result (NeareastNeighbor or Linear)
                 convergence = 1e-6, # threshold for convergence (can make algorithm very slow)
                 ignore_affine = False, # ignore the affine matrix information extracted from the image header 
                 ignore_header = False, # ignore the orientation information and affine matrix information extracted from the image header
                 save_data = True, # save output data to file
                 overwrite = True, # overwrite existing results 
                 output_dir = path_output, # output directory
                 file_name = "syn", # output basename
                 )

# rename final deformations
os.rename(os.path.join(path_output,"syn_ants-map.nii.gz"),
          os.path.join(path_output,"ana2epi.nii.gz"))
os.rename(os.path.join(path_output,"syn_ants-invmap.nii.gz"),
          os.path.join(path_output,"epi2ana.nii.gz"))

"""
apply deformation
"""
# ana -> epi
apply_coordinate_mappings(file_t1, # input 
                          os.path.join(path_output,"ana2epi.nii.gz"), # cmap
                          interpolation = "linear", # nearest or linear
                          padding = "zero", # closest, zero or max
                          save_data = True, # save output data to file (boolean)
                          overwrite = True, # overwrite existing results (boolean)
                          output_dir = path_output, # output directory
                          file_name = "ana2epi_example" # base name with file extension for output
                          )

# epi -> ana
apply_coordinate_mappings(file_mean_epi, # input 
                          os.path.join(path_output,"epi2ana.nii.gz"), # cmap
                          interpolation = "linear", # nearest or linear
                          padding = "zero", # closest, zero or max
                          save_data = True, # save output data to file (boolean)
                          overwrite = True, # overwrite existing results (boolean)
                          output_dir = path_output, # output directory
                          file_name = "epi2ana_example" # base name with file extension for output
                          )

# rename final deformation examples
os.rename(os.path.join(path_output,"ana2epi_example_def-img.nii.gz"),
          os.path.join(path_output,"ana2epi_example.nii.gz"))
os.rename(os.path.join(path_output,"epi2ana_example_def-img.nii.gz"),
          os.path.join(path_output,"epi2ana_example.nii.gz"))

# clean intermediate files
if cleanup:
    sh.rmtree(path_temp, ignore_errors=True)