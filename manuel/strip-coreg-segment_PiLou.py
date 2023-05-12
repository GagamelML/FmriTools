# C-2022 Pierre-Louise Bazin, who kindly provided me with this script

import nighres
import os
import nibabel as nib

# working directories

in_dir = '/data/tu_mlohoff/ABC_MRI/2023-04-03_S01_34522.15/nighres'
out_dir = '//data/tu_mlohoff/ABC_MRI/2023-04-03_S01_34522.15/nighres/results'

# the input MP2RAGE data (native space)

uni = '/data/tu_mlohoff/ABC_MRI/2023-04-03_S01_34522.15/nighres/S4_mp2rage_0p7_iPAT2_phPF68_FA5-3_TI900-2750_BW250_UNI_Images_2.01.nii'
inv2 = '/data/tu_mlohoff/ABC_MRI/2023-04-03_S01_34522.15/nighres/S5_mp2rage_0p7_iPAT2_phPF68_FA5-3_TI900-2750_BW250_INV2_2.01.nii'

# the input EPI data (after alignment etc)

epi = '/data/tu_mlohoff/ABC_MRI/2023-04-03_S01_34522.15/nighres/meanS18_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii'


# Skull stripping
# ----------------
# First we perform skull stripping. Only the second inversion image is required
# to calculate the brain mask. But if we input the T1map and T1w image as well,
# they will be masked for us. We also save the outputs in the ``out_dir``
# specified above and use a subject ID as the base file_name.
strip = nighres.brain.mp2rage_skullstripping(second_inversion=inv2,
                                            t1_weighted=uni,
                                            save_data=True, overwrite=False,
                                            file_name='mp2rage',
                                            output_dir=out_dir)

# ANTs co-registration
# ------------------------
# We coregister non-linearly the stripped T1w to EPI twice, hopefully that works
# (note that there's a few things we can tweak if not)
ants1 = nighres.registration.embedded_antsreg(
                        source_image=strip['t1w_masked'],
                        target_image=epi,
                        run_rigid=True, run_affine=True, run_syn=True,
                        rigid_iterations=1000,
                        affine_iterations=500,
                        coarse_iterations=180,
                        medium_iterations=60, fine_iterations=30,
                        cost_function='MutualInformation',
                        interpolation='NearestNeighbor',
                        regularization='High',
                        ignore_affine=False,
                        save_data=True, overwrite=False, file_name="mp2rage2epi-step1",
                        output_dir=out_dir)

# Co-registration to an atlas works better in two steps
ants2 = nighres.registration.embedded_antsreg(
                        source_image=ants1['transformed_source'],
                        target_image=epi,
                        run_rigid=True, run_affine=True, run_syn=True,
                        rigid_iterations=1000,
                        affine_iterations=500,
                        coarse_iterations=180,
                        medium_iterations=60, fine_iterations=30,
                        cost_function='MutualInformation',
                        interpolation='NearestNeighbor',
                        regularization='High',
                        ignore_affine=False,
                        save_data=True, overwrite=False, file_name="mp2rage2epi-step2",
                        output_dir=out_dir)

# Basic tissue segmentation GM/WM/CSF
# ---------------------------------------
rfcm = nighres.segmentation.fuzzy_cmeans(strip['t1w_masked'], clusters=3,
                    max_iterations=50, max_difference=0.01,
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=False,
                    save_data=True, overwrite=False, output_dir=out_dir,
                    file_name='tissues')

# Transfer to EPI space
# -------------------------
rfcm2epi = nighres.registration.apply_coordinate_mappings(
                        image=rfcm['classification'],
                        mapping1=ants1['mapping'],
                        mapping2=ants2['mapping'],
                        interpolation='nearest',
                        save_data=True, overwrite=False, file_name="rfcm2epi",
                        output_dir=out_dir)



