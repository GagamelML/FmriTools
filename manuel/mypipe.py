# general
import os

# nighres
from nighres.segmentation import fuzzy_cmeans
from echo_tools import calc_echotimes

# personal
from transform_classification import transform_classification
from block_editing_tools import epi_average_images, epi_average_blocks
from layer_tools import grow_layers

path = '/data/p_02808/Sessions/2023-04-03_S01_34522.15/'
out_dir = os.path.join(path, 'mypipe/')

t1w_transformed = '/data/p_02808/Sessions/2023-04-03_S01_34522.15/nighres/mp2rage2epi-step2_ants-def0.nii.gz'

epi_MTon_11 = ['/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS18_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS20_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS22_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS24_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS26_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS28_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS30_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS32_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii'
               ]
epi_MTon_27 = ['/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS18_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii,'
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS20_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii,'
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS22_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii,'
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS24_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii,'
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS26_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii,'
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS28_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii,'
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS30_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii,'
               '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS32_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii'
               ]
epi_MToff_11 = ['/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS19_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS21_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS23_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS25_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS27_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS29_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS31_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS33_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii'
                ]
epi_MToff_27 = ['/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS19_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS21_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS23_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS25_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS27_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS29_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS31_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii,'
                '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/epi/uS33_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii'
                ]

mask = '/data/p_02808/Sessions/2023-04-03_S01_34522.15/imagedata/S01_handknop_mask.nii.gz'
epi_names = ['epi_MTon_11', 'epi_MTon_27', 'epi_MToff_11', 'epi_MToff_27', 'epi_MTon_00', 'epi_MTon_00']
epi_lists = [epi_MTon_11, epi_MTon_27, epi_MToff_11, epi_MToff_27]


# no edits below this line

#calc TE0
epi_MTon_TE00 = []
epi_MToff_TE00 = []
for i in range(len(epi_MTon_11)):
    this_epi_MTon_00 = calc_echotimes(epi_MTon_11[i], 11.8, epi_MTon_27[i], 27.8, 'calc-epi_MTon_TE00_num' + str(i))
    epi_MTon_TE00.append(this_epi_MTon_00['S0_map'])
epi_lists.append(epi_MTon_TE00)
for i in range(len(epi_MToff_11)):
    this_epi_MToff_00 = calc_echotimes(epi_MToff_11[i], 11.8, epi_MToff_27[i], 27.8, 'calc-epi_MToff_TE00_num' + str(i))
    epi_MToff_TE00.append(this_epi_MToff_00['S0_map'])
epi_lists.append(epi_MToff_TE00)

epi_combined_series = []
a = len(epi_lists)
for k in range(len(epi_lists)):
    combined = epi_average_images(epi_lists[k],
                                  out_name=epi_names[k] + '_combined-series',
                                  out_dir=os.path.join(out_dir, 'epi-series-average/'))
    epi_combined_series.append(combined)

epi_combined_blocks = []
for k in range(len(epi_combined_series)):
    combined = epi_average_blocks(epi_combined_series[k], blocks=[6, 6], skip_first=True,
                                  out_dir=os.path.join(out_dir, 'epi-block-average'))
    epi_combined_blocks.append(combined)

rcfm = fuzzy_cmeans(t1w_transformed, clusters=3,
                    max_iterations=50, max_difference=0.01,
                    smoothing=0.1, fuzziness=2.0, mask_zero=True, map_intensity=False,
                    save_data=True, overwrite=False, output_dir=out_dir,
                    file_name='tissues')

tissue_class_transformed_masked = transform_classification(rcfm['classification'],
                                                           out_name='tissue_class_transformed_masked',
                                                           mask=mask,
                                                           out_dir=out_dir)

num_layers = 10
layers = grow_layers(tissue_class_transformed_masked, num_layers, out_name='layers_N' + str(num_layers))



showme