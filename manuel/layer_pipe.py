import numpy as np
import os
import matplotlib.pyplot as plt

from block_editing_tools import snipp_blocks
from layer_tools import epi_layeravgs_vs_time


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

# input:
num_images = 4

out_dir = '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/mypipe'

epi = [
    [
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS30_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS32_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS34_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS43_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_11.8.nii'
    ],
    ['/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS30_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS32_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS34_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS43_kp_epi2d_v1c3_1p5_R4x2_pF_MTCon_TR60_TE11p8-27p8_27.8.nii'],
    ['/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS31_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS33_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS35_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS44_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_11.8.nii'],
    ['/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS31_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS33_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS35_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii',
     '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi/uS44_kp_epi2d_v1c3_1p5_R4x2_pF_MTCoff_TR60_TE11p8-27p8_27.8.nii']
]
epi_names = [
    "MTon_TE11",
    "MTon_TE27",
    "MToff_TE11",
    "MToff_TE27"
]

glms = ['/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/results/spmT/native/spmT_MTon_TE11_ME_tapping_baseline_Sess12.nii',
        '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/results/spmT/native/spmT_MTon_TE28_ME_tapping_baseline_Sess12.nii',
        '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/results/spmT/native/spmT_MToff_TE11_ME_tapping_baseline_Sess12.nii',
        '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/results/spmT/native/spmT_MToff_TE28_ME_tapping_baseline_Sess12.nii']

layers = '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/mypipe/layers_N10.nii'
num_layers = 10
baseline = (8,11)
#layer_blocks = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
layer_blocks = [(1, 3), (4, 6), (7, 9)]
mask = '/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/imagedata/epi_hanknop_mask.nii.gz'

colors = get_color_gradient('#FF0000', '#0000FF', num_layers)

# separate epi-blocks
blocks = []
for para in range(len(epi)):
    blocks_thispara = []
    for k in range(len(epi[para])):
        name = epi_names[para] + '_img' + str(k+1)
        dir = os.path.join(out_dir, 'block_img', epi_names[para])
        blocks_thispara += snipp_blocks(epi[para][k], (6, 6), skip_first=True, out_name=name, out_dir=dir)
    blocks.append(blocks_thispara)


# get data to plot
data = []
test = len(epi)
for para in range(len(epi)):
    data.append(epi_layeravgs_vs_time(blocks[para], layers, mask, num_layers=num_layers, baseline=baseline, layer_blocks=layer_blocks))

timelength = len(data[0][0])
times = np.arange(timelength) + 1
# generate plot
plt.figure(figsize=(13,9.5))
for k in range(len(epi_names)):
    plt.subplot(2, 2, k+1, xlabel='time', label=epi_names[k])
    for lb in range(len(layer_blocks)):
        means = []
        sems = []
        for t in range(timelength):
            means.append(data[k][lb][t]['mean'])
            sems.append(data[k][lb][t]['sem'])
        plt.errorbar(times, means, yerr=sems, capsize=4, ecolor=colors[lb], fmt='none')
        plt.plot(times, means, label=str(data[k][lb][0]["laminars"][0]) + '-' + str(data[k][lb][0]["laminars"][1]), c=colors[lb])



plt.legend(loc='upper left')
plt.show()
plt.savefig('/data/tu_mlohoff/ABC_MRI/2023-03-01_Sess12_40108.fb/mypipe/testfig3.jpg')
test