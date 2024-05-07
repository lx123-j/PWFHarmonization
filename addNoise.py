import os
import nibabel as nib
import numpy as np
import  random
import imageio

nii_path = r'/opt/data/private/MRI_harm/datasets/v2v_nii//'
save_path = r'"/opt/data/private/MRI_harm/datasets/v2v_nii/add_noise_nii//'


def read_niifile(niifilepath):  # 读取niifile文件
    img = nib.load(niifilepath)  # 下载niifile文件（其实是提取文件）
    img_affine = img.affine
    img_fdata = img.get_fdata()  # 获取niifile数据
    return img_fdata,img_affine

for i in os.listdir(nii_path):

    for j in os.listdir(nii_path+i):

            nii_data,nii_affine = read_niifile(nii_path+i+'//'+j)  # 调用上面的函数，获得数据
            num=np.sum(nii_data==0)

            for m in range(255):
                for n in range(255):
                    for k in range(255):
                        if nii_data[m][n][k] == 0:
                            nii_data[m][n][k] = random.random()


            # nii_zero=np.where(nii_data.any(nii_data==0),np.random.randn(1,np.sum(nii_data==0)),nii_data)

            j_spilt=j.split('.')[0]


            if not os.path.exists(save_path):
                os.makedirs(save_path)

            nib.Nifti1Image(nii_data, nii_affine).to_filename(save_path+j_spilt+'_noise.nii.gz')


