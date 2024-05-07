import os
import numpy as np
import nibabel as nib
import imageio
from PIL import Image


# 把nii文件转为图片 注释里的分别是y和Z轴
def read_niifile(niifilepath):  # 读取niifile文件
    img = nib.load(niifilepath)  # 下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()
    img_fdata=np.array(img_fdata)# 获取niifile数据
    return img_fdata

# 第一个位置是sagittal 矢状面, 第二个位置是coronal 冠状面, 第三个面是Transverse 横断面
def save_fig(niifilepath, savepath,name):  # 保存为图片
    fdata = read_niifile(niifilepath)  # 调用上面的函数，获得数据
    nii_image = nib.Nifti1Image(fdata, np.eye(4))
    (x, y, z) = fdata.shape  # 获得数据shape信息：（长，宽，维度-切片数量，第四维）
    for k in range(x):
        silce = fdata[k ,:, :]  # 三个位置表示三个不同角度的切片
        if not os.path.exists(os.path.join(savepath,name)):
            os.makedirs(os.path.join(savepath,name))
        imageio.imwrite(os.path.join(savepath,name,name+'_{}.png'.format(k)), silce)
        sagittal_slices = Image.open(os.path.join(savepath,name,name+'_{}.png'.format(k))).convert('L')
        sagittal_slices = np.array(sagittal_slices)

if __name__ == '__main__':

    img_path_B = r'nii data path'

    save_path_B = r'save data path'

    for j in os.listdir(img_path_B):
        name=j.split('.')[0]

        if not os.path.exists(save_path_B):
            os.makedirs(save_path_B)
        save_fig(img_path_B+j, save_path_B,name)


