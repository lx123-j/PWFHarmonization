# # coding=utf-8
import os.path

import SimpleITK as sitk
import pydicom

def dcm2nii(dcms_path, nii_path):
    # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dcms_path)
    # 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    for i in range(len(series_IDs)):

        reader = sitk.ImageSeriesReader()
        dicom_names_all = reader.GetGDCMSeriesFileNames(dcms_path, series_IDs[i])
        for dicom_name in dicom_names_all:
            ds = pydicom.dcmread(dicom_name)

            # 检查DICOM标签中的序列类型信息
            # if 'T2W' in ds.SeriesDescription.upper() or 'BTFE' in ds.SeriesDescription.upper():

            reader.SetFileNames(dicom_names_all)
            image2 = reader.Execute()
            # 2.将整合后的数据转为array，并获取dicom文件基本信息
            image_array = sitk.GetArrayFromImage(image2)  # z, y, x
            origin = image2.GetOrigin()  # x, y, z
            spacing = image2.GetSpacing()  # x, y, z
            direction = image2.GetDirection()  # x, y, z
            # 3.将array转为img，并保存为.nii.gz
            image3 = sitk.GetImageFromArray(image_array)
            image3.SetSpacing(spacing)
            image3.SetDirection(direction)
            image3.SetOrigin(origin)
            sitk.WriteImage(image3, os.path.join(nii_path, os.path.basename(dicom_names_all[0]).split('_')[0] + '.nii.gz'))


if __name__ == '__main__':
     root_path=r'dicom path'
     for path in os.listdir(root_path):
         for dicom_path in os.listdir(os.path.join(root_path,path)):


            nii_path = r'save path'  # 所需.nii.gz文件保存路径
            save_path=os.path.join(nii_path,os.path.basename(path))
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            dcm2nii(os.path.join(root_path, path, dicom_path), save_path)

