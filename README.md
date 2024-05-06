# PWFHarmonization  ![Static Badge](https://img.shields.io/badge/pytorch-16878C?style=flat)

Official PyTorch implementation of Harmonizing Three-Dimensional MRI using Pseudo-Warping Field Guided GAN.

## Requirements

```
- torch==1.12.0+cu113
- torchvision==0.9.0a0
- python==3.8.8
- numpy==1.20.1
- nibabel==5.0.1
```

## Data Preparation

The data utilized in this article can be accessed from: 

[Alzheimer’s Disease Neuroimaging Initiative (ADNI)](adni.loni.usc.edu).

[Nathan Kline Institute Rockland sample (NKI-RS)](https://fcon1000.projects.nitrc.org/indi/enhanced/). 

[UK Biobank](https://www.ukbiobank.ac.uk/).

## Data Structure

The model is trained to harmonize data between trainA and trainB. Please ensure that all scans are properly coregistered and free from artifacts. Scans in trainA and trainB are sliced into .png format.

```
./datasets
 	---- train
 	|	---- trainA
 	|	|	---- trainName_0.png
 	|	|	---- trainName_1.png
	|	|	---- …
	|	---- trainB
	|	|	---- …
 	---- test
 	|	---- testA
	| 	|	---- testName_0.png
	|	|	---- testName_1.png
	|	|	---- …
	|	---- testB
```



## Training & Inference

Training

```
python train.py --dataroot ./datasets --model pwfharmonization --dataset_mode ADNI --input_nc 1 --output_nc 1 --batchSize 9 --resize_mode square --loadSize 256 --loadSizeH 256 --loadSizeW 256 --fineSize 256 --fineSizeH 256 --fineSizeW 256 --no_dropout --name your_experiment_name --which_model_netG resnet_6blocks
```

Inference

```
python test.py --dataroot ./datasets/ --model pwfharmonization --dataset_mode ADNI --input_nc 1 --output_nc 1 --name your_experiment_name --loadSize 256 --resize_mode square --fineSize 256 --crop_mode square --which_model_netG resnet_6blocks --which_epoch latest --no_dropout
```

