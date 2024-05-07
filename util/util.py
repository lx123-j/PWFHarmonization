from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import matplotlib.pyplot as plt
import collections
import torch.nn.functional as F
import pylab
import random
import cv2
from torch.autograd import Variable

# wrapping 操作 即生成下一帧 创建文件夹保存图像
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 转换的操作 即生成下一帧   将输入图像和光流做双线性插值得到新的一帧图像
def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()       # 将网格扩散到和输入图像一样大小
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid - flo                       # 计算偏移之后的网格坐标

    # Scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0           #  将网格坐标归一化到【-1，1】
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear')       # 进行图像变换操作  F.grid_sample 对输入图像进行重采样
    return output

# 消融实验结果
# def warp(x, padding_mode='border'):
#     B, C, H, W = x.size()
#
#     # Mesh grid
#     xx = torch.arange(0, W).view(1,-1).repeat(H,1)
#     # x3=np.array(xx)
#     yy = torch.arange(0, H).view(-1,1).repeat(1,W)
#     xx = xx.view(1,1,H,W).repeat(B,1,1,1)
#     yy = yy.view(1,1,H,W).repeat(B,1,1,1)
#     grid = torch.cat((xx,yy),1).float()       # 将网格扩散到和输入图像一样大小
#     if x.is_cuda:
#         grid = grid.cuda()
#     vgrid = grid                  # 计算偏移之后的网格坐标
#
#
#     # Scale grid to [-1,1]
#     vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0           #  将网格坐标归一化到【-1，1】
#     vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
#     vgrid = vgrid.permute(0,2,3,1)
#     # vgrid = np.array(vgrid)
#     #
#     # v2 = vgrid * 255
#     # v2 = Image.fromarray(v2[0,0], mode='L')
#     # v2.save("/opt/data/private/BlindHarmony-main/Dataset/v1.jpeg")
#     output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='bilinear')       # 进行图像变换操作  F.grid_sample 对输入图像进行重采样
#     # x1=np.array(x)
#     # x1=x1*255
#     # x2=Image.fromarray(np.squeeze(x1),mode='L')
#     #
#     # x2.save("/opt/data/private/BlindHarmony-main/Dataset/x1.jpeg")
#
#     # img=np.array(output)
#     #
#     # img=img*255
#     # img2 = np.squeeze(img)
#     #
#     # cv2.imwrite("/opt/data/private/BlindHarmony-main/Dataset/img.jpeg",img2)
#     return output



def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array, from [-1, 1] to [0, 255], with the shape from (c, h, w) to (h, w, c)
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))    # 变成三通道
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0   # 将x第二维度挪到第一维上，第三维移到第二维上，原本的第一维移动到第三维上
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)     # 实现array到image的转换
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


# 创建文件夹保存模型和采样输出视频
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
