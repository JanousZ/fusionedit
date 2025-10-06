import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from PIL import Image
from torchvision import transforms


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean(), ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1), ssim_map


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


src_path = '/home/yanzhang/dataset/customp2p/src/mirror.jpg'
ref_path = '/home/yanzhang/dataset/customp2p/ref/cat2/03.jpg'
tar_path = '../baseline/VCT/cat.png'

src = Image.open(src_path).convert('RGB')  # 保证图片是RGB格式
ref = Image.open(ref_path).convert('RGB')
tar = Image.open(tar_path).convert('RGB')

# 定义转换操作：将图片转换为Tensor并做归一化
transform = transforms.Compose([
    transforms.Resize((512,512)),  # 根据需要调整尺寸 改为512
    transforms.ToTensor(),  # 转换为 [0, 1] 范围的 Tensor
])

# 转换图像
src = transform(src).unsqueeze(0)  # 增加一个维度，变为 [1, C, H, W]
ref = transform(ref).unsqueeze(0)
tar = transform(tar).unsqueeze(0)


ssim_value, ssim_map = ssim(src, tar)

# 输出计算的 SSIM 值
print(f"SSIM Value for src: {ssim_value.item()}")  # 输出 SSIM 均值
print(f"SSIM Map Shape: {ssim_map.shape}")  # 输出 SSIM 图的形状

ssim_value, ssim_map = ssim(ref, tar)
print(f"SSIM Value for ref: {ssim_value.item()}")  # 输出 SSIM 均值
print(f"SSIM Map Shape: {ssim_map.shape}")  # 输出 SSIM 图的形状
