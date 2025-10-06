import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from PIL import Image
from torchvision import transforms
import lpips

import dnnlib
class NvidiaVGG16:
    def __init__(self, path='/home/yanzhang/diffuerslearning/customp2p/metrics/vgg16.pt', device='cuda'):
        # 直接从本地加载 JIT-traced 模型
        self.model = torch.jit.load(path).eval().to(device)

    def __call__(self, img):
        img = (img + 1) / 2 * 255  # 归一化到 [0, 255]
        return self.model(img, resize_images=False, return_lpips=True)


def perc(target: torch.tensor, image: torch.tensor, vgg: torch.nn.Module, downsampling: bool):
    # Downsample image to 224x224 if needed (VGG's expected resolution)
    if image.shape[2] > 224 and downsampling:
        image = F.interpolate(image, size=(224, 224), mode='area')
    if target.shape[2] > 224 and downsampling:
        target = F.interpolate(target, size=(224, 224), mode='area')

    # Features for synth images
    image_features = vgg(image)
    target_features = vgg(target)

    diff = (image_features - target_features).square()
    return diff.sum().mean()


# 初始化 VGG
VGG = NvidiaVGG16()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 改为 VGG 的标准输入尺寸
    transforms.ToTensor(),  # 转换为 [0, 1] 范围的 Tensor
])


def calculate_lpips(src_path, ref_path, tar_path):
    # 加载并处理图像
    src = transform(Image.open(src_path).convert('RGB')).unsqueeze(0)
    ref = transform(Image.open(ref_path).convert('RGB')).unsqueeze(0)
    tar = transform(Image.open(tar_path).convert('RGB')).unsqueeze(0)

    # 移至 GPU（如果可用）
    if torch.cuda.is_available():
        src = src.cuda()
        ref = ref.cuda()
        tar = tar.cuda()

    # 计算 LPIPS
    lpips_src = perc(tar, src, vgg=VGG, downsampling=True)
    lpips_ref = perc(tar, ref, vgg=VGG, downsampling=True)

    return lpips_src, lpips_ref


if __name__ == "__main__":
    src_paths = ['/home/yanzhang/dataset/customp2p/src/mirror.jpg',
    '/home/yanzhang/dataset/customp2p/src/strawberrymug.jpg',
    '/home/yanzhang/dataset/customp2p/src/sushi.jpg',
    '/home/yanzhang/dataset/customp2p/src/woman.jpg']

    ref_path = '/home/yanzhang/dataset/customp2p/ref/cat2/03.jpg'

    tgt_paths = ['../baseline/attention/cat.png',
    '../baseline/attention/cup.jpg',
    '../baseline/attention/dog.png',
    '../baseline/attention/woman.jpg']

    # tgt_paths = ['../baseline/swapanything/cat.png',
    # '../baseline/swapanything/cup.png',
    # '../baseline/swapanything/dog.png',
    # '../baseline/swapanything/woman.png']

    for i in range(len(src_paths)):
        src_path = src_paths[i]
        tar_path = tgt_paths[i]
        lpips_src, lpips_ref = calculate_lpips(src_path, ref_path, tar_path)
        print(f"LPIPS with source: {lpips_src.item():.4f}")
        #print(f"LPIPS with reference: {lpips_ref.item():.4f}")

    
