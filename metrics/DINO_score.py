import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.nn import functional as F
import os
from dinov2 import hubconf
import torch.nn as nn

# class FrozenDinoV2Encoder(AbstractEncoder):
#     """
#     Uses the DINOv2 encoder for image
#     """
#     def __init__(self, device="cuda", freeze=True):
#         super().__init__()
#         dinov2 = hubconf.dinov2_vitg14() 
#         state_dict = torch.load(DINOv2_weight_path)
#         dinov2.load_state_dict(state_dict, strict=False)
#         self.model = dinov2.to(device)
#         self.device = device
#         if freeze:
#             self.freeze()
#         self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#         self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
#         self.projector = nn.Linear(1536,1024)

#     def freeze(self):
#         self.model.eval()
#         for param in self.model.parameters():
#             param.requires_grad = False

#     def forward(self, image):
#         if isinstance(image,list):
#             image = torch.cat(image,0)

#         image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
#         features = self.model.forward_features(image)
#         tokens = features["x_norm_patchtokens"]
#         image_features  = features["x_norm_clstoken"]
#         image_features = image_features.unsqueeze(1)
#         hint = torch.cat([image_features,tokens],1) # 8,257,1024
#         hint = self.projector(hint)
#         return hint

#     def encode(self, image):
#         return self(image)

class DINOScoreCalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._load_dino_model()
        self.transform = self._get_transform()
        # 添加投影层
        self.projector = nn.Linear(1536, 1024).to(device)
        # 添加标准化参数
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        self.image_std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
    
    
    def _load_dino_model(self):
        dinov2 = hubconf.dinov2_vitg14() # Dreambooth原文用的是DINO VITS16 这里用的Anydoor的DINOv2 VITG14
        state_dict = torch.load("/home/yanzhang/diffuerslearning/customp2p/metrics/dinov2_vitg14_pretrain.pth")
        dinov2.load_state_dict(state_dict, strict=False)
        dinov2.eval()
        return dinov2.to(self.device)
    
    def _get_transform(self):
        """定义图像预处理流程"""
        return transforms.Compose([
            transforms.Resize(518),  # DINOv2推荐尺寸
            transforms.CenterCrop(518),
            transforms.ToTensor(),
        ])
    
    def extract_features(self, image_path):
        """提取单张图像的DINO特征"""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        # 标准化
        image = (image - self.image_mean) / self.image_std
        
        # with torch.no_grad():
        #     features = self.model(image)
        #     # 归一化特征
        #     features = F.normalize(features, p=2, dim=1)
        
        # return features.cpu().numpy()
        with torch.no_grad():
            # 提取特征
            features = self.model.forward_features(image)
            tokens = features["x_norm_patchtokens"]
            image_features = features["x_norm_clstoken"]
            image_features = image_features.unsqueeze(1)
            # 合并特征
            hint = torch.cat([image_features, tokens], 1)
            # 投影到1024维度
            hint = self.projector(hint)
        
        return hint.cpu().numpy()
    
    def compute_similarity(self, features1, features2):
        """计算两组特征之间的余弦相似度"""
        # 确保特征形状正确
        f1 = features1.reshape(-1)
        f2 = features2.reshape(-1)
        # # 计算余弦相似度
        # similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        # return similarity
        similarity = F.cosine_similarity(torch.tensor(f1), torch.tensor(f2), dim=0)
        return similarity.item()  # 转换为标量
    
    def calculate_dino_score(self, real_image_path, generated_image_path):
        """计算两张图片之间的DINO分数"""
        real_features = self.extract_features(real_image_path)
        gen_features = self.extract_features(generated_image_path)
        similarity = self.compute_similarity(real_features, gen_features)
        return similarity
    
    def calculate_batch_dino_score(self, real_images_dir, generated_images_dir):
        """计算两个目录中图片的平均DINO分数"""
        similarities = []
        real_images = sorted(os.listdir(real_images_dir))
        gen_images = sorted(os.listdir(generated_images_dir))
        
        for real_img, gen_img in zip(real_images, gen_images):
            real_path = os.path.join(real_images_dir, real_img)
            gen_path = os.path.join(generated_images_dir, gen_img)
            score = self.calculate_dino_score(real_path, gen_path)
            similarities.append(score)
        
        return np.mean(similarities)


if __name__ == "__main__":

    calculator = DINOScoreCalculator()

    src_paths = ['/home/yanzhang/dataset/customp2p/src/mirror.jpg',
    '/home/yanzhang/dataset/customp2p/src/strawberrymug.jpg',
    '/home/yanzhang/dataset/customp2p/src/sushi.jpg',
    '/home/yanzhang/dataset/customp2p/src/woman.jpg']

    ref_path = '/home/yanzhang/dataset/customp2p/ref/cat2/03.jpg'

    # tgt_paths = ['../baseline/attention/cat.png',
    # '../baseline/attention/cup.jpg',
    # '../baseline/attention/dog.png',
    # '../baseline/attention/woman.jpg']

    tgt_paths = ['../baseline/swapanything/cat.png',
    '../baseline/swapanything/cup.png',
    '../baseline/swapanything/dog.png',
    '../baseline/swapanything/woman.png']

    for i in range(len(src_paths)):
        src_path = src_paths[i]
        tar_path = tgt_paths[i]
        score = calculator.calculate_dino_score(src_path, tar_path)
        print(f"Single image DINO score: {score:.4f}")
    
    # # 批量计算
    # real_dir = "path/to/real/images"
    # generated_dir = "path/to/generated/images"
    # batch_score = calculator.calculate_batch_dino_score(real_dir, generated_dir)
    # print(f"Average DINO score: {batch_score:.4f}")






