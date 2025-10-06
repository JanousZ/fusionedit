import torch
import open_clip
from PIL import Image
import numpy as np
from torch.nn import functional as F
import os
from torchvision import transforms
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CLIPIScoreCalculator:
    def __init__(self, model_path, arch="ViT-H-14", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._load_clip_model(model_path, arch)
        # 添加标准化参数
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
        self.image_std = torch.tensor([0.26862954, 0.26130258, 0.275777]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)

    def _load_clip_model(self, model_path, arch):
        """加载CLIP视觉编码器"""
        # 创建模型
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device('cpu'),
            pretrained=None  # 不下载预训练权重
        )

        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)

        # 只保留视觉部分
        del model.transformer
        model.visual.output_tokens = True
        model.visual.eval()

        # 添加投影层
        self.projector_token = nn.Linear(1280, 1024).to(self.device)
        self.projector_embed = nn.Linear(1024, 1024).to(self.device)

        return model.visual.to(self.device)

    def extract_features(self, image_path):
        """提取单张图像的CLIP特征"""
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        image = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])(image).unsqueeze(0).to(self.device)

        # 标准化
        image = (image - self.image_mean) / self.image_std

        with torch.no_grad():
            # 提取特征和tokens
            image_features, tokens = self.model(image)
            # 处理特征
            image_features = image_features.unsqueeze(1)
            image_features = self.projector_embed(image_features)
            tokens = self.projector_token(tokens)
            # 合并特征
            features = torch.cat([image_features, tokens], 1)

        return features.cpu().numpy()

    def compute_similarity(self, features1, features2):
        """计算两组特征之间的余弦相似度，返回百分比形式"""
        f1 = features1.reshape(-1)
        f2 = features2.reshape(-1)
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        return similarity * 100

    def calculate_clip_i_score(self, real_image_path, generated_image_path):
        """计算两张图片之间的CLIP-I分数"""
        real_features = self.extract_features(real_image_path)
        gen_features = self.extract_features(generated_image_path)
        similarity = self.compute_similarity(real_features, gen_features)
        return similarity

    def calculate_batch_clip_i_score(self, real_images_dir, generated_images_dir):
        """计算两个目录中图片的平均CLIP-I分数"""
        similarities = []
        real_images = sorted(os.listdir(real_images_dir))
        gen_images = sorted(os.listdir(generated_images_dir))

        for real_img, gen_img in zip(real_images, gen_images):
            real_path = os.path.join(real_images_dir, real_img)
            gen_path = os.path.join(generated_images_dir, gen_img)
            score = self.calculate_clip_i_score(real_path, gen_path)
            similarities.append(score)

        return np.mean(similarities)


class CLIPTScoreCalculator:
    LAYERS = [
        "last",
        "penultimate"
    ]

    def __init__(self, model_path, arch="ViT-H-14", device='cuda' if torch.cuda.is_available() else 'cpu',
                 max_length=77, layer="last"):
        self.device = device
        assert layer in self.LAYERS
        self.model = self._load_clip_model(model_path, arch)
        self.max_length = max_length
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def _load_clip_model(self, model_path, arch):
        """加载CLIP模型"""
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device('cpu'),
            pretrained=None
        )
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)

        del model.visual  # 删除视觉部分
        model.eval()
        return model.to(self.device)

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        """transformer前向传播"""
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode_text_with_transformer(self, text):
        """文本编码"""
        x = self.model.token_embedding(text)
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        # 取最后一个token的特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return x

    def extract_text_features(self, text):
        """提取文本特征"""
        tokens=open_clip.tokenize(text)
        with torch.no_grad():
            text_features = self.encode_text_with_transformer(tokens.to(self.device))
        return text_features

    def calculate_clip_t_score(self, image_features, prompt):
        """计算CLIP-T分数"""
        text_features = self.extract_text_features(prompt)
        # 计算余弦相似度
        image_features = torch.from_numpy(image_features).to(self.device)
         # 平均所有token的特征
        image_features = image_features.mean(dim=1)  # [B, 257, D] -> [B, D]
        similarity = torch.cosine_similarity(image_features, text_features)
        return similarity.item() * 100

    def calculate_batch_clip_t_score(self, image_features_list, prompts):
        """批量计算CLIP-T分数"""
        similarities = []
        assert len(image_features_list) == len(prompts), "特征和文本数量不匹配"

        for img_features, prompt in zip(image_features_list, prompts):
            score = self.calculate_clip_t_score(img_features, prompt)
            similarities.append(score)

        return np.mean(similarities)


if __name__ == "__main__":
    model_path = "./open_clip_pytorch_model.bin"

    #src_path = '/home/yanzhang/dataset/customp2p/src/mirror.jpg'

    src_paths = ['/home/yanzhang/dataset/customp2p/src/mirror.jpg',
    '/home/yanzhang/dataset/customp2p/src/strawberrymug.jpg',
    '/home/yanzhang/dataset/customp2p/src/sushi.jpg',
    '/home/yanzhang/dataset/customp2p/src/woman.jpg']

    ref_path = '/home/yanzhang/dataset/customp2p/ref/cat2/03.jpg'

    tgt_paths = ['../baseline/attention/cat.png',
    '../baseline/attention/cup.jpg',
    '../baseline/attention/dog.png',
    '../baseline/attention/woman.jpg']

    tgt_paths = ['../baseline/swapanything/cat.png',
    '../baseline/swapanything/cup.png',
    '../baseline/swapanything/dog.png',
    '../baseline/swapanything/woman.png']

    ref_mask_path = "/home/yanzhang/dataset/customp2p/ref/car/car2.png"
    tgt_mask_path = "/home/yanzhang/dataset/customp2p/src/car_mask.jpg"

    # ref_image = Image.open(ref_path).convert('RGBA')
    # ref_mask = Image.open(ref_mask_path).convert('L')
    # tgt_image = Image.open(tgt_path).convert('RGBA')
    # tgt_mask = Image.open(tgt_mask_path).convert('L').resize((512,512))

    # masked_ref_image = Image.composite(ref_image, Image.new("RGBA", ref_image.size, (0, 0, 0, 0)), ref_mask).convert("RGB")
    # masked_tgt_image = Image.composite(tgt_image, Image.new("RGBA", tgt_image.size, (0, 0, 0, 0)), tgt_mask).convert("RGB")
    # masked_ref_image.save('masked_ref_image.jpg')
    # masked_tgt_image.save('masked_tgt_image.jpg')

    # ref_path = "./masked_ref_image.jpg"
    # tgt_path = "./masked_tgt_image.jpg"

    icalculator = CLIPIScoreCalculator(model_path)
    for i in range(len(src_paths)):
        src_path = src_paths[i]
        tgt_path = tgt_paths[i]
        # 计算CLIP-I分数
        iscore = icalculator.calculate_clip_i_score(src_path, tgt_path)
        print(f"CLIP-I Score: {iscore:.2f}")

    # tcalculator = CLIPTScoreCalculator(model_path)
    # tgt_feature = icalculator.extract_features(tgt_path)
    # prompt = "a photo of a cat"
    # tscore = tcalculator.calculate_clip_t_score(tgt_feature, prompt)
    # print(f"CLIP-T Score: {tscore:.2f}")
