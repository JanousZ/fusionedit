import torch
import open_clip
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEncoder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for image
    """

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", freeze=True):
        super().__init__()
        model, _, preprocess= open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.transformer
        self.model = model
        self.model.visual.output_tokens = True
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.26862954, 0.26130258, 0.275777]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.projector_token = nn.Linear(1280,1024)
        self.projector_embed = nn.Linear(1024,1024)

    def freeze(self):
        self.model.visual.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image,list):
            image = torch.cat(image,0)
        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        image_features, tokens = self.model.visual(image)
        image_features = image_features.unsqueeze(1)
        image_features = self.projector_embed(image_features)
        tokens = self.projector_token(tokens)
        hint = torch.cat([image_features,tokens],1)
        return hint

    def encode(self, image):
        return self(image)