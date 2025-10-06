from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.models import UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
import gc
import os
from PIL import Image
from torchvision.transforms import PILToTensor

class DIFTLatentStore:
    def __init__(self):
        self.dift_features = {}

    def __call__(self, features: torch.Tensor, t: int, layer_index: int):
        self.dift_features[f'{layer_index}'] = features

    def copy(self):
        copy_dift = DIFTLatentStore(self.up_ft_indices)

        for key, value in self.dift_features.items():
            copy_dift.dift_features[key] = value.clone()

        return copy_dift

    def reset(self):
        self.dift_features = {}

def register_DIFT(unet: UNet2DConditionModel, dift_latent_store: DIFTLatentStore):   
    
    def DIFTUNetforward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            deprecate(
                "T2I should not use down_block_additional_residuals",
                "1.3.0",
                "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                       and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                       for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
                standard_warn=False,
            )
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            #only changed code
            if dift_latent_store is not None:
                dift_latent_store(sample.clone().detach(), t=timestep, layer_index=i) 

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    unet.forward = DIFTUNetforward.__get__(unet, UNet2DConditionModel)
    print("register unet for dift extract!")
    
class SDFeaturizer:

    def __init__(self, null_prompt='', model=None):
        gc.collect()
        null_prompt_embeds = model._encode_prompt(
            prompt=null_prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=True)[0] # [1, 77, dim]
        self.null_prompt_embeds = null_prompt_embeds
        self.null_prompt = null_prompt

    @torch.no_grad()
    def forward(self,
                img_path,
                prompt='',
                t=261,    #加噪时间
                up_ft_index=1,    #提取的unet层index
                ensemble_size=8,
                model=None):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        img = Image.open(img_path).convert('RGB')
        img_size = 512
        img = img.resize((img_size, img_size))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        if prompt == self.null_prompt:
            prompt_embeds = self.null_prompt_embeds
        else:
            prompt_embeds = model._encode_prompt(
                prompt=prompt,
                device='cuda',
                num_images_per_prompt=1,
                do_classifier_free_guidance=True)[1] # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)

        device = model._execution_device
        latents = model.vae.encode(img_tensor).latent_dist.sample() * model.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = model.scheduler.add_noise(latents, noise, t)    #先加噪到t时刻
        unet_output = model.unet(latents_noisy,
                               t,
                               encoder_hidden_states=prompt_embeds)
        
def gen_dift_map_dict(src_dift, ref_dift, ref_mask=None, use_topk=False, topk = 16):
    """
    src_dift   [1,c,h,w]
    ref_dift   [1,c,h,w]
    ref_mask   [1,1,h,w]
    """
    dift_map_dict = {}
    dift_distance_dict = {}
    c = src_dift.shape[1]

    if ref_mask is not None:
        print("dift ref mask enabled!")
    else:
        print("dift ref mask disabled!")

    resolution = [8, 16, 32, 64]   #only for image [512,512]
    for res in resolution:
        resized_src_dift = F.interpolate(src_dift, size=(res, res), mode="bilinear")
        resized_src_dift = F.normalize(resized_src_dift, dim=1)  # [1, c, h, w]
        resized_src_dift = resized_src_dift.squeeze(0).permute(1, 2, 0).reshape(-1, c)  # [h*w, c]

        resized_ref_dift = F.interpolate(ref_dift, size=(res, res), mode="bilinear")
        resized_ref_dift = F.normalize(resized_ref_dift, dim=1)  # [1, c, h, w]
        resized_ref_dift = resized_ref_dift.squeeze(0).permute(1, 2, 0).reshape(-1, c)  # [h*w, c]

        dift_distance = torch.matmul(resized_src_dift, resized_ref_dift.T)  # [h*w, h*w]
        if ref_mask is not None:
            mask = ref_mask.clone().to("cuda")
            mask = F.interpolate(mask, size=(res, res), mode="nearest").squeeze(0).squeeze(0).reshape(-1)  #[h*w]
            fill_value = dift_distance.min()
            dift_distance = torch.where(mask == 1, dift_distance, fill_value)
        
        if res >= 32 and use_topk:
            topk_distance = dift_distance.clone().cpu().numpy()
            topk_index = np.argsort(topk_distance, axis = 1)[:,::-1][:,:topk]
            topk_distance = np.take_along_axis(topk_distance, topk_index, axis=1)
            topk_distance = topk_distance / np.sum(topk_distance, axis=1, keepdims=True)
            map_index_coords = np.stack(np.unravel_index ( np.arange(res * res), (res, res) ), axis=1)  # [h*w, 2]
            topk_coords = map_index_coords[topk_index]  # [h*w, topk, 2]
            dift_map = np.einsum('ijk,ij->ik', topk_coords, topk_distance)  # [h*w, 2]
            dift_map = np.round(dift_map).astype(int)
            dift_map = dift_map[:, 0] * res + dift_map[:, 1]
            dift_map = torch.tensor(dift_map).to(dift_distance.device)
        else:
            _ , dift_map = dift_distance.max(dim=1)  #[h*w] and element in (0,..,h*w-1)

        dift_map_dict[f"{res}"] = dift_map
        dift_distance_dict[f"{res}"] = dift_distance

    return dift_map_dict, dift_distance_dict

def visualize_dift(dift_map_dict, src_mask, ref_mask, ref_image_path, mode="color_all"):

    def is_in_part(x, y, part):
        real_part = (int(y) // 4) * 16 + (int(x) // 4)
        return real_part == part

    for res, dift_map in dift_map_dict.items():
        res = int(res)
        resized_src_mask = F.interpolate(src_mask, size=(res,res), mode="nearest").squeeze(0).squeeze(0)
        resized_src_mask = (resized_src_mask > 0).int()
        resized_src_mask = ((resized_src_mask) * 255).numpy().astype(np.uint8)
        resized_ref_mask = F.interpolate(ref_mask, size=(res,res), mode="nearest").squeeze(0).squeeze(0)
        resized_ref_mask = (resized_ref_mask > 0).int()
        resized_ref_mask = ((resized_ref_mask) * 255).numpy().astype(np.uint8)

        img_ref = np.array(Image.open(ref_image_path).resize((res, res)))
        img = Image.new('RGB', (res, res))
        pixels = img.load()
        img_map = Image.new('RGB', (res, res))
        pixels_map = img_map.load()
        part = 60   #0-255

        for i in range(res):
            for j in range(res):
                # 使用像素位置生成一个渐变颜色
                map_i, map_j = np.unravel_index(dift_map[(i - 1) * res + j].cpu().numpy(),  (res, res))
                angle = (i * res + j) / (res * res) * 1.7 * np.pi  # 计算每个像素点的角度
                r = int((np.sin(angle) + 1) * 127)  # 红色渐变
                g = int((np.sin(angle + 2 * np.pi / 3) + 1.3) * 127)  # 绿色渐变
                b = int((np.sin(angle + 4 * np.pi / 3) + 1) * 127)  # 蓝色渐变

                if resized_ref_mask[i,j] < 128:
                    pixels[j, i] = (0, 0, 0)
                elif "part" in mode:
                    pixels[j, i] = (255, 255, 255)
                elif "color" in mode:
                    pixels[j, i] = (r, g, b)
                elif "img" in mode:
                    pixels[j, i] = tuple(img_ref[i, j])

                angle = (map_i * res +  map_j) / (res * res) * 1.7 * np.pi  # 计算每个像素点的角度
                r = int((np.sin(angle) + 1) * 127)  # 红色渐变
                g = int((np.sin(angle + 2 * np.pi / 3) + 1.3) * 127)  # 绿色渐变
                b = int((np.sin(angle + 4 * np.pi / 3) + 1) * 127)  # 蓝色渐变

                if resized_src_mask[i,j] < 128:
                    pixels_map[j, i] = (0, 0, 0)
                elif "part" in mode and not is_in_part(i, j, part):
                    pixels_map[j, i] = (255, 255, 255)
                elif "color" in mode:
                    pixels_map[j, i] = (r, g, b)
                    pixels[map_j, map_i] = (r, g, b)
                elif "img" in mode:
                    pixels_map[j, i] = tuple(img_ref[map_i, map_j])
            
        if res >= 64:    
            img.save(f"dift_ud.png")
            img_map.save(f"dift_ud_map.png")
        
        img_ref = np.array(Image.open(ref_image_path).resize((res, res)))
        img = Image.new('RGB', (res, res))
        pixels = img.load()
        img_map = Image.new('RGB', (res, res))
        pixels_map = img_map.load()
        
        for i in range(res):
            for j in range(res):
                # 使用像素位置生成一个渐变颜色
                map_i, map_j = np.unravel_index(dift_map[(i - 1) * res + j].cpu().numpy(),  (res, res))
                angle = (j * res + i) / (res * res) * 1.7 * np.pi  # 计算每个像素点的角度
                r = int((np.sin(angle) + 1) * 127)  # 红色渐变
                g = int((np.sin(angle + 2 * np.pi / 3) + 1.3) * 127)  # 绿色渐变
                b = int((np.sin(angle + 4 * np.pi / 3) + 1) * 127)  # 蓝色渐变

                if resized_ref_mask[i,j] < 128:
                    pixels[j, i] = (0, 0, 0)
                elif "part" in mode:
                    pixels[j, i] = (255, 255, 255)
                elif "color" in mode:
                    pixels[j, i] = (r, g, b)
                elif "img" in mode:
                    pixels[j, i] = tuple(img_ref[i, j])

                angle = (map_j * res +  map_i) / (res * res) * 1.7 * np.pi  # 计算每个像素点的角度
                r = int((np.sin(angle) + 1) * 127)  # 红色渐变
                g = int((np.sin(angle + 2 * np.pi / 3) + 1.3) * 127)  # 绿色渐变
                b = int((np.sin(angle + 4 * np.pi / 3) + 1) * 127)  # 蓝色渐变

                if resized_src_mask[i,j] < 128:
                    pixels_map[j, i] = (0, 0, 0)
                elif "part" in mode and not is_in_part(i, j, part):
                    pixels_map[j, i] = (255, 255, 255)
                elif "color" in mode:
                    pixels_map[j, i] = (r, g, b)
                    pixels[map_j, map_i] = (r, g, b)
                elif "img" in mode:
                    pixels_map[j, i] = tuple(img_ref[map_i, map_j])

        if res >= 64:    
            img.save(f"dift_lr.png")
            img_map.save(f"dift_lr_map.png")

