"""
注意力控制器的定义和实现部分
"""
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline,DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner as seq_aligner
from ddim_inversion import DDIM_Inversion
from einops import rearrange, repeat
from PIL import Image
import time
import json
import os
import argparse
from DIFT import SDFeaturizer, DIFTLatentStore, register_DIFT, gen_dift_map_dict, visualize_dift
import torch.nn as nn
import math
import datetime
import itertools
import BeLM

#以下所有维度标注仅在图像为512*512的情况下，如果是不同分辨率则是等倍扩大/缩小

class LocalBlend:    
    def __call__(self, x_t, attention_store, t):
        """
        blend x_t
        """
        k = 1
        
        if self.mask is None:
            cross_maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            cross_maps = [item.reshape(len(prompts), -1, 16, 16, item.shape[-1]) for item in cross_maps]    #[b, heads, 16, 16, 77] * layers
            cross_maps = torch.cat(cross_maps, dim=1)
            cross_maps = cross_maps.sum(1) / cross_maps.shape[1]    #[b,16,16,77]
            
            #cross_maps = cross_maps[:,:,:,2].unsqueeze(1)   #[b,1,16,16]

            cross_maps = cross_maps * self.alpha_layers.squeeze(1).squeeze(1) #[b,16,16,77]
            cross_maps = cross_maps.sum(-1).unsqueeze(1)   #[b,1,16,16]

            cross_maps = nnf.interpolate(cross_maps, size=(x_t.shape[2:]), mode="bilinear").squeeze(1)  #[b,64,64]
            cross_maps = cross_maps / cross_maps.max(dim=2, keepdims=True).values.max(dim=1, keepdims=True).values
            cross_maps = cross_maps.gt(0.25)    #best 0.25
            mask = cross_maps.float().unsqueeze(1)
        else:
            mask = self.mask.float()

        x_t[1:2] = x_t[:1] + mask[0:1,...] * (x_t[1:2] - x_t[:1])

        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3, tokenizer=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold
        self.mask = None
    
    def set_mask(self, mask):
        """
        mask size:[64,64],type 0 or 1
        """
        self.mask = mask  

class AttentionControl(abc.ABC):
    #假设有batchsize = 2, heads = 8, 并且在uncond以及text共同堆叠推断下
    #对于一个样本的8个注意力头是连续的，并且对应于一个uncond或者text
    #此外一般是所有的uncond都位于text前，且按照样本顺序排列
    #[0-7,...]属于样本1 uncond  [8-15,...]属于样本2 uncond  
    #[16-23,...]属于样本1 text  [24-31,...]属于样本2 text
    #crossattn[batchsize * heads * 2, h * w, len_text]
    #selfattn[batchsize * heads * 2, h * w, h * w]
    
    #action will do to x_t-1 after go through the whole unet for one timestep and update x_t
    def step_callback(self, x_t, t):
        return x_t
    
    #action will do to the controller for managing attention storage
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        H = W = int(np.sqrt(q.shape[1]))
        b = q.shape[0] // num_heads    #只有q的batch是b，kv的batch都是1
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)   #[batch * heads, len, dim]  -->  [heads, batch * len, dim]   [8,b*len,d]  len为分辨率
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)   #[8,len,d]
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)   #[8,len,d]

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")    #[8, qlen, klen]

        if kwargs.get("mode") == 'mask':
            #mask可能是导致质量低的原因
            mask = self.get_ref_mask(ref_mask, mask_weight=args.mask_weight, H=H, W=W).to(device)   #[1,1,512,512]
            mask = mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim[..., H*W:] = sim[..., H*W:] + mask

            #new add
            mask = self.get_src_mask(src_mask, H=H, W=W).to(device)   #[res*res] 0背景 1前景
            foreground_index, background_index = torch.where(mask == 1)[0], torch.where(mask == 0)[0]
            sim[:, background_index, H*W:] = torch.finfo(sim.dtype).min
            #sim[:, foreground_index, :H*W] = torch.finfo(sim.dtype).min

        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)    #[8, b*len, d]
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out
    
    @abc.abstractmethod
    def cross_forward(self, attn, is_cross: bool, place_in_unet: str):
        """
        return new cross attn
        """
        raise NotImplementedError
    
    def self_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        q [16b or 8b, h*w, dim_q]   16b = 2(cond&uncond) * 8(num_heads) * b
        k [16b or 8b, h*w, dim_k]
        v [16b or 8b, h*w, dim_k]
        """
        if not self.LOW_RESOURCE:
            qu, qc = q.chunk(2)   #[8 * b, h*w, dim_q]
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)

            out_u_src = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, None, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_src = self.attn_batch(qc[:num_heads], kc[:num_heads], vc[:num_heads], None, None, is_cross, place_in_unet, num_heads, **kwargs)
            out_u_ref = self.attn_batch(qu[2*num_heads:], ku[2*num_heads:], vu[2*num_heads:], None, None, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_ref = self.attn_batch(qc[2*num_heads:], kc[2*num_heads:], vc[2*num_heads:], None, None, is_cross, place_in_unet, num_heads, **kwargs)

            if self.cur_step < args.self_src_q_time_end and self.cur_att_layer // 2 in self.self_src_q_layer_idx:
            #q_tgt = q_src  k_tgt = k_src 保持整体结构
                qu[num_heads: 2*num_heads] = qu[:num_heads]
                qc[num_heads: 2*num_heads] = qc[:num_heads]
                ku[num_heads: 2*num_heads] = ku[:num_heads]
                kc[num_heads: 2*num_heads] = kc[:num_heads]

            #最好来个时间限制
            res = int(math.sqrt(q.shape[1]))
            if self.dift_map_dict is not None:
                if self.cur_step < args.self_ref_q_time_end and self.cur_step >= args.self_ref_q_time_start and self.cur_att_layer // 2 in args.self_ref_q_layer_idx:
                    mask = self.get_src_mask(src_mask, H=res, W=res).to(device)   #[res*res] 0背景 1前景
                    foreground_index, background_index = torch.where(mask == 1)[0], torch.where(mask == 0)[0]
                    dift_map = self.dift_map_dict[f"{res}"] #[h*w]
                    q_mix_scale = 1.0
                    qu[num_heads: 2*num_heads, foreground_index, :] = qu[2*num_heads:, dift_map[foreground_index], :] * q_mix_scale + qu[num_heads:2*num_heads, foreground_index, :] * (1 - q_mix_scale)
                    qc[num_heads: 2*num_heads, foreground_index, :] = qc[2*num_heads:, dift_map[foreground_index], :] * q_mix_scale + qc[num_heads:2*num_heads, foreground_index, :] * (1 - q_mix_scale)

            if self.cur_step < args.self_ref_kv_time_end and self.cur_att_layer // 2 in self.self_ref_kv_layer_idx:
                #q_tgt [k_tgt,k_ref] [v_tgt,v_ref] 
                out_u_tgt = self.attn_batch(qu[num_heads:2*num_heads], ku[num_heads:], vu[num_heads:],None, None, is_cross, place_in_unet, num_heads, mode="mask", **kwargs)
                out_c_tgt = self.attn_batch(qc[num_heads:2*num_heads], kc[num_heads:], vc[num_heads:],None, None, is_cross, place_in_unet, num_heads, mode="mask", **kwargs)

                #q_tgt k_ref v_ref 
                # out_u_tgt = self.attn_batch(qu[num_heads:2*num_heads], ku[2*num_heads:], vu[2*num_heads:],None, None, is_cross, place_in_unet, num_heads, **kwargs)
                # out_c_tgt = self.attn_batch(qc[num_heads:2*num_heads], kc[2*num_heads:], vc[2*num_heads:],None, None, is_cross, place_in_unet, num_heads, **kwargs)
            else:
                #q_tgt k_tgt v_tgt
                out_u_tgt = self.attn_batch(qu[num_heads:2*num_heads], ku[num_heads:2*num_heads], vu[num_heads:2*num_heads], None, None, is_cross, place_in_unet, num_heads, **kwargs)
                out_c_tgt = self.attn_batch(qc[num_heads:2*num_heads], kc[num_heads:2*num_heads], vc[num_heads:2*num_heads], None, None, is_cross, place_in_unet, num_heads, **kwargs)
            
            out = torch.cat([out_u_src, out_u_tgt, out_u_ref, out_c_src, out_c_tgt, out_c_ref],dim=0)
            return out
            
        else:
            pass
        
        return out

    def batch_to_head_dim(self, tensor: torch.Tensor, head_size) -> torch.Tensor:
        """
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def __call__(self, q, k, v, sim, attn, is_cross: bool, place_in_unet: str, num_heads, **kwargs):    
        if is_cross:
            #cross_attn_control, only activate when calculate cond_noise, deactivate when calculate uncond_noise
            if self.cur_att_layer >= self.num_uncond_att_layers:
                if self.LOW_RESOURCE:
                    attn = self.cross_forward(attn, is_cross, place_in_unet)
                    out = torch.einsum("b i j, b j d -> b i d", attn, v)
                    out = self.batch_to_head_dim(out, num_heads) 
                else:
                    h = attn.shape[0]
                    attn[h // 2:] = self.cross_forward(attn[h // 2:], is_cross, place_in_unet)
                    out = torch.einsum("b i j, b j d -> b i d", attn, v)
                    out = self.batch_to_head_dim(out, num_heads)
        else:
            #self_attn_control, activate when calculate both cond_noise and uncond_noise
            out = self.self_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        print(self.cur_att_layer, "cross" if is_cross else "self", out.shape)

        self.cur_att_layer += 1    #record that finish one attn layer

        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1     #record that finish one timestep
            self.between_steps()   #attn accumulation
        return out

    def get_ref_mask(self, ref_mask, mask_weight, H, W):
        ref_mask = nnf.interpolate(ref_mask, (H, W))
        ref_mask = (ref_mask > 0).float() * mask_weight
        ref_mask = ref_mask.flatten()
        return ref_mask

    def get_src_mask(self, src_mask, H, W):
        src_mask = nnf.interpolate(src_mask, (H, W))
        src_mask = (src_mask > 0).float().flatten()
        return src_mask
    
    def set_dift_map_dict(self, dift_map_dict = None):
        self.dift_map_dict = dift_map_dict

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
    
    def set_enable(self, enable):
        self.enable = enable

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1    #Unet中注册的的CrossAttention层总数
        self.cur_att_layer = 0
        self.LOW_RESOURCE = LOW_RESOURCE
        self.self_src_q_layer_idx = list(range(0, 16))  
        self.self_ref_kv_layer_idx = list(range(0, 16))
        self.cross_src_layer_idx = list(range(0, 16))
        self.dift_map_dict = None
        self.enable = True

class EmptyControl(AttentionControl):
    
    def cross_forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def cross_forward(self, attn, is_cross: bool, place_in_unet: str):    
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        """
        accumulate attn from step store to attention_store after finish the last attn layer in Unet in one timestep
        """
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        """
        return attn average by timestep for each layer
        """
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()    
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
     
class AttentionControlEdit(AttentionStore, abc.ABC): 
    
    def step_callback(self, x_t, t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, t)
        return x_t
        
    # def replace_self_attention(self, attn_base, att_replace):
    #     if att_replace.shape[2] <= 16 ** 2:
    #         return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
    #     else:      
    #         return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def cross_forward(self, attn, is_cross: bool, place_in_unet: str):    
        super(AttentionControlEdit, self).cross_forward(attn, is_cross, place_in_unet)
        if is_cross: 
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])     #[b, heads, h*w, len_words]
            attn_base, attn_repalce = attn[0], attn[1:2]
            #[timestep+1, b-1, len_words], 提取当前timestep的alpha_words, alpha_words值为1时，就对crossattn进行修改；alpha_words值为0时，不改动；              
            alpha_words = self.cross_replace_alpha[self.cur_step]
            alpha_words = alpha_words[0:1]    
            attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
            attn[1:2] = attn_repalce_new
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 tokenizer):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)    #[timesteps + 1, b - 1, 1, 1, words]
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None):
    #[b-1, word_inds, word_inds]
    #[a rabbit cake],[a cat cake]
    #mapper:
    #[1,0,0]
    #[0,1,0]
    #[0,0,1]
    #mapper的第i列代表edit prompt的第i个词，由srcprompt的attnmap线性组合，线性比例为第i列的值去乘以相对应的列
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)     
        
class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer = None):
        #[a cake], [a birthday cake]
        #mapper = [0, 1, -1, 2, 3, 4, ...]
        #alphas = [1, 1, 0,  1, 1, 1, ...]
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        #equalizer [1,1,3,1,1,1,1,,....]
        #有一个词是3倍
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]], tokenizer = None):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    """
    aggregate_attention for certain resolution layer based average, timestep-averaged, selected prompt
    """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, tokenizer = None):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    return ptp_utils.view_images(np.stack(images, axis=0))
    
def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))

def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, ref_latent=None, ref_uncond_embeddings=None, seed=3407, data_inds=None):

    print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, 
                                                  generator=generator, low_resource=LOW_RESOURCE, uncond_embeddings=uncond_embeddings, ref_latent=ref_latent, 
                                                  ref_uncond_embeddings=ref_uncond_embeddings, randomzt=False)
    Image.fromarray(images[1]).save(os.path.join(output_dir,f"{data_inds}.jpg"))
    images = ptp_utils.view_images(images)
    images.save(os.path.join(output_dir,f"compare_{data_inds}.jpg"))

    # controller.reset()
    # images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, 
    #                                               generator=generator, low_resource=LOW_RESOURCE, uncond_embeddings=uncond_embeddings, ref_latent=ref_latent, 
    #                                               ref_uncond_embeddings=ref_uncond_embeddings, randomzt=True)
    # Image.fromarray(images[1]).save(os.path.join(output_dir,"P2P_tgt_randomzt.png"))
    # images = ptp_utils.view_images(images)
    # images.save(os.path.join(output_dir,"P2P_randomzt.png"))
    
    return images, x_t

def load_mask(mask_path, H=512, W=512):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask) / 255.0   #convert to 0 or 1, 1是前景 0是背景
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
    mask = nnf.interpolate(mask, size=(H,W), mode='nearest')
    mask = (mask > 0).float()
    return mask

if __name__ == "__main__":
    #超参数设置
    parser = argparse.ArgumentParser(description="hyper parameters")

    parser.add_argument("--expid", required=True, type=int)
    parser.add_argument("--self_ref_kv_time_end", type=int, default="50")
    parser.add_argument("--self_ref_kv_layer_idx", type=int, nargs=2, default=[0, 16])
    parser.add_argument("--self_ref_q_time_start", type=int, default="0")
    parser.add_argument("--self_ref_q_time_end", type=int, default="50")
    parser.add_argument("--self_ref_q_layer_idx", type=int, nargs=2, default=[0, 16])
    parser.add_argument("--self_src_q_time_end", type=int, default="50")
    parser.add_argument("--self_src_q_layer_idx", type=int, nargs=2, default=[0, 16])
    parser.add_argument("--mask_weight", type=float, default="3.0")
    parser.add_argument("--cross_step", type=float, default="0.6")
    parser.add_argument("--cross_src_layer_idx", type=int, nargs=2, default=[0, 16])
    parser.add_argument("--topk", action="store_true")
    args = parser.parse_args()

    output_dir = f"./results/{datetime.date.today()}/{args.expid}"
    os.makedirs(output_dir, exist_ok=True)
    # 保存为 JSON 文件
    param_save_path = os.path.join(output_dir, f"exp_{args.expid}_config.json")
    with open(param_save_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    #model loading and config
    LOW_RESOURCE = False 
    NUM_DIFFUSION_STEPS = 50
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained("/mnt/disk1/fengyanzhang/stable-diffusion-v1-5", scheduler=scheduler).to(device)
    seed = int(time.time())
    g_cpu = torch.Generator().manual_seed(seed)
    
    dataset_path = "/home/yanzhang/dataset/customp2p"
    json_path = "prompt.json"
    
    json_path = os.path.join(dataset_path,json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        json_datas = json.load(f)

    #注意每过一次unet DIFT就会被覆盖
    dift_latent_store = DIFTLatentStore()
    register_DIFT(unet=ldm_stable.unet, dift_latent_store=dift_latent_store)

    #load images & prompts & masks

    for i in range(len(json_datas)):
        src_image_path = os.path.join(dataset_path, json_datas[i]["src_image"])
        ref_image_path = os.path.join(dataset_path, json_datas[i]["ref_image"])
        ref_mask_path = os.path.join(dataset_path, json_datas[i]["ref_mask"])
        src_mask_path = os.path.join(dataset_path, json_datas[i]["src_mask"])
        src_prompt = json_datas[i]["src_prompt"]
        ref_prompt = json_datas[i]["ref_prompt"]
        tgt_prompt = json_datas[i]["tgt_prompt"]
        prompts = [src_prompt, tgt_prompt, ref_prompt]
        ref_mask = load_mask(ref_mask_path)   #[1,1,512,512]
        src_mask = load_mask(src_mask_path)

        # get dift features & gen dift map
        print("calculating dift features......")
        sdfeaturizer = SDFeaturizer(model=ldm_stable)
        dift_latent_store.reset()
        sdfeaturizer.forward(src_image_path, prompt=src_prompt, ensemble_size=8, t=451, up_ft_index=1, model=ldm_stable) 
        src_dift = dift_latent_store.dift_features["1"].mean(0, keepdim=True)   # 1, c, 32, 32
        dift_latent_store.reset()
        sdfeaturizer.forward(ref_image_path, prompt=ref_prompt, ensemble_size=8, t=451, up_ft_index=1, model=ldm_stable)
        ref_dift = dift_latent_store.dift_features["1"].mean(0, keepdim=True)   # 1, c, 32, 32
        dift_map_dict, _ = gen_dift_map_dict(src_dift, ref_dift, ref_mask=ref_mask, use_topk=args.topk)
        print("finishing dift features mapping!")

        visualize_dift(dift_map_dict, src_mask, ref_mask, ref_image_path)

        #src image and ref image inversion
        print("inverting src image and ref image......")
        
        ddim_inversion = DDIM_Inversion(ldm_stable, NUM_DDIM_STEPS=NUM_DIFFUSION_STEPS)
        _, ddim_latents = ddim_inversion.invert(src_image_path, prompts[0])
        uncond_embeddings = ddim_inversion.null_optimization(ddim_latents, num_inner_steps=10, epsilon=1e-5)
        
        _, ref_latents = ddim_inversion.invert(ref_image_path, prompts[2])
        ref_uncond_embeddings = ddim_inversion.null_optimization(ref_latents, 10, 1e-5)
            
        print("finishing src image and ref image inversion!")

        lb_words = json_datas[i]["local_blend"]
        lb = LocalBlend(prompts, lb_words, tokenizer=ldm_stable.tokenizer)
        lb.set_mask(load_mask(src_mask_path, H=64, W=64).to("cuda"))
        controller = AttentionReplace(prompts, NUM_DIFFUSION_STEPS,
                                    cross_replace_steps={"default_": 1.0, lb_words[1]: args.cross_step},   #数值越大，覆盖越多，数值越小，保留越多
                                    self_replace_steps=0.2,   #已弃用，等价self_src_q_time_end
                                    local_blend=lb,
                                    tokenizer=ldm_stable.tokenizer
                                    )
        controller.set_dift_map_dict(dift_map_dict=dift_map_dict)

        #register controller and run
        
        run_and_display(prompts, controller, latent=ddim_latents[-1], run_baseline=True, uncond_embeddings=uncond_embeddings, 
                        ref_latent=ref_latents[-1], ref_uncond_embeddings=ref_uncond_embeddings, seed=seed, data_inds=i)

        # show_cross_attention(controller, res=16, from_where=("up", "down"), tokenizer=ldm_stable.tokenizer, select=0).save(os.path.join(output_dir,"show_cross_attn_0.png"))
        # show_cross_attention(controller, res=16, from_where=("up", "down"), tokenizer=ldm_stable.tokenizer, select=1).save(os.path.join(output_dir,"show_cross_attn_1.png"))
        # show_cross_attention(controller, res=16, from_where=("up", "down"), tokenizer=ldm_stable.tokenizer, select=2).save(os.path.join(output_dir,"show_cross_attn_2.png"))

        controller.enable = False

    # controller = AttentionStore()
    # image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)
    # show_cross_attention(controller, res=16, from_where=("up", "down"))

