import torch
from torchvision import transforms
from PIL import Image
import ptp_utils
import os
import tqdm
import numpy as np

def intermediate_to_latent(sd_pipe, controller, sd_params, src_intermediate=None, src_intermediate_second = None, 
                           ref_intermediate = None, ref_intermediate_second = None, freeze_step = 0, output_dir="./"):
    controller.reset()
    ptp_utils.register_attention_control(sd_pipe, controller)    #将Unet中所有的CrossAttention层forward绑定一个controller

    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    seed = sd_params['seed']
    guidance_scale = sd_params['guidance_scale']
    num_inference_steps = sd_params['num_inference_steps']

    uncond_input = sd_pipe.tokenizer(
            [""] * 3, padding="max_length", max_length=sd_pipe.tokenizer.model_max_length,
            return_tensors="pt"
        )
    uncond_embeddings = sd_pipe.text_encoder(uncond_input.input_ids.to(sd_pipe.device))[0]
    text_input = sd_pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=sd_pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = sd_pipe.text_encoder(text_input.input_ids.to(sd_pipe.device))[0]
    prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])

    torch.manual_seed(seed)
    sd_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = sd_pipe.scheduler.timesteps

    xis = []
    do_classifier_free_guidance = guidance_scale > 1.0

    xis.append(torch.cat([src_intermediate, src_intermediate, ref_intermediate], dim=0))
    intermediate = torch.cat([src_intermediate, src_intermediate, ref_intermediate], dim=0)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            print('###', i)
            if i < freeze_step:
                continue
            latent_model_input = torch.cat([intermediate] * 2) if do_classifier_free_guidance else intermediate
            noise_pred = sd_pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if i < num_inference_steps - 1:
                alpha_s = sd_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = sd_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5


            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            if i == freeze_step:
                print('have intermediate_second')
                intermediate = torch.cat([src_intermediate_second, src_intermediate_second, ref_intermediate_second], dim=0)
            else:
                # calculate i-1
                alpha_p = sd_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5

                # calculate t
                t_p, t_t, t_s = sigma_p / alpha_p, sigma_t / alpha_t, sigma_s / alpha_s

                # calculate delta
                delta_1 = t_t - t_p
                delta_2 = t_s - t_t
                delta_3 = t_s - t_p

                # calculate coef
                coef_1 = delta_2 * delta_3 * alpha_s / delta_1
                coef_2 = (delta_2/delta_1)**2*(alpha_s/alpha_p)
                coef_3 = (delta_1 - delta_2)*delta_3/(delta_1**2)*(alpha_s / alpha_t)

                # iterate
                intermediate = coef_1 * noise_pred + coef_2 * xis[-2] + coef_3 * xis[-1]
            xis.append(intermediate)

    images = latent2image(sd_pipe.vae, xis[-1])
    Image.fromarray(images[1]).save(os.path.join(output_dir,"P2P_tgt.png"))
    images = ptp_utils.view_images(images)
    images.save(os.path.join(output_dir,"P2P.png")) 

def latent_to_intermediate(sd_pipe, sd_params, latent=None, freeze_step = 0):

    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    seed = sd_params['seed']
    guidance_scale = sd_params['guidance_scale']
    num_inference_steps = sd_params['num_inference_steps']
    dtype = torch.float32

    uncond_input = sd_pipe.tokenizer(
            [""], padding="max_length", max_length=sd_pipe.tokenizer.model_max_length,
            return_tensors="pt"
        )
    uncond_embeddings = sd_pipe.text_encoder(uncond_input.input_ids.to(sd_pipe.device))[0]
    text_input = sd_pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=sd_pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = sd_pipe.text_encoder(text_input.input_ids.to(sd_pipe.device))[0]
    prompt_embeds = torch.cat([uncond_embeddings, text_embeddings])

    torch.manual_seed(seed)
    sd_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
    timesteps = sd_pipe.scheduler.timesteps

    xis = []
    do_classifier_free_guidance = guidance_scale > 1.0
    if latent is None:
        shape = (1, 4, 64, 64)
        latent = torch.randn(shape, generator=None, device='cuda', dtype=dtype)
        print('latents are None')

    xis.append(latent)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if i >= num_inference_steps - freeze_step:
                continue
            #print('###', i)
            index = num_inference_steps - i - 1
            time = timesteps[index + 1] if index < num_inference_steps - 1 else 1
            latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
            noise_pred = sd_pipe.unet(
                latent_model_input,
                time,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if index < num_inference_steps - 1:
                alpha_i = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = sd_pipe.scheduler.alphas_cumprod[timesteps[index + 1]].to(torch.float32)
            else:
                alpha_i = sd_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = 1

            sigma_i = (1 - alpha_i)**0.5
            sigma_i_minus_1 = (1 - alpha_i_minus_1)**0.5
            alpha_i = alpha_i**0.5
            alpha_i_minus_1 = alpha_i_minus_1**0.5


            if i == 0:
                latent = (alpha_i/alpha_i_minus_1)*latent+(sigma_i-(alpha_i/alpha_i_minus_1)*sigma_i_minus_1) * noise_pred
            else:
                alpha_i_minus_2 = 1 if i == 1 else sd_pipe.scheduler.alphas_cumprod[timesteps[index + 2]].to(torch.float32)
                sigma_i_minus_2 = (1 - alpha_i_minus_2) ** 0.5
                alpha_i_minus_2 = alpha_i_minus_2 ** 0.5

                h_i = sigma_i/alpha_i - sigma_i_minus_1/alpha_i_minus_1
                h_i_minus_1 = sigma_i_minus_1/alpha_i_minus_1 - sigma_i_minus_2/alpha_i_minus_2

                coef_x_i_minus_2 = (alpha_i/alpha_i_minus_2)*(h_i**2)/(h_i_minus_1**2)
                coef_x_i_minus_1 = (alpha_i/alpha_i_minus_1)*(h_i_minus_1**2 - h_i**2)/(h_i_minus_1**2)
                coef_eps = alpha_i*(h_i_minus_1 + h_i)*h_i/h_i_minus_1
                latent = coef_x_i_minus_2 * xis[-2] + coef_x_i_minus_1 * xis[-1] + coef_eps * noise_pred
            xis.append(latent)

    return xis[-1], xis[-2]

def pil_to_latents(image_path, sd_pipe):
    pil_image = Image.open(image_path).convert('RGB').resize((512, 512))
    image_tensor = transforms.Compose([transforms.PILToTensor()])(pil_image).to('cuda')
    # print(image_tensor.shape)
    if image_tensor.shape[0] == 3:
        pass
    elif image_tensor.shape[0] == 1:
        image_tensor = image_tensor.repeat(3, 1, 1)
    else:
        raise ValueError("")
    # print('11',image_tensor)
    image_tensor = image_tensor / 255.0
    image_tensor = (image_tensor - 0.5) / 0.5
    # print('image_tensor.shape = ',image_tensor.shape)
    with torch.no_grad():
        latents = sd_pipe.vae.encode(image_tensor.unsqueeze(0), return_dict=False)[0].sample()
        # print('latent.shape = ',latents.shape)
    latents = latents * sd_pipe.vae.config.scaling_factor
    return latents
    
@torch.no_grad()
def latent2image(vae, latents):     #latent code->image
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image