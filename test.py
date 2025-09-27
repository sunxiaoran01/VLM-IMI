import os
import yaml
import argparse
import glob
from PIL import Image
import numpy as np
import torch
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor

from models.ipfm import IPFM, T5TextEmbedder


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/config.yml')
    parser.add_argument("--pretrained_model_path", type=str,
                        default='pretrained/stable-diffusion-2.1')
    parser.add_argument("--image_path", type=str, default='data/low')
    parser.add_argument("--caption_path", type=str, default='data/text')
    parser.add_argument("--output_dir", type=str, default='data/result')
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    args, config = parse_args_and_config()

    # step-1: settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.device = device

    # Step-2: instantiate models and schedulers
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", revision=None).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_path, subfolder="unet", revision=None).to(device)
    controlnet = ControlNetModel.from_pretrained("output/checkpoint/controlnet").to(device)
    llm_model = T5TextEmbedder(config, 'pretrained/flan-t5-large').to(device)

    noise_scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae_image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_convert_rgb=True, do_normalize=True)

    ipfm = IPFM.from_pretrained("output/checkpoint/ipfm").to(device)

    vae.requires_grad_(False)
    llm_model.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)
    ipfm.requires_grad_(False)

    # Step-3: prepare data
    if os.path.isdir(args.image_path):
        image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
    else:
        image_names = [args.image_path]
    save_path = os.path.join(args.output_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_idx, image_name in enumerate(tqdm(image_names, desc="Processing images")):
        image = Image.open(image_name)
        pil_image = image.copy()

        base_name = os.path.basename(image_name)
        caption_name = os.path.join(args.caption_path, os.path.splitext(base_name)[0] + ".txt")

        if os.path.exists(caption_name):
            with open(caption_name, 'r', encoding='utf-8') as file:
                caption = file.read().strip()

        with torch.no_grad():
            # LLM and ipfm
            timesteps = torch.randint(0, 1000, (1,), device="cuda:0")
            timesteps = timesteps.long()

            text_embedding = llm_model(caption)
            prompt_embeds = ipfm(text_embedding, timesteps)

            width, height = image.size
            new_width = int(8 * np.ceil(width / 8.0))
            new_height = int(8 * np.ceil(height / 8.0))

            # pre-process image
            image = vae_image_processor.preprocess(image, height=new_height, width=new_width).to(
                device=vae.device)  # image now is tensor in [-1,1]
            b, c, h, w = image.size()

            # set/load random seed
            generator = torch.Generator()
            generator.manual_seed(args.seed)  # one can also adjust this seed to get different results

            # set the noise or latents
            latents = torch.randn((1, 4, h//vae_scale_factor, w//vae_scale_factor), generator=generator).cuda()

            # set the time step
            noise_scheduler.set_timesteps(args.num_inference_steps, device=vae.device)
            timesteps = noise_scheduler.timesteps
            timesteps = timesteps.long()

            # feedforward
            for i, t in enumerate(timesteps):
                down_block_res_samples, mid_block_res_sample = controlnet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=image,
                    return_dict=False,
                )

                # diffusion unet
                noise_pred = unet(latents,
                                  t,
                                  encoder_hidden_states=prompt_embeds,
                                  down_block_additional_residuals=down_block_res_samples,
                                  mid_block_additional_residual=mid_block_res_sample,
                                  ).sample

                # update the latents
                latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # post-process
            pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            pred = vae_image_processor.postprocess(pred, output_type='pil')[0]

            image_final = pred.resize((width, height), Image.BICUBIC)

        image_final.save(os.path.join(save_path, base_name))
    print('---------done-----------')
