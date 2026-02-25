import argparse
import logging
import math
import os
import yaml
from pathlib import Path

import accelerate
import torch
import torch.utils.data
import torch.utils.checkpoint
import torch.nn.functional as F
import utils
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    ControlNetModel
)
from models.I2P import IPFMs, T5TextEmbedder
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from datasets.dataset import ImageCaptionDataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.set_device(1)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__)


def sampling_train(vae, unet, scheduler, args, accelerator, noisy_latents, encoder_hidden_states, weight_dtype,
                   down_block_res_samples, mid_block_res_sample, num_inference_steps):
    with torch.no_grad():
        scheduler.set_timesteps(num_inference_steps)
        latents = noisy_latents

        for t in scheduler.timesteps:
            model_pred = unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = latents.detach()

        latents = latents.to(dtype=weight_dtype)
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        image_pred = vae.decode(latents / vae.config.scaling_factor, return_dict=False, generator=generator)[0]

    return image_pred


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='LLM_Diff Models')
    parser.add_argument("--config", default='./configs/config.yml', type=str,
                        help="Path to the config file")
    parser.add_argument("--output_dir", type=str, default="./output_models",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="The data directory where the datasets are used.")
    parser.add_argument("--train_dataset", type=str, default="lowlight", help="The name of the train dataset.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="all the images in the train/validation dataset will be resized to this.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--checkpointing_steps", type=int, default=100,
                        help="Save a checkpoint of the training state every X updates.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="LLM-Diff",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    args = parser.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
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


# def main(args):
args, config = parse_args_and_config()

logging_dir = Path(args.output_dir, args.logging_dir)

from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    kwargs_handlers=[ddp_kwargs]
)
config.device = accelerator.device

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    # transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()
else:
    # transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

# If passed along, set the training seed now.
if args.seed is not None:
    set_seed(args.seed)

# Handle the repository creation
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


# load LLM and IPFM
llm_model = T5TextEmbedder(config, pretrained_path="./pretrained/flan-t5-large")

ipfms = IPFMS(config)

# Load scheduler and models
noise_scheduler = DDPMScheduler.from_pretrained("./pretrained/stable-diffusion-2.1",
                                                subfolder="scheduler")

vae = AutoencoderKL.from_pretrained("./pretrained/stable-diffusion-2.1",
                                    subfolder="vae")
unet = UNet2DConditionModel.from_pretrained("./pretrained/stable-diffusion-2.1",
                                            subfolder="unet")

logger.info("Initializing controlnet weights from unet")
controlnet = ControlNetModel.from_unet(unet)

# `accelerate` 0.16.0 will have better support for customized saving
if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        assert len(models) == 2 and len(weights) == 2
        for i, model in enumerate(models):
            sub_dir = "controlnet" if isinstance(model, ControlNetModel) else "ipfms"
            model.save_pretrained(os.path.join(output_dir, sub_dir))
            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        assert len(models) == 2
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if not isinstance(model, ControlNetModel):
                load_model = IPFMs.from_pretrained(os.path.join(input_dir, "ipfms"))
            else:
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

            # model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


vae.requires_grad_(False)
unet.requires_grad_(False)
llm_model.requires_grad_(False)
controlnet.train()
ipfms.train()

if args.enable_xformers_memory_efficient_attention:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError("xformers is not available. Make sure it is installed correctly")

if args.gradient_checkpointing:
    controlnet.enable_gradient_checkpointing()

# Check that all trainable models are in full precision
low_precision_error_string = (
    " Please make sure to always have all model weights in full float32 precision when starting training - even if"
    " doing mixed precision training, copy of the weights should still be float32."
)

if accelerator.unwrap_model(controlnet).dtype != torch.float32:
    raise ValueError(
        f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    )
if accelerator.unwrap_model(ipfms).dtype != torch.float32:
    raise ValueError(
        f"IPFMs loaded as datatype {accelerator.unwrap_model(ipfms).dtype}. {low_precision_error_string}"
    )

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True

if args.scale_lr:
    args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    )

# Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
if args.use_8bit_adam:
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
        )

    optimizer_class = bnb.optim.AdamW8bit
else:
    optimizer_class = torch.optim.AdamW

# Optimizer creation
print(f'=================Optimize ControlNet and IPFMs ======================')
params_to_optimize = list(controlnet.parameters()) + list(ipfms.parameters())

print(f'start to load optimizer...')

optimizer = optimizer_class(
    params_to_optimize,
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

train_dataset = ImageCaptionDataset(dir=args.data_dir,
                                    filelist='{}_train.txt'.format(args.train_dataset))

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               num_workers=args.num_workers,
                                               batch_size=args.train_batch_size,
                                               shuffle=True,
                                               pin_memory=True)

# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    num_training_steps=args.max_train_steps * accelerator.num_processes,
    num_cycles=args.lr_num_cycles,
    power=args.lr_power,
)

# Prepare everything with our `accelerator`.
controlnet, ipfms, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    controlnet, ipfms, optimizer, train_dataloader, lr_scheduler
)

# For mixed precision training we cast the text_encoder and vae weights to half-precision
# as these models are only used for inference, keeping weights in full precision is not required.
weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
# print(weight_dtype)

# Move vae, unet and text_encoder to device and cast to weight_dtype
vae.to(accelerator.device, dtype=weight_dtype)
unet.to(accelerator.device, dtype=weight_dtype)
llm_model.to(accelerator.device, dtype=weight_dtype)
controlnet.to(accelerator.device, dtype=weight_dtype)
ipfms.to(accelerator.device, dtype=weight_dtype)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
if overrode_max_train_steps:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

# We need to initialize the trackers we use, and also store our configuration.
# The trackers initializes automatically on the main process.
if accelerator.is_main_process:
    tracker_config = dict(vars(args))
    accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

# Train!
total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

logger.info("***** Running training *****")

logger.info(f"  Num examples = {len(train_dataset)}")
logger.info(f"  Num batches each epoch = {len(train_dataloader)}")

logger.info(f"  Num Epochs = {args.num_train_epochs}")
logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps = {args.max_train_steps}")
global_step = 0
first_epoch = 0

# Potentially load in the weights and states from a previous save
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        initial_global_step = 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
else:
    initial_global_step = 0

progress_bar = tqdm(
    range(0, args.max_train_steps),
    initial=initial_global_step,
    desc="Steps",
    # Only show the progress bar once on each machine.
    disable=not accelerator.is_local_main_process,
)

for epoch in range(first_epoch, args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(controlnet) and accelerator.accumulate(ipfms):
            # print(type(batch))
            image = batch["image"].to(accelerator.device, dtype=weight_dtype)
            # Convert images to latent space
            latents = vae.encode(image).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # # Get the text embedding for conditioning
            text_embedding = llm_model(batch["caption"])
            encoder_hidden_states = ipfms(text_embedding, timesteps).to(accelerator.device, dtype=weight_dtype)

            controlnet_image = batch["condition_image"].to(accelerator.device, dtype=weight_dtype)

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # Predict the noise residual
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            # image_pred = sampling_train(vae, unet, noise_scheduler, args, accelerator, noisy_latents,
            #                             encoder_hidden_states, weight_dtype, down_block_res_samples,
            #                             mid_block_res_sample, 50).to(accelerator.device)
            #
            # noise_loss, color_loss = noise_estimation_loss(model_pred, noise, image_pred)
            # loss = noise_loss + color_loss  # + clip_loss
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                accelerator.clip_grad_norm_(ipfms.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=args.set_grads_to_none)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

# Create the pipeline using using the trained modules and save it.
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    controlnet = accelerator.unwrap_model(controlnet)
    controlnet.save_pretrained(args.output_dir)

    ipfms = accelerator.unwrap_model(ipfms)
    ipfms.save_pretrained(args.output_dir)

accelerator.end_training()
