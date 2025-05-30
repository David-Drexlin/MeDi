#!/usr/bin/env python3
import pandas as pd
import argparse
import json
import gc
from sample import generate_images_for_fid
from torch_fidelity import calculate_metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
import numpy as np
import os
from datetime import timedelta
import traceback
import wandb
from torchvision.transforms import ToPILImage

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, WeightedRandomSampler
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torchvision import transforms
from tqdm.auto import tqdm

from load_TCGA import load_metadata, load_dataset, custom_collate_fn
from unet import UNet2DModel

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model (UNet).")
    parser.add_argument('--optimization_steps', type=int, required=True, help="Number of optimization steps")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate for the optimizer")
    parser.add_argument('--resolution', type=int, required=True, help="Dimensionality of the images")
    parser.add_argument('--FID_tracker', type=int, default=50000000, help="Compute FID every X steps")
    parser.add_argument('--cond_type', type=str, default='linear',
                        choices=['additive', 'concat', 'linear'], help="Method to condition the model")
    parser.add_argument('--use_wandb', action='store_true', help="Whether to log to wandb")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save outputs. If not provided, a name based on args will be generated.")
    parser.add_argument('--holdout_mask', type=str, nargs='+', default=None,
                        help="Path to the holdout mask file")
    parser.add_argument('--domains_to_condition', type=str, nargs='+', default=None,
                        help="Names of metadata columns to condition on")
    parser.add_argument('--cancer_types', type=str, nargs='+', default=None,
                        help="Cancer types to train on")
    parser.add_argument('--use_ema', action='store_true',
                        help="Whether to use Exponential Moving Average for model weights.")
    parser.add_argument('--ema_inv_gamma', type=float, default=1.0, help="Inverse gamma for EMA decay.")
    parser.add_argument('--ema_power', type=float, default=3/4, help="Power value for EMA decay.")
    parser.add_argument('--ema_max_decay', type=float, default=0.9999, help="Max decay for EMA.")
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'],
                        help="Use mixed precision")
    parser.add_argument('--checkpointing_steps', type=int, default=100000, help="Save checkpoint every X steps")
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='Learning rate scheduler type')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Steps to accumulate gradients before updating")
    parser.add_argument(
        '--data_root',
        type=str,
        default='./TCGA',
        help="Path to your TCGA image directory (defaults to ./TCGA)"
    )
    return parser.parse_args()


def save_noise_scheduler_config(noise_scheduler, output_dir):
    scheduler_config = noise_scheduler.config
    scheduler_save_path = os.path.join(output_dir, 'scheduler_config.json')
    with open(scheduler_save_path, 'w') as f:
        json.dump(scheduler_config, f, indent=4)
    logger.info(f"Noise scheduler config saved to {scheduler_save_path}")


def main():
    args = parse_args()

    # Generate output directory name if not provided
    if args.output_dir is None:
        device_count = torch.cuda.device_count() or 1
        args.output_dir = (
            f"model_bs{args.batch_size*device_count}"  
            f"_lr{args.learning_rate}"  
            f"_res{args.resolution}"  
            f"_{args.cond_type}"
        )
        if args.cancer_types:
            args.output_dir += "_" + "_".join(sorted(args.cancer_types))
        args.output_dir = os.path.join("models", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb:
        wandb.login(key=os.getenv('WANDB_API_KEY'))

    # Initialize logging and accelerator
    data_root = args.data_root
    logging_dir = args.output_dir
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with='wandb' if args.use_wandb else None,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )
    logger.info(accelerator.state, main_process_only=False)
    accelerator.init_trackers(
        project_name="Diffusion_TCGA", 
        config=vars(args),
        init_kwargs={"wandb": {"entity": "master_david", "name": args.output_dir}}
    )

    # Load data
    train_dataset, sampler, _ = load_dataset(
        holdout=args.holdout_mask,
        sample_type='cancer_type',
        cancer_types=args.cancer_types,
        resolution=args.resolution,
        default=len(args.holdout_mask) == 1,
        data_root=data_root
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        collate_fn=custom_collate_fn
    )

    # Prepare metadata for conditioning
    alt_name = "./train_metadata_df_complex.csv"
    df = load_metadata(
        path = ("./train_metadata_df.csv" if len(args.holdout_mask)==1 else "./train_metadata_df_complex.csv"),
        data_root = args.data_root
    )
    df.rename(columns={"age_at_index": "age_p"}, inplace=True)

    domain_dim = {}
    domain_value_mapping = {}
    reverse_domain_value_mapping = {}
    pos_domain_ranges = {}
    positional_domains = []

    if args.domains_to_condition:
        for domain in args.domains_to_condition:
            if domain in df.columns:
                df[domain].replace(
                    ["Unknown", "--", "'--'", "??", "'--", '"--', "'--\""],
                    np.nan, inplace=True
                )
                df[domain].fillna('Unknown', inplace=True)

            if domain.endswith('_p'):
                df[domain] = pd.to_numeric(df[domain], errors="coerce").fillna(60).clip(0, 100)
                domain_tensor = torch.tensor(df[domain].values, dtype=torch.float)
                positional_domains.append(domain)
                pos_domain_ranges[domain] = (0, 100)
            else:
                all_vals = df[domain].unique()
                if 'Unknown' not in all_vals:
                    all_vals = np.append(all_vals, 'Unknown')
                domain_dim[domain] = all_vals
                value_to_idx = {v: i for i, v in enumerate(all_vals)}
                idx_to_value = {i: v for i, v in enumerate(all_vals)}
                domain_value_mapping[domain] = value_to_idx
                reverse_domain_value_mapping[domain] = idx_to_value

    revised_domain_embeds = {k: v for k, v in domain_dim.items()}

    # Cancer type mapping
    selected_cancers = args.cancer_types or df['cancer_type'].unique().tolist()
    selected_cancers = sorted(selected_cancers)
    cancer_type_mapping = {c: i for i, c in enumerate(selected_cancers)}
    reverse_cancer_type_mapping = {i: c for c, i in cancer_type_mapping.items()}

    with open(os.path.join(args.output_dir, 'cancer_type_mapping.json'), 'w') as f:
        json.dump(cancer_type_mapping, f)
    with open(os.path.join(args.output_dir, 'domain_value_mapping.json'), 'w') as f:
        json.dump(domain_value_mapping, f)

    # Initialize UNet model
    sample_size = args.resolution
    in_channels = 3
    out_channels = 3
    block_out_channels = (128, 256, 512, 512, 1024, 1024)
    down_block_types = ("DownBlock2D",) * len(block_out_channels)
    up_block_types = ("UpBlock2D",) * len(block_out_channels)
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        norm_num_groups=32,
        class_embed_type=args.cond_type,
        num_class_embeds=len(cancer_type_mapping),
        domain_embeds=revised_domain_embeds,
        positional_domains=positional_domains,
        pos_domain_ranges=pos_domain_ranges
    )

    # EMA setup
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config
        )

    # Optimizer and schedulers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-08
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.optimization_steps * 0.01),
        num_training_steps=args.optimization_steps
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type="epsilon"
    )

    # Prepare with accelerator
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if args.use_ema:
        ema_model.to(accelerator.device)

    # Training loop
    global_step = 0
    model.train()
    progress_bar = tqdm(total=args.optimization_steps, disable=not accelerator.is_local_main_process)

    while global_step < args.optimization_steps:
        for images, metadata in dataloader:
            images = images.to(accelerator.device)
            df_batch = pd.DataFrame(metadata).rename(columns={"age_at_index": "age_p"}).fillna('Unknown')
            label = torch.tensor(df_batch["cancer_type"].map(cancer_type_mapping)).to(accelerator.device)

            domains = {}
            if args.domains_to_condition:
                for domain in args.domains_to_condition:
                    if domain.endswith('_p'):
                        vals = pd.to_numeric(df_batch[domain], errors="coerce").fillna(60).clip(0, 100)
                        domains[domain] = torch.tensor(vals.values, dtype=torch.float).to(accelerator.device)
                    else:
                        idxs = df_batch[domain].map(domain_value_mapping[domain]).fillna(0).astype(int)
                        domains[domain] = torch.tensor(idxs.values, dtype=torch.long).to(accelerator.device)

            with accelerator.accumulate(model):
                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (images.shape[0],), device=accelerator.device
                ).long()
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                model_output = model(
                    noisy_images, timesteps,
                    class_labels=label,
                    domain_labels=domains
                ).sample
                loss = F.mse_loss(model_output, noise)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                global_step += 1
                
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                if args.use_ema:
                    logs["ema_decay"] = ema_model.cur_decay_value
                #progress_bar.set_postfix(**logs)
                progress_bar.set_postfix(**logs, refresh=False)
                progress_bar.update(1)

                accelerator.log(logs, step=global_step)

                # Checkpoint
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    accelerator.save_state(ckpt_dir)
                    save_noise_scheduler_config(noise_scheduler, args.output_dir)
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(unwrapped.state_dict(), os.path.join(ckpt_dir, 'model_final.pth'))
                    if args.use_ema:
                        ema_model.save_pretrained(os.path.join(ckpt_dir, "ema_model"))
                    logger.info(f"Saved state to {ckpt_dir}")

                # FID tracking
                if global_step % args.FID_tracker == 0 and accelerator.is_main_process:
                    generated_dir = os.path.join(
                        args.output_dir,
                        f"generated_images_FID_step_{global_step}"
                    )
                    os.makedirs(generated_dir, exist_ok=True)

                    for ctype in selected_cancers:
                        idx = cancer_type_mapping[ctype]
                        batch_labels = torch.tensor(
                            [idx] * args.batch_size,
                            device=accelerator.device
                        )
                        if args.domains_to_condition:
                            dom = args.domains_to_condition[0]
                            dom_labels = {
                                dom: torch.tensor([0] * args.batch_size, device=accelerator.device)
                            }
                        else:
                            dom_labels = {}

                        generate_images_for_fid(
                            model=accelerator.unwrap_model(model),
                            noise_scheduler=noise_scheduler,
                            class_labels=batch_labels,
                            domain_labels=dom_labels,
                            batch_size=args.batch_size,
                            num_inference_steps=50,
                            save_dir=generated_dir,
                            cancer_type_name=ctype,
                            n_samples=500
                        )

                    metrics = calculate_metrics(
                        input1=generated_dir,
                        input2=args.data_root,
                        cuda=torch.cuda.is_available(),
                        fid=True,
                        isc=False,
                        kid=False,
                        verbose=False,
                        samples_find_deep=True
                    )
                    fid_score = metrics["frechet_inception_distance"]

                    fid_txt = os.path.join(args.output_dir, "FID_apprx.txt")
                    with open(fid_txt, "a") as f:
                        f.write(f"step {global_step}: {fid_score}\n")

                    accelerator.log({"FID": fid_score}, step=global_step)

            if global_step >= args.optimization_steps:
                break

    progress_bar.close()

    # Save final model
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if args.use_ema:
            ema_model.store(unwrapped.parameters())
            ema_model.copy_to(unwrapped.parameters())
        #torch.save(unwrapped.state_dict(), os.path.join(args.output_dir, 'model_final.pth'))
        #unwrapped.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")
    accelerator.end_training()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

