#!/usr/bin/env python3
from torchvision.transforms import ToPILImage
import time
import argparse
import os
import gc
import random
import json
import math
import re
import warnings
from multiprocessing import Process
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import lpips
import wandb

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
)
from safetensors.torch import load_file

from unet import UNet2DModel
from load_TCGA import load_metadata, CustomImageDataset, compute_fid

warnings.simplefilter(action='ignore', category=FutureWarning)

PARTIAL = 4  # Controls how many domain values are sampled

def uint8_transform(x):
    return (x * 255).to(torch.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="Sample images using a trained diffusion model.")
    parser.add_argument('--path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--mode', type=str, choices=['OOD','eval','full'], required=True,
                        help='Mode: eval (FID), OOD (generate OOD), full (both).')
    parser.add_argument('--number_of_different_conditional', type=int,
                        help='How many new TSS/gender/race combos per cancer type.')
    parser.add_argument('--cancer_types', type=str, nargs='+',
                        help='List of cancer types to generate samples for.')
    parser.add_argument('--n', type=int, default=2048, help='Number of images per condition.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for image generation.')
    parser.add_argument('--domains_to_condition', type=str, nargs='+', default=None,
                        help='Metadata domains to condition on.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save generated images.')
    parser.add_argument('--real_data_root', type=str, default='./TCGA',
                        help='Root directory of real TCGA images (per–cancer subfolders).')
    return parser.parse_args()

def prepare_model(model_path, num_class_embeds, class_embed_type,
                  domain_dim, positional_domains, pos_domain_ranges, device):
    """
    Loads only the UNet model weights (no VAE).
    """
    try:
        resolution = int(model_path.split("res:")[1].split("__")[0])
    except:
        resolution = 128

    model = UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D",)* (6 if "deep" in model_path else 4),
        up_block_types=("UpBlock2D",)* (6 if "deep" in model_path else 4),
        block_out_channels=(128,256,512,512,1024,1024) if "deep" in model_path else (128,256,512,1024),
        norm_num_groups=32,
        class_embed_type=class_embed_type,
        num_class_embeds=num_class_embeds,
        domain_embeds=domain_dim,
        positional_domains=positional_domains,
        pos_domain_ranges=pos_domain_ranges,
    )

    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def generate_images(
    n,
    output_dir,
    model,
    scheduler,
    class_label=None,
    domain_labels=None,
    batch_size=1,
    num_inference_steps=50,
    cancer_type_name="Unknown",
    domain="Unknown",
    domain_value="Unknown",
    start_index=0,
):
    device = model.device
    image_size = model.config.sample_size
    channels   = model.config.in_channels
    MIN_BATCH_SIZE = 1
    images_generated = 0

    while images_generated < n:
        bs = min(batch_size, n - images_generated)
        try:
            latents = torch.randn((bs, channels, image_size, image_size), device=device)
            batch_class = class_label.repeat(bs) if class_label is not None else None

            batch_domain = {}
            if domain_labels is not None:
                for k,v in domain_labels.items():
                    if k=="age_p" and v.item()==-1:
                        batch_domain["age_p"] = torch.randint(0,101,(bs,),device=device)
                    else:
                        batch_domain[k] = v.repeat(bs)

            scheduler.set_timesteps(num_inference_steps)
            with torch.no_grad():
                for t in scheduler.timesteps:
                    out = model(latents, t, class_labels=batch_class, domain_labels=batch_domain).sample
                    latents = scheduler.step(out, t, latents).prev_sample

            images = (latents/2+0.5).clamp(0,1)
            for i,img_t in enumerate(images):
                idx = start_index + images_generated + i
                pil = transforms.ToPILImage()(img_t.cpu())
                dv  = str(domain_value).strip('_')
                sd  = os.path.join(output_dir, cancer_type_name, dv)
                os.makedirs(sd, exist_ok=True)
                path = os.path.join(sd, f"{idx}.png")
                pil.save(path)
                print(f"Saved {path}")

            images_generated += bs
            print(f"Generated {images_generated}/{n} for {cancer_type_name}/{domain}/{domain_value}")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                batch_size = max(MIN_BATCH_SIZE, batch_size//2)
                print(f"OOM, reducing batch_size to {batch_size}")
                del latents; torch.cuda.empty_cache(); gc.collect()
            else:
                raise

    print(f"Done generating for {cancer_type_name}/{domain}/{domain_value}")

def process_gpu(
    gpu_idx,
    domain_values_list,
    cancer_type_name,
    class_index,
    domain_value,
    domain_value_mapping,
    model_path,
    output_dir,
    n,
    batch_size,
    domain_dim,
    domains_to_condition,
    num_class_embeds,
    class_embed_type,
    reverse_domain_value_mapping,
    positional_domains,
    pos_domain_ranges
):
    # --- normalize single-element tuples so we get "3L" instead of ("3L",) ---
    if isinstance(domain_value, (tuple, list)) and len(domain_value) == 1:
        domain_value = domain_value[0]

    device = torch.device(f'cuda:{gpu_idx}')
    model = prepare_model(
        model_path,
        num_class_embeds,
        class_embed_type,
        domain_dim,
        positional_domains,
        pos_domain_ranges,
        device
    )
    print(f"[GPU {gpu_idx}] loaded model for values {domain_values_list}")

    class_label = torch.tensor([class_index], device=device)

    scheduler_path = os.path.join(os.path.dirname(os.path.dirname(model_path)),
                                  "scheduler_config.json")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        scheduler_path, safety_checker=None
    )

    # build the label dict
    labels = {}
    if isinstance(domain_value, (tuple, list)) and len(domain_value) == 4:
        a, b, c, age = domain_value
        labels["tissue_source_site"] = torch.tensor(
            domain_value_mapping["tissue_source_site"][a], device=device
        )
        labels["gender"]  = torch.tensor(
            domain_value_mapping["gender"][b], device=device
        )
        labels["race"]    = torch.tensor(
            domain_value_mapping["race"][c], device=device
        )
        labels["age_p"]   = torch.tensor([age], device=device)
    else:
        # here domain_value is guaranteed to be a scalar
        labels["tissue_source_site"] = torch.tensor(
            domain_value_mapping["tissue_source_site"][domain_value], device=device
        )

    dv_str = str(domain_value).strip('_')
    save_dir = os.path.join(output_dir, cancer_type_name, dv_str)
    os.makedirs(save_dir, exist_ok=True)
    existing = [
        int(os.path.splitext(f)[0])
        for f in os.listdir(save_dir) if f.endswith('.png') and f.split('.')[0].isdigit()
    ]
    start = max(existing) + 1 if existing else 0

    remaining = n - start
    if remaining > 0:
        print(f"[GPU {gpu_idx}] Generating {remaining} imgs for {cancer_type_name}/{dv_str}")
        generate_images(
            remaining,
            output_dir,
            model,
            scheduler,
            class_label=class_label,
            domain_labels=labels,
            batch_size=batch_size,
            num_inference_steps=100,
            cancer_type_name=cancer_type_name,
            domain=";".join(map(str, domain_value)) if isinstance(domain_value, (tuple, list)) else domain_value,
            domain_value=domain_value,
            start_index=start,
        )
    else:
        print(f"[GPU {gpu_idx}] Already have {n} images, skipping.")

def prepare_data(model_path, args):
    df = load_metadata("./train_metadata_df_complex.csv")
    df.rename(columns={"age_at_index":"age_p"}, inplace=True)

    domain_dim = {}
    domain_value_mapping = {}
    reverse_domain_value_mapping = {}
    pos_domain_ranges = {}
    positional_domains  = []

    if args.domains_to_condition:
        for dom in args.domains_to_condition:
            if dom.endswith('_p'):
                df[dom] = pd.to_numeric(df[dom],errors="coerce").fillna(60).clip(0,100)
                pos_domain_ranges[dom] = (0,100)
                positional_domains.append(dom)
            else:
                vals = df[dom].fillna('Unknown').unique()
                domain_dim[dom] = vals
                domain_value_mapping[dom] = {v:i for i,v in enumerate(vals)}
                reverse_domain_value_mapping[dom] = {i:v for v,i in domain_value_mapping[dom].items()}

    cancers = sorted(df['cancer_type'].unique())
    cancer_type_mapping = {c:i for i,c in enumerate(cancers)}
    num_class_embeds = len(cancers)

    return (df, domain_dim, domain_value_mapping,
            reverse_domain_value_mapping, num_class_embeds,
            cancer_type_mapping, positional_domains, pos_domain_ranges)

def generate_OOD(
    df, domain_dim, domain_value_mapping, reverse_domain_value_mapping,
    num_class_embeds, cancer_type_mapping, positional_domains,
    pos_domain_ranges, model_path, output_dir,
    n, batch_size, domains_to_condition, class_embed_type, num_gpus
):
    global PARTIAL
    df.columns = df.columns.str.strip()
    seen_map = df.groupby('cancer_type')['tissue_source_site'].unique().to_dict()

    for ctype, seen in seen_map.items():
        all_vals   = set(domain_dim["tissue_source_site"])
        complement = list(all_vals - set(seen))
        if not complement:
            print(f"No OOD for {ctype}")
            continue
        picks = random.sample(complement, min(PARTIAL,len(complement)))
        gpu_map = {i:[] for i in range(num_gpus)}
        for i,v in enumerate(picks):
            gpu_map[i%num_gpus].append(v)

        procs=[]
        for gpu,vals in gpu_map.items():
            for v in vals:
                p = Process(target=process_gpu, args=(
                    gpu, vals, ctype, cancer_type_mapping[ctype],
                    v, domain_value_mapping, model_path,
                    output_dir, n, batch_size,
                    domain_dim, domains_to_condition,
                    num_class_embeds, class_embed_type,
                    reverse_domain_value_mapping,
                    positional_domains, pos_domain_ranges
                ))
                p.start(); procs.append(p)
        for p in procs: p.join()

def generate_and_evaluate_ID(
    df,
    domain_dim,
    domain_value_mapping,
    reverse_domain_value_mapping,
    num_class_embeds,
    cancer_type_mapping,
    positional_domains,
    pos_domain_ranges,
    model_path,
    output_dir,
    n,                    # no longer drives count
    initial_batch_size,
    domains_to_condition,
    class_embed_type,
    num_gpus,
    real_data_root       # pass this in from main()
):
    """
    For each (cancer_type, TSS[,(gender,race,age_p)]):
      1. Find all slide_submitter_id in the group.
      2. Count real images under real_data_root/<cancer_type>/<slide_submitter_id>/...
      3. Spawn exactly that many synthetic images.
    """
    print("\n=== Generating ID images (1:1 real→synthetic) ===")
    id_dir = os.path.join(os.path.dirname(model_path), "ID_images")
    os.makedirs(id_dir, exist_ok=True)

    df.columns = df.columns.str.strip()
    # pick grouping keys
    if len(domains_to_condition) == 1:
        grouped = df.groupby(['cancer_type','tissue_source_site'])
    elif len(domains_to_condition) == 4:
        grouped = df.groupby(['cancer_type','tissue_source_site','gender','race','age_p'])
    else:
        print("ID eval supports only 1 or 4 domains.")
        return

    for keys, subdf in grouped:
        # unpack
        if isinstance(keys, tuple):
            ctype = keys[0]
            combo = keys[1:]
            tss   = combo[0]
        else:
            ctype, tss = keys, keys
            combo = tss

        # collect all slide IDs in this group
        slide_ids = subdf['slide_submitter_id'].unique()

        # count real images across those slide folders
        real_count = 0
        for sid in slide_ids:
            slide_dir = os.path.join(real_data_root, ctype, sid)
            if not os.path.isdir(slide_dir):
                continue
            for _root, _dirs, files in os.walk(slide_dir):
                real_count += sum(f.lower().endswith(('.png','.jpg','.jpeg'))
                                  for f in files)

        if real_count == 0:
            print(f"[SKIP] {ctype}/{tss} → no real images found.")
            continue

        print(f"[{ctype}/{tss}] real_count = {real_count} → generating {real_count}")

        # dispatch to GPU(s) — here we just put them all on GPU 0
        procs = []
        for gpu_idx in range(num_gpus):
            # each process_gpu will be told to generate `real_count`
            p = Process(
                target=process_gpu,
                args=(
                    gpu_idx,
                    [combo],
                    ctype,
                    cancer_type_mapping[ctype],
                    combo,
                    domain_value_mapping,
                    model_path,
                    id_dir,
                    real_count,
                    initial_batch_size,
                    domain_dim,
                    domains_to_condition,
                    num_class_embeds,
                    class_embed_type,
                    reverse_domain_value_mapping,
                    positional_domains,
                    pos_domain_ranges
                )
            )
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    print("=== Done generating ID images ===\n")

def compute_fid_per_cancer_type(
    df,
    real_root: str,
    gen_root: str,
    model_resolution: int = 256,
):
    """
    For each cancer_type:
      downsample real & gen to model_resolution,
      then call load_TCGA.compute_fid().
    """
    fid_tf = transforms.Compose([
        transforms.Resize(model_resolution),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.uint8),
    ])

    scores = {}
    for ctype in df['cancer_type'].unique():
        rd = os.path.join(real_root, ctype)
        gd = os.path.join(gen_root,  ctype)
        if not os.path.isdir(rd) or not os.path.isdir(gd):
            print(f"[SKIP] {ctype}: missing dir"); continue
        try:
            val = compute_fid(rd, gd, transform=fid_tf)
            scores[ctype] = val
            print(f"FID[{ctype}] @ {model_resolution}px → 299px = {val:.2f}")
        except Exception as e:
            scores[ctype] = None
            print(f"[ERR] FID {ctype}: {e}")
    valid = [v for v in scores.values() if v is not None]
    avg   = float(np.mean(valid)) if valid else float('nan')
    print(f"Average FID @ {model_resolution}px: {avg:.2f}")
    return scores, avg



def generate_images_for_fid(
    model,
    noise_scheduler,
    class_labels,
    domain_labels,
    batch_size,
    num_inference_steps,
    save_dir,
    cancer_type_name,
    n_samples=500
):
    model.eval()
    device = class_labels.device
    out_dir = os.path.join(save_dir, cancer_type_name)
    os.makedirs(out_dir, exist_ok=True)

    to_pil = ToPILImage()
    generated = 0

    while generated < n_samples:
        bs = min(batch_size, n_samples - generated)
        latents = torch.randn(
            (bs,
             model.config.in_channels,
             model.config.sample_size,
             model.config.sample_size),
            device=device
        )

        noise_scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            for t in noise_scheduler.timesteps:
                denoised = model(
                    latents,
                    t,
                    class_labels=class_labels[:bs],
                    domain_labels={k: v[:bs] for k, v in domain_labels.items()}
                ).sample
                latents = noise_scheduler.step(denoised, t, latents).prev_sample

        # decode (here latents are already in [−1,1] space for pixel output)
        images = (latents / 2 + 0.5).clamp(0, 1).cpu()

        for i in range(bs):
            img = images[i]
            to_pil(img).save(os.path.join(out_dir, f"{generated + i}.png"))

        generated += bs
    
def main():
    args = parse_args()
    # adjust PARTIAL if specified
    if args.number_of_different_conditional:
        global PARTIAL
        PARTIAL = args.number_of_different_conditional

    # reproducibility
    random.seed(42)

    # paths
    model_path = args.path
    output_dir = args.output_dir or os.path.join(os.path.dirname(model_path), 'OOD_images')
    os.makedirs(output_dir, exist_ok=True)

    # extract resolution for later (if needed)
    try:
        resolution = int(model_path.split("res:")[1].split("__")[0])
    except Exception:
        resolution = 128

    # load metadata and domain mappings
    (df,
     domain_dim,
     domain_value_mapping,
     reverse_domain_value_mapping,
     num_class_embeds,
     cancer_type_mapping,
     positional_domains,
     pos_domain_ranges) = prepare_data(model_path, args)

    # filter by cancer types if provided
    if args.cancer_types:
        df = df[df['cancer_type'].isin(args.cancer_types)].reset_index(drop=True)

    # GPU count
    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")

    # determine class embedding type from filename
    if "linearembed" in model_path:
        class_embed_type = "linear"
    elif "concatembed" in model_path:
        class_embed_type = "concat"
    elif "additive" in model_path:
        class_embed_type = "additive"
    else:
        class_embed_type = None

    # ─────────── CLASS‐ONLY MODE ───────────
    if not args.domains_to_condition:
        print("No metadata domains specified – running class‐only sampling.")
        # load scheduler once
        scheduler_path = os.path.join(
            os.path.dirname(os.path.dirname(model_path)),
            "scheduler_config.json"
        )
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            scheduler_path, safety_checker=None
        )

        # generate for each cancer type
        for ctype, class_index in cancer_type_mapping.items():
            print(f"[CLASS‐ONLY] Generating {args.n} for {ctype}")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = prepare_model(
                model_path,
                num_class_embeds,
                class_embed_type,
                domain_dim={},                # no domain embeddings
                positional_domains=[],
                pos_domain_ranges={},
                device=device
            )
            class_label = torch.tensor([class_index], device=device)

            # figure out start index
            save_dir = os.path.join(output_dir, ctype, "class_only")
            os.makedirs(save_dir, exist_ok=True)
            existing = [
                int(os.path.splitext(f)[0])
                for f in os.listdir(save_dir)
                if f.endswith('.png') and f.split('.')[0].isdigit()
            ]
            start = max(existing) + 1 if existing else 0
            remaining = args.n - start

            if remaining > 0:
                generate_images(
                    n=remaining,
                    output_dir=output_dir,
                    model=model,
                    scheduler=scheduler,
                    class_label=class_label,
                    domain_labels=None,
                    batch_size=args.batch_size,
                    num_inference_steps=50,
                    cancer_type_name=ctype,
                    domain="class_only",
                    domain_value="class_only",
                    start_index=start,
                )
            else:
                print(f"[CLASS‐ONLY] Already have {args.n} images for {ctype}, skipping.")
        return

    # ───── METADATA‐DRIVEN MODES ─────
    if args.mode == "OOD":
        generate_OOD(
            df, domain_dim, domain_value_mapping, reverse_domain_value_mapping,
            num_class_embeds, cancer_type_mapping, positional_domains,
            pos_domain_ranges, model_path, output_dir,
            args.n, args.batch_size,
            args.domains_to_condition, class_embed_type, num_gpus
        )

    elif args.mode == "eval":
        # generate ID images & compute FID
        id_dir = os.path.join(os.path.dirname(model_path), "ID_images")
        generate_and_evaluate_ID(
            df, domain_dim, domain_value_mapping, reverse_domain_value_mapping,
            num_class_embeds, cancer_type_mapping, positional_domains,
            pos_domain_ranges, model_path, output_dir,
            args.n, args.batch_size,
            args.domains_to_condition, class_embed_type, num_gpus,
            args.real_data_root
        )
        compute_fid_per_cancer_type(df, args.real_data_root, id_dir, resolution)

    elif args.mode == "full":
        generate_OOD(
            df, domain_dim, domain_value_mapping, reverse_domain_value_mapping,
            num_class_embeds, cancer_type_mapping, positional_domains,
            pos_domain_ranges, model_path, output_dir,
            args.n, args.batch_size,
            args.domains_to_condition, class_embed_type, num_gpus
        )
        generate_and_evaluate_ID(
            df, domain_dim, domain_value_mapping, reverse_domain_value_mapping,
            num_class_embeds, cancer_type_mapping, positional_domains,
            pos_domain_ranges, model_path, output_dir,
            args.n, args.batch_size,
            args.domains_to_condition, class_embed_type, num_gpus,
            args.real_data_root
        )
        id_dir = os.path.join(os.path.dirname(model_path), "ID_images")
        compute_fid_per_cancer_type(df, args.real_data_root, id_dir, resolution)

    else:
        raise ValueError("Mode must be one of: OOD, eval, full")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
