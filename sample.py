#!/usr/bin/env python3
import time
import argparse
import os
import gc
import random
import math
import json
import warnings

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing as mp
from multiprocessing import Process

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from torch_fidelity import calculate_metrics
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler

from unet2old import UNet2DModel
from load_TCGA import load_metadata, CustomImageDataset, compute_fid

warnings.simplefilter(action='ignore', category=FutureWarning)

PARTIAL = 4  # how many domain values to sample

def uint8_transform(x):
    return (x * 255).to(torch.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="Sample images using a trained diffusion model.")
    parser.add_argument('--path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['OOD','eval','full','vanilla'],
                        help='OOD: out-of-domain generation; eval: in-domain FID; full: both; vanilla: class-only sampling')
    parser.add_argument('--number_of_different_conditional', type=int,
                        help='How many new domain combos per cancer type')
    parser.add_argument('--cancer_types', type=str, nargs='+',
                        help='Subset of cancer types to process')
    parser.add_argument('--n', type=int, default=2048,
                        help='Images per condition')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for sampling')
    parser.add_argument('--domains_to_condition', type=str, nargs='+',
                        help='Which metadata domains to condition on')
    parser.add_argument('--output_dir', type=str,
                        help='Where to save generated images')
    return parser.parse_args()

def prepare_model(model_path, num_class_embeds, class_embed_type,
                  domain_dim, positional_domains, pos_domain_ranges, device):
    """
    Always builds a UNet2DModel (no VAE), loads its weights, and returns (model, None).
    """
    # extract resolution from path if present
    try:
        resolution = int(model_path.split("res:")[1].split("__")[0])
    except:
        resolution = 128

    sample_size = resolution
    in_channels = 3
    out_channels = 3

    # choose architecture size
    if "deep" in model_path:
        down = up = ("DownBlock2D",)*6
        channels = (128,256,512,512,1024,1024)
    else:
        down = up = ("DownBlock2D",)*4
        channels = (128,256,512,1024)

    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        down_block_types=down,
        up_block_types=up,
        block_out_channels=channels,
        norm_num_groups=32,
        class_embed_type=class_embed_type,
        num_class_embeds=num_class_embeds,
        domain_embeds=domain_dim,
        positional_domains=positional_domains,
        pos_domain_ranges=pos_domain_ranges,
    )

    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(device).eval()
    return model, None

def prepare_data(args):
    """
    Loads metadata, builds domain mappings and cancer_type mapping.
    """
    df = load_metadata("./train_metadata_df_complex.csv")
    df.rename(columns={"age_at_index":"age_p"}, inplace=True)

    domain_dim = {}
    domain_value_mapping = {}
    reverse_domain_value_mapping = {}
    pos_domain_ranges = {}
    positional_domains = []

    if args.domains_to_condition:
        for d in args.domains_to_condition:
            if d not in df.columns: continue
            df[d].replace(["Unknown","--","'--'","??"], np.nan, inplace=True)
            df[d].fillna('Unknown', inplace=True)
            if d.endswith('_p'):
                df[d] = pd.to_numeric(df[d], errors='coerce').fillna(60).clip(0,100)
                pos_domain_ranges[d] = (0,100)
                positional_domains.append(d)
            else:
                vals = df[d].unique().tolist()
                domain_dim[d] = vals

        for d, vals in domain_dim.items():
            v2i = {v:i for i,v in enumerate(vals)}
            i2v = {i:v for v,i in v2i.items()}
            domain_value_mapping[d] = v2i
            reverse_domain_value_mapping[d] = i2v

    # cancer types
    cancer_types = sorted(df['cancer_type'].unique().tolist())
    c2i = {c:i for i,c in enumerate(cancer_types)}
    return (df, domain_dim, domain_value_mapping, reverse_domain_value_mapping,
            len(c2i), c2i, positional_domains, pos_domain_ranges)

def generate_images(n, output_dir, model, scheduler,
                    class_label=None, domain_labels=None,
                    batch_size=1, num_inference_steps=50,
                    cancer_type_name="Unknown", domain="Unknown",
                    domain_value="Unknown", start_index=0, vae=None):
    device = model.device
    img_sz = model.config.sample_size
    ch = model.config.in_channels
    generated = 0

    while generated < n:
        bs = min(batch_size, n-generated)
        latents = torch.randn(bs, ch, img_sz, img_sz, device=device)
        cls = class_label.repeat(bs) if class_label is not None else None

        dom = {}
        if domain_labels:
            for k,v in domain_labels.items():
                if k=='age_p' and v.item()==-1:
                    dom[k] = torch.randint(0,101,(bs,),device=device)
                else:
                    dom[k] = v.repeat(bs)

        scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            for t in scheduler.timesteps:
                out = model(latents, t, class_labels=cls, domain_labels=dom).sample
                latents = scheduler.step(out, t, latents).prev_sample

        imgs = (latents/2+0.5).clamp(0,1)
        for i,img in enumerate(imgs):
            idx = start_index + generated + i
            save_dir = os.path.join(output_dir, cancer_type_name, str(domain_value))
            os.makedirs(save_dir, exist_ok=True)
            img_pil = transforms.ToPILImage()(img.cpu())
            img_pil.save(os.path.join(save_dir, f"{idx}.png"))
        generated += bs

def process_gpu(gpu_idx, domain_values, cancer_type, cidx, domain, dom_map,
                model_path, out_dir, n, bs, dom_dim, doms, num_cls, ctype, rev_dom, pos_dom, pos_range):
    dev = torch.device(f'cuda:{gpu_idx}')
    model, _ = prepare_model(model_path, num_cls, ctype, dom_dim, pos_dom, pos_range, dev)
    cls = torch.tensor([cidx],device=dev)
    sched_path = os.path.join(os.path.dirname(os.path.dirname(model_path)),"scheduler_config.json")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(sched_path,safety_checker=None)
    dom_labels = {}
    if isinstance(domain, (list,tuple)) and len(domain)==4:
        a,b,c,d = domain
        dom_labels.update({
            "tissue_source_site": torch.tensor([dom_map["tissue_source_site"][a]],device=dev),
            "gender":             torch.tensor([dom_map["gender"][b]],device=dev),
            "race":               torch.tensor([dom_map["race"][c]],device=dev),
            "age_p":              torch.tensor([d],device=dev),
        })
    else:
        val = domain[0] if isinstance(domain,tuple) else domain
        dom_labels["tissue_source_site"] = torch.tensor([dom_map["tissue_source_site"][val]],device=dev)

    save_dir = os.path.join(out_dir,cancer_type,str(domain))
    os.makedirs(save_dir,exist_ok=True)
    exist = [int(os.path.splitext(f)[0]) for f in os.listdir(save_dir) if f.endswith('.png')]
    start = max(exist)+1 if exist else 0
    rem = n - start
    if rem>0:
        generate_images(rem, out_dir, model, scheduler, cls, dom_labels,
                        bs,50,cancer_type,"","",start,None)

def generate_OOD(df, dom_dim, dom_map, rev_dom, num_cls, c2i, pos_dom, pos_range,
                 model_path, out_dir, n, bs, doms, ctype, num_gpus):
    global PARTIAL
    df.columns = df.columns.str.strip()
    train_vals = df.groupby('cancer_type')['tissue_source_site'].unique().to_dict()
    for cancer in train_vals:
        idx = c2i[cancer]
        all_vals = set(dom_dim["tissue_source_site"])
        used    = set(train_vals[cancer])
        comp    = all_vals - used
        if not comp: continue

        if len(doms)==1:
            sample = random.sample(list(comp), min(PARTIAL,len(comp)))
            queues = {i:[] for i in range(num_gpus)}
            for i,val in enumerate(sample):
                queues[i%num_gpus].append(val)
            procs=[]
            for gpu,vals in queues.items():
                for v in vals:
                    p=Process(target=process_gpu,args=(gpu,vals,cancer,idx, v,dom_map,
                                                        model_path,out_dir,n,bs,dom_dim,doms,
                                                        num_cls,ctype,rev_dom,pos_dom,pos_range))
                    p.start(); procs.append(p)
            for p in procs: p.join()

        elif len(doms)==4:
            gvals = random.sample(dom_dim["gender"],min(PARTIAL,len(dom_dim["gender"]))
)
            rvals = random.sample(dom_dim["race"],  min(PARTIAL,len(dom_dim["race"]))
)
            avals = random.sample(list(comp),      min(PARTIAL,len(comp)))
            for a in avals:
                for b in gvals:
                    for c in rvals:
                        combo = (a,b,c,-1)
                        p=Process(target=process_gpu,args=(0,[combo],cancer,idx, combo,dom_map,
                                                           model_path,out_dir,n,bs,dom_dim,doms,
                                                           num_cls,ctype,rev_dom,pos_dom,pos_range))
                        p.start(); p.join()
        else:
            print("Only 1 or 4 domains supported for OOD")

def generate_and_evaluate_ID(df, dom_dim, dom_map, rev_dom, num_cls, c2i, pos_dom, pos_range,
                             model_path, out_dir, n, bs, doms, ctype, num_gpus):
    print("\n=== ID Generation & FID ===")
    id_dir = os.path.join(os.path.dirname(model_path),"ID_images")
    os.makedirs(id_dir, exist_ok=True)

    if len(doms)==1:
        groups = df.groupby(['cancer_type','tissue_source_site'])
    elif len(doms)==4:
        groups = df.groupby(['cancer_type','tissue_source_site','gender','race','age_p'])
    else:
        print("Only 1 or 4 domains supported for ID"); return

    # spawn procs
    for keys,grp in groups:
        cancer = keys[0] if isinstance(keys,tuple) else keys
        idx = c2i[cancer]
        combo = keys[1:] if isinstance(keys,tuple) else keys
        images_n = max(2048, len(grp))//len(grp)
        p=Process(target=process_gpu,args=(0,[combo],cancer,idx,combo,dom_map,
                                           model_path,id_dir,images_n,bs,
                                           dom_dim,doms,num_cls,ctype,rev_dom,pos_dom,pos_range))
        p.start(); p.join()

    # compute FID per cancer type
    compute_fid_per_cancer_type(df, id_dir)

def compute_fid_per_cancer_type(df, base_dir):
    scores = {}
    for c in df['cancer_type'].unique():
        real = os.path.join('./TCGA',c)
        gen  = os.path.join(base_dir,c)
        if os.path.isdir(real) and os.path.isdir(gen):
            try:
                fid = compute_fid(real,gen)
                scores[c]=fid
                print(f"[FID_ID] {c}: {fid:.3f}")
            except Exception as e:
                print(f"Error FID {c}: {e}")
    # write out
    with open(os.path.join(base_dir,"FID_ID_results.txt"),'w') as f:
        for c,v in scores.items():
            f.write(f"{c}: {v}\n")

def FID_TSS_Class(df, model_path):
    model_dir = os.path.dirname(model_path)
    out = os.path.join(model_dir,"FID_TSS_Class_results.txt")
    df.columns=df.columns.str.strip()
    scores={}
    for (c,tss),grp in df.groupby(['cancer_type','tissue_source_site']):
        real_dir = os.path.join('./TCGA',c)
        gen_dir  = os.path.join(model_dir,"ID_images",c,str(tss))
        if not os.path.isdir(real_dir) or not os.path.isdir(gen_dir):
            continue
        try:
            fid = compute_fid(real_dir,gen_dir)
            scores[(c,tss)] = fid
            print(f"[FID_TSS] {c}/{tss}: {fid:.3f}")
        except: pass
    with open(out,'w') as f:
        for (c,tss),v in scores.items():
            f.write(f"{c}/{tss}: {v}\n")

def sampler_FID(path,args):
    """
    Ablation over samplers & step counts as in main.
    """
    print("=== Ablate Samplers ===")
    # implement as needed...
    pass

def generate_images_vanilla_model(n, df, out_dir, model, scheduler, steps, vae, c2i):
    device = next(model.parameters()).device
    for c, idx in c2i.items():
        cls = torch.tensor([idx],device=device)
        print(f"Vanilla gen for {c}")
        generate_images(n, out_dir, model, scheduler, cls, None,
                        batch_size=128, num_inference_steps=steps,
                        cancer_type_name=c, domain="no_domain",
                        domain_value="no_domain", start_index=0, vae=None)

def main():
    args = parse_args()
    global PARTIAL
    if args.number_of_different_conditional is not None:
        PARTIAL = args.number_of_different_conditional

    random.seed(42)
    model_path = args.path
    out_dir = args.output_dir or os.path.join(os.path.dirname(model_path),'OOD_images')
    os.makedirs(out_dir, exist_ok=True)

    print("GPUs:",torch.cuda.device_count())
    df, dom_dim, dom_map, rev_dom, num_cls, c2i, pos_dom, pos_range = prepare_data(args)

    if args.cancer_types:
        df = df[df['cancer_type'].isin(args.cancer_types)].reset_index(drop=True)

    num_gpus = torch.cuda.device_count() or 1

    # scheduler for vanilla
    sched_path = os.path.join(os.path.dirname(os.path.dirname(model_path)),"scheduler_config.json")
    vanilla_sched = DPMSolverMultistepScheduler.from_pretrained(sched_path,safety_checker=None)

    if args.mode=="OOD":
        generate_OOD(df, dom_dim, dom_map, rev_dom, num_cls, c2i, pos_dom, pos_range,
                     model_path, out_dir, args.n, args.batch_size,
                     args.domains_to_condition, args, num_gpus)

    elif args.mode=="eval":
        generate_and_evaluate_ID(df, dom_dim, dom_map, rev_dom, num_cls, c2i,
                                 pos_dom, pos_range, model_path, out_dir,
                                 args.n, args.batch_size, args.domains_to_condition,
                                 args, num_gpus)
        FID_TSS_Class(df,model_path)

    elif args.mode=="full":
        generate_OOD(df, dom_dim, dom_map, rev_dom, num_cls, c2i,
                     pos_dom, pos_range, model_path, out_dir,
                     args.n, args.batch_size, args.domains_to_condition,
                     args, num_gpus)
        generate_and_evaluate_ID(df, dom_dim, dom_map, rev_dom, num_cls, c2i,
                                 pos_dom, pos_range, model_path, out_dir,
                                 args.n, args.batch_size, args.domains_to_condition,
                                 args, num_gpus)
        FID_TSS_Class(df,model_path)

    elif args.mode=="vanilla":
        model, _ = prepare_model(model_path, num_cls, None,
                                 dom_dim, pos_dom, pos_range,
                                 torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        generate_images_vanilla_model(args.n, df, out_dir, model,
                                      vanilla_sched, 50, None, c2i)
        # FID per cancer
        compute_fid_per_cancer_type(df,out_dir)

    else:
        raise ValueError("Unknown mode")

    print("\n=== Done! ===")

if __name__=="__main__":
    mp.set_start_method('spawn', force=True)
    main()

