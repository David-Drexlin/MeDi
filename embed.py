#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
import timm
from PIL import Image
from torchvision import transforms

from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from unet import UNet2DModel
from load_TCGA import load_metadata

# your mapping from TCGA project IDs → descriptive cancer_type names
project_id_to_cancer_type = {
    'TCGA-ACC': 'Adrenocortical_carcinoma',
    'TCGA-BLCA': 'Bladder_Urothelial_Carcinoma',
    'TCGA-LGG': 'Brain_Lower_Grade_Glioma',
    'TCGA-BRCA': 'Breast_invasive_carcinoma',
    'TCGA-CESC': 'Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma',
    'TCGA-CHOL': 'Cholangiocarcinoma',
    'TCGA-COAD': 'Colon_adenocarcinoma',
    'TCGA-ESCA': 'Esophageal_carcinoma',
    'TCGA-GBM': 'Glioblastoma_multiforme',
    'TCGA-HNSC': 'Head_and_Neck_squamous_cell_carcinoma',
    'TCGA-KICH': 'Kidney_Chromophobe',
    'TCGA-KIRC': 'Kidney_renal_clear_cell_carcinoma',
    'TCGA-KIRP': 'Kidney_renal_papillary_cell_carcinoma',
    'TCGA-LIHC': 'Liver_hepatocellular_carcinoma',
    'TCGA-LUAD': 'Lung_adenocarcinoma',
    'TCGA-LUSC': 'Lung_squamous_cell_carcinoma',
    'TCGA-DLBC': 'Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma',
    'TCGA-MESO': 'Mesothelioma',
    'TCGA-OV': 'Ovarian_serous_cystadenocarcinoma',
    'TCGA-PAAD': 'Pancreatic_adenocarcinoma',
    'TCGA-PCPG': 'Pheochromocytoma_and_Paraganglioma',
    'TCGA-PRAD': 'Prostate_adenocarcinoma',
    'TCGA-READ': 'Rectum_adenocarcinoma',
    'TCGA-SARC': 'Sarcoma',
    'TCGA-SKCM': 'Skin_Cutaneous_Melanoma',
    'TCGA-STAD': 'Stomach_adenocarcinoma',
    'TCGA-TGCT': 'Testicular_Germ_Cell_Tumors',
    'TCGA-THYM': 'Thymoma',
    'TCGA-THCA': 'Thyroid_carcinoma',
    'TCGA-UCS': 'Uterine_Carcinosarcoma',
    'TCGA-UCEC': 'Uterine_Corpus_Endometrial_Carcinoma',
    'TCGA-UVM': 'Uveal_Melanoma',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Sample and embed via diffusion+uni.")
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the model SafeTensor checkpoint.')
    parser.add_argument('--cancer_types', type=str, nargs='+', required=True,
                        help='List of descriptive cancer types to sample.')
    parser.add_argument('--mode_union', action='store_true',
                        help='If set, sample the UNION of TSS across selected cancers; otherwise keep per-cancer.')
    parser.add_argument('--tss_union_csv', type=str, default=None,
                        help='CSV with columns cls,tss,count (cls = TCGA project ID). Falls back to train_metadata_df_complex.csv.')
    parser.add_argument('--n', type=int, default=2048,
                        help='Number of samples per (class,TSS).')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--infer_steps', type=int, default=100,
                        help='Inference timesteps.')
    parser.add_argument('--sampler', type=str, choices=['ddim','dpmsolver'], default='ddim',
                        help='Which scheduler to use.')
    parser.add_argument('--domains_to_condition', type=str, nargs='+', default=['tissue_source_site'],
                        help='Domains to condition on.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--holdout_meta', type=str, required=True,
                        help='Path to holdout_metadata_df_complex.csv')
    parser.add_argument('--real_data_root', type=str, required=True,
                        help='Root dir for real TCGA slides')
    return parser.parse_args()

def prepare_model(model_path, num_class_embeds, class_embed_type,
                  domain_dim, positional_domains, pos_domain_ranges, device):
    try:
        resolution = int(model_path.split('res:')[1].split('__')[0])
    except:
        resolution = 128
    down = ('DownBlock2D',)*(6 if 'deep' in model_path else 4)
    up   = ('UpBlock2D',)*(6 if 'deep' in model_path else 4)
    channels = (128,256,512,512,1024,1024) if 'deep' in model_path else (128,256,512,1024)

    model = UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
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
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def sample_latents(model, scheduler, n, batch_size, steps, device,
                   class_label=None, domain_labels=None):
    scheduler.set_timesteps(steps)
    imgs = []
    generated = 0
    C, H = model.config.in_channels, model.config.sample_size
    while generated < n:
        bs = min(batch_size, n - generated)
        lat = torch.randn((bs, C, H, H), device=device)
        cls = class_label.repeat(bs) if class_label is not None else None
        dom = {k:v.repeat(bs) for k,v in domain_labels.items()} if domain_labels else None
        with torch.no_grad():
            for t in scheduler.timesteps:
                out = model(lat, t, class_labels=cls, domain_labels=dom).sample
                lat = scheduler.step(out, t, lat).prev_sample
        for i in range(bs):
            imgs.append(lat[i].cpu().numpy())
        generated += bs
    return [img for img in imgs]


def embed_real_images(uni, holdout_csv, real_root, selected_cancers, device, out_root):
    dfh = load_metadata(holdout_csv)
    dfh.columns = dfh.columns.str.strip()
    dfh = dfh[dfh['cancer_type'].isin(selected_cancers)]
    to_tensor = transforms.ToTensor()

    for ctype, sub in tqdm(dfh.groupby('cancer_type'), desc="Embedding real images"):
        save_dir = os.path.join(out_root, ctype)
        os.makedirs(save_dir, exist_ok=True)
        idx = 0
        for sid in sub['slide_submitter_id'].unique():
            slide_dir = os.path.join(real_root, ctype, sid)
            if not os.path.isdir(slide_dir):
                continue
            for fname in os.listdir(slide_dir):
                if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                    continue
                img = Image.open(os.path.join(slide_dir, fname)).convert('RGB')
                tensor = to_tensor(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = uni(tensor).cpu().numpy()[0]
                outp = os.path.join(save_dir, f"{sid}_{idx}.npz")
                np.savez_compressed(outp, embedding=emb)
                idx += 1


def main():
    args = parse_args()
    os.makedirs(args.output_dir or os.path.join(os.path.dirname(args.path),'embeddings'), exist_ok=True)

    # 1) load train metadata
    df = load_metadata('./train_metadata_df_complex.csv')
    df.columns = df.columns.str.strip()

    # 2) build tss_df from either union CSV or train CSV (use project_id → map to descriptive name)
    if args.tss_union_csv:
        tss_df = pd.read_csv(args.tss_union_csv)[['cls','tss']]
    else:
        tss_df = df[['project_id','tissue_source_site']].dropna()
        tss_df.columns = ['cls','tss']

    # map TCGA project IDs → descriptive cancer_type; drop any unmapped
    tss_df['cls'] = tss_df['cls'].map(project_id_to_cancer_type)
    tss_df = tss_df.dropna(subset=['cls'])

    # 3) union vs separate, **restricted** to args.cancer_types
    if args.mode_union:
        tss_df = tss_df[tss_df['cls'].isin(args.cancer_types)]
        all_tss = sorted(tss_df['tss'].unique().tolist())
        dom_map = {t:i for i,t in enumerate(all_tss)}
    else:
        all_tss = None

    # 4) rest of your original code unchanged:
    cancers = sorted(df['cancer_type'].unique())
    num_class_embeds = len(cancers)
    domain_dim = {
        'tissue_source_site': df['tissue_source_site']
                                    .dropna()
                                    .unique()
                                    
    }
    positional_domains = []
    pos_domain_ranges   = {}

    mpath = args.path
    if 'linearembed' in mpath:   t = 'linear'
    elif 'concatembed' in mpath: t = 'concat'
    elif 'additive' in mpath:    t = 'additive'
    else:                         t = None

    model = prepare_model(args.path, num_class_embeds, t,
                          domain_dim, positional_domains, pos_domain_ranges,
                          args.device)
    scheduler = (DDIMScheduler if args.sampler=='ddim'
                 else DPMSolverMultistepScheduler) \
                .from_pretrained(os.path.dirname(os.path.dirname(args.path)))

    uni = timm.create_model('hf-hub:MahmoodLab/uni',
                             pretrained=True, init_values=1e-5,
                             dynamic_img_size=True).to(args.device)
    uni.eval()

    # 5) sampling loop
    for c in args.cancer_types:
        class_idx   = cancers.index(c)
        class_label = torch.tensor([class_idx], device=args.device)

        if args.mode_union:
            tss_vals = all_tss
        else:
            # only TSS for this cancer
            tss_vals = sorted(tss_df[tss_df['cls']==c]['tss'].tolist())
        per_map = {t:i for i,t in enumerate(tss_vals)}

        for tss in tss_vals:
            dom_idx = dom_map[tss] if args.mode_union else per_map[tss]
            dom_label = {'tissue_source_site': torch.tensor([dom_idx], device=args.device)}

            latents = sample_latents(model, scheduler, args.n, args.batch_size,
                                     args.infer_steps, args.device, class_label, dom_label)
            with torch.no_grad():
                out = uni(torch.from_numpy(np.stack(latents,0)).to(args.device))
            embeds = out.cpu().numpy()

            outd = os.path.join(args.output_dir, c, str(tss))
            os.makedirs(outd, exist_ok=True)
            for i, emb in enumerate(embeds):
                np.savez_compressed(os.path.join(outd, f"{c}_{tss}_{i}.npz"), embedding=emb)
                print(f"Saved {c}_{tss}_{i}.npz in {outd}")

    # 6) embed real images
    for tag, meta in [('test', args.holdout_meta),
                      ('train','./train_metadata_df_complex.csv')]:
        out_root = os.path.join(os.getcwd(), f"embeddings_{tag}")
        embed_real_images(uni, meta, args.real_data_root, cancers, args.device, out_root)

if __name__=='__main__':
    main()
