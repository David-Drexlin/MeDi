#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm

from models.medi import MeDiGenerator
from models.cls_only import ClsOnlyGenerator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',      required=True, choices=['rcc','uterine','nsclc'])
    p.add_argument('--method',    required=True, choices=['medi','cls_only'])
    p.add_argument('--gen-config', required=True,
                   help="CSV with columns [cls, tss, count]")
    p.add_argument('--output-base', default='generated_images',
                   help="Base dir for synthetic outputs")
    p.add_argument('--augment',    action='store_true',
                   help="Apply random augment before embedding")
    return p.parse_args()

def get_generator(method):
    if method=='medi':
        ckpt = '/home/space/datasets/tcga_uniform/deep_TSS_only_concatembed/'\
               'checkpoint-800000/model.safetensors'
        return MeDiGenerator(model_path=ckpt)
    else:
        ckpt = '/home/space/datasets/tcga_uniform/deep_class_only/'\
               'checkpoint-800000/model.safetensors'
        return ClsOnlyGenerator(model_path=ckpt)

def generate_images(df, generator, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # make one subfolder per class
    for cls in df['cls'].unique():
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    idx = 0
    bs_default = 32

    for _, row in tqdm(df.iterrows(), total=len(df)):
        cls, tss, cnt = row['cls'], row['tss'], int(row['count'])
        generated = 0
        images = []
        while generated < cnt:
            bs = min(bs_default, cnt - generated)
            generated += bs
            if generator.__class__.__name__ == 'MeDiGenerator':
                imgs = generator.generate(
                    class_index=generator.cancer_type_mapping[cls],
                    tss=tss,
                    current_batch_size=bs
                )
            else:
                imgs = generator.generate(
                    class_index=generator.cancer_type_mapping[cls],
                    current_batch_size=bs
                )
            images.extend(imgs)

        for img in images:
            fname = f"{str(idx).zfill(5)}_{tss}.jpg"
            img.save(os.path.join(output_dir, cls, fname))
            idx += 1

class SyntheticImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.label2idx = {}
        for i, cls in enumerate(sorted(os.listdir(root_dir))):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            self.label2idx[cls] = i
            for f in os.listdir(cls_dir):
                if f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(cls_dir, f), i))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def embed_dataset(dataset, model, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, labs in tqdm(loader):
            imgs = imgs.to(device)
            reps = model(imgs)
            embeddings.append(reps.cpu().numpy())
            labels.append(labs.numpy())
    return np.concatenate(embeddings,axis=0), np.concatenate(labels,axis=0)

def main():
    args = parse_args()

    # 1) Synthetic generation
    cfg = pd.read_csv(args.gen_config)
    out_synth = os.path.join(args.output_base, args.task, args.method)
    os.makedirs(out_synth, exist_ok=True)

    gen = get_generator(args.method)
    print("→ Generating synthetic images:", len(cfg), "rows")
    generate_images(cfg, gen, out_synth)

    # 2) Build transform for embedding
    if args.augment:
        tf = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomResizedCrop(128, scale=(0.33,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4,0.4,0.2,0.1),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    # 3) Embed
    print("→ Building dataset from", out_synth)
    ds = SyntheticImageDataset(out_synth, transform=tf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("→ Loading Uni backbone for embedding")
    backbone = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )

    print("→ Embedding", len(ds), "images")
    embs, labs = embed_dataset(ds, backbone, device)

    # 4) Save
    emb_dir = os.path.join(os.path.dirname(args.gen_config), 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    aug_tag = '_augment' if args.augment else ''
    out_file = os.path.join(
        emb_dir,
        f"{args.method}_{args.task}{aug_tag}.npz"
    )
    np.savez_compressed(out_file, embeddings=embs, labels=labs)
    print("→ Saved embeddings to", out_file)

if __name__=="__main__":
    main()
#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm

from models.medi import MeDiGenerator
from models.cls_only import ClsOnlyGenerator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',      required=True, choices=['rcc','uterine','nsclc'])
    p.add_argument('--method',    required=True, choices=['medi','cls_only'])
    p.add_argument('--gen-config', required=True,
                   help="CSV with columns [cls, tss, count]")
    p.add_argument('--output-base', default='generated_images',
                   help="Base dir for synthetic outputs")
    p.add_argument('--augment',    action='store_true',
                   help="Apply random augment before embedding")
    return p.parse_args()

def get_generator(method):
    if method=='medi':
        ckpt = '/home/space/datasets/tcga_uniform/deep_TSS_only_concatembed/'\
               'checkpoint-800000/model.safetensors'
        return MeDiGenerator(model_path=ckpt)
    else:
        ckpt = '/home/space/datasets/tcga_uniform/deep_class_only/'\
               'checkpoint-800000/model.safetensors'
        return ClsOnlyGenerator(model_path=ckpt)

def generate_images(df, generator, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # make one subfolder per class
    for cls in df['cls'].unique():
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    idx = 0
    bs_default = 32

    for _, row in tqdm(df.iterrows(), total=len(df)):
        cls, tss, cnt = row['cls'], row['tss'], int(row['count'])
        generated = 0
        images = []
        while generated < cnt:
            bs = min(bs_default, cnt - generated)
            generated += bs
            if generator.__class__.__name__ == 'MeDiGenerator':
                imgs = generator.generate(
                    class_index=generator.cancer_type_mapping[cls],
                    tss=tss,
                    current_batch_size=bs
                )
            else:
                imgs = generator.generate(
                    class_index=generator.cancer_type_mapping[cls],
                    current_batch_size=bs
                )
            images.extend(imgs)

        for img in images:
            fname = f"{str(idx).zfill(5)}_{tss}.jpg"
            img.save(os.path.join(output_dir, cls, fname))
            idx += 1

class SyntheticImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.label2idx = {}
        for i, cls in enumerate(sorted(os.listdir(root_dir))):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            self.label2idx[cls] = i
            for f in os.listdir(cls_dir):
                if f.lower().endswith('.jpg'):
                    self.samples.append((os.path.join(cls_dir, f), i))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def embed_dataset(dataset, model, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device).eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, labs in tqdm(loader):
            imgs = imgs.to(device)
            reps = model(imgs)
            embeddings.append(reps.cpu().numpy())
            labels.append(labs.numpy())
    return np.concatenate(embeddings,axis=0), np.concatenate(labels,axis=0)

def main():
    args = parse_args()

    # 1) Synthetic generation
    cfg = pd.read_csv(args.gen_config)
    out_synth = os.path.join(args.output_base, args.task, args.method)
    os.makedirs(out_synth, exist_ok=True)

    gen = get_generator(args.method)
    print("→ Generating synthetic images:", len(cfg), "rows")
    generate_images(cfg, gen, out_synth)

    # 2) Build transform for embedding
    if args.augment:
        tf = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.RandomResizedCrop(128, scale=(0.33,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4,0.4,0.2,0.1),
            transforms.RandomGrayscale(0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    # 3) Embed
    print("→ Building dataset from", out_synth)
    ds = SyntheticImageDataset(out_synth, transform=tf)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("→ Loading Uni backbone for embedding")
    backbone = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True
    )

    print("→ Embedding", len(ds), "images")
    embs, labs = embed_dataset(ds, backbone, device)

    # 4) Save
    emb_dir = os.path.join(os.path.dirname(args.gen_config), 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    aug_tag = '_augment' if args.augment else ''
    out_file = os.path.join(
        emb_dir,
        f"{args.method}_{args.task}{aug_tag}.npz"
    )
    np.savez_compressed(out_file, embeddings=embs, labels=labs)
    print("→ Saved embeddings to", out_file)

if __name__=="__main__":
    main()

