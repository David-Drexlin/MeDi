from diffusers import DDPMScheduler, AutoencoderKL
from collections import Counter

import csv
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler, ConcatDataset, Dataset, WeightedRandomSampler
from torchvision import models, transforms
import numpy as np
import torch 
from pathlib import Path

from PIL import Image
from torchvision.transforms import ToPILImage
from torchvision import transforms

from collections import Counter
from torch_fidelity import calculate_metrics

class CancerDataset(Dataset):
    def __init__(self, image_metadata_list, transform=None):
        self.image_metadata_list = image_metadata_list
        self.transform = transform

    def __len__(self):
        return len(self.image_metadata_list)

    def __getitem__(self, idx):
        img_path, metadata = self.image_metadata_list[idx]

        # Read the image using PIL
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, metadata

# used by sample without metadata loading 
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of images and metadata.

    Args:
        batch (list of tuples): Each tuple contains (image, metadata).

    Returns:
        images (torch.Tensor): Batched images tensor.
        metadata (list): List of metadata corresponding to each image.
    """
    images, metadata = zip(*batch)  # Unzip the batch into images and metadata
    images = torch.stack(images, dim=0)  # Stack images into a single tensor
    
    return images, list(metadata)  # Return images and metadata separately

def load_metadata(path=None, data_root='./TCGA'):
    if path:
        metadata_csv_path = path
    else:
        # look for all_slide_metadata.csv one level up from your data_root
        metadata_csv_path = os.path.join(os.path.dirname(data_root), 'all_slide_metadata.csv')
    return pd.read_csv(metadata_csv_path).reset_index(drop=True)


def load_dataset(holdout=None, sample_type='cancer_type', use_VAE=False, cancer_types=None, resolution=256, default=True, data_root='./TCGA'):
    
    print(f"default: {default}")

    NORM = transforms.Compose([
    transforms.Resize(resolution),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # Scales to [-1, 1]

    ])
    

    if not default: 
        alt_name='./train_metadata_df_complex.csv'
        alth_name='./holdout_metadata_df_complex.csv'
    else: 
        alt_name='./train_metadata_df.csv'
        alth_name='./holdout_metadata_df.csv'

    image_metadata_list = []
    holdout_image_metadata_list = []
    sample_type_list = []


    if not Path(alt_name).exists() or not Path(alth_name).exists(): 

        # Initialize cancer_types_set
        if cancer_types is not None:
            cancer_types_set = set(cancer_types)
        else:
            cancer_types_set = None

        metadata_csv_path = './all_slide_metadata.csv'
        metadata_df = pd.read_csv(metadata_csv_path)
        metadata_df = metadata_df.reset_index(drop=True)

        # Create mapping from project_id to cancer_type
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

        # Map project_id to cancer_type using the mapping
        metadata_df['cancer_type'] = metadata_df['project_id'].map(project_id_to_cancer_type)

        # Drop entries with NaN cancer_type (i.e., project_ids not in the mapping)
        metadata_df = metadata_df.dropna(subset=['cancer_type'])

        # Filter metadata_df based on cancer_types
        if cancer_types is not None:
            metadata_df = metadata_df[metadata_df['cancer_type'].isin(cancer_types)]
            metadata_df = metadata_df.reset_index(drop=True)

        # Debugging prints
        print(f"Number of entries in metadata_df after filtering: {len(metadata_df)}")
        print(f"Sample cancer_types in metadata_df: {metadata_df['cancer_type'].unique()}")

        # Proceed with holdout split as before
        if holdout is None:
            holdout_mask = (metadata_df.index % 4) == 0

        elif holdout[0] == "tissue_source_site" and len(holdout)==1: 
            holdout_mask = pd.Series(False, index=metadata_df.index)
            holdout_col = 'tissue_source_site'

            # Clean and prepare holdout column
            metadata_df[holdout_col].replace(["Unknown", "--", "'--'", "??", "\"--\""], 'unknown', inplace=True)
            metadata_df[holdout_col] = metadata_df[holdout_col].astype(str).str.strip().str.lower()
            metadata_df[holdout_col].fillna('unknown', inplace=True)

            # Iterate over each project_id and assign every 4th tissue_source_site to holdout
            for project_id, group in metadata_df.groupby('project_id'):
                unique_tissue_sources = sorted(group[holdout_col].unique())
                if len(unique_tissue_sources) < 4:
                    holdout_tissue_sources = unique_tissue_sources[:1]  # Assign at least one if less than 4
                else:
                    holdout_tissue_sources = unique_tissue_sources[3::4]  # Every 4th starting from index 3

                # Assign holdout_mask based on selected tissue_source_sites
                is_holdout = group[holdout_col].isin(holdout_tissue_sources)
                holdout_mask.loc[group.index] = is_holdout

                # Debugging prints
            # Verify that holdout_mask has True values
        
        else:
            metadata_df["tissue_source_site"].replace(["Unknown", "--", "'--'", "??", "'--", "\"--", "'--\""], np.nan, inplace=True)
            metadata_df["race"].replace(["Unknown", "--", "'--'", "??", "'--", "\"--", "'--\""], np.nan, inplace=True)
            metadata_df["gender"].replace(["Unknown", "--", "'--'", "??", "'--", "\"--", "'--\""], np.nan, inplace=True)

            metadata_df["tissue_source_site"].fillna('Unknown', inplace=True)       
            metadata_df["race"].fillna('Unknown', inplace=True)       
            metadata_df["gender"].fillna('Unknown', inplace=True)       

            metadata_df['combined_holdout_key'] = (
                metadata_df['tissue_source_site'].astype(str) + '_' +
                metadata_df['gender'].astype(str) + '_' +
                metadata_df['race'].astype(str)
                )
            
            holdout_col = holdout[0]
            holdout_mask = pd.Series(False, index=metadata_df.index)
            for project_id, group in metadata_df.groupby('project_id'):
                
                if default: 
                    holdout_values = group[holdout_col].unique()
                else: 
                    holdout_values = group['combined_holdout_key'].unique()

                n_holdout_values = len(holdout_values)
                n_train = int(n_holdout_values * 7 / 10)

                holdout_values_shuffled = holdout_values
                train_values = holdout_values_shuffled[:n_train]
                holdout_values_selected = holdout_values_shuffled[n_train:]

                if default:
                    is_holdout = group[holdout_col].isin(holdout_values_selected)
                else:
                    is_holdout = group['combined_holdout_key'].isin(holdout_values_selected) 

                holdout_mask.loc[group.index] = is_holdout

        # Split metadata into training and holdout sets
        train_metadata_df = metadata_df[~holdout_mask]
        holdout_metadata_df = metadata_df[holdout_mask]
        
        tss_column = "tissue_source_site"
        train_tss = set(train_metadata_df[tss_column].dropna().unique())
        test_tss = set(holdout_metadata_df[tss_column].dropna().unique())
        overlap = train_tss.intersection(test_tss)
        print(len(holdout_metadata_df))
        print(len(holdout_metadata_df["project_id"].unique()))

        if overlap:
            print(f">>> Found {len(overlap)} TSS that appear in both sets. Pushing them fully into holdout.")
            # Identify rows in train where TSS is in overlap
            overlap_mask = train_metadata_df[tss_column].isin(overlap)
            # Move those rows from train -> holdout
            overlap_df = train_metadata_df[overlap_mask]
            train_metadata_df = train_metadata_df[~overlap_mask]
            holdout_metadata_df = pd.concat([holdout_metadata_df, overlap_df], ignore_index=True)
            print("REVISED DIMENSIONS")
            print(len(holdout_metadata_df))
            print(len(holdout_metadata_df["project_id"].unique()))
        
        # Save the filtered metadata DataFrames to CSV files
        if not default: 
            alt_name='./train_metadata_df_complex.csv'
            alth_name='./holdout_metadata_df_complex.csv'

        train_metadata_df.to_csv('./train_metadata_df.csv' if default else alt_name, index=False)
        holdout_metadata_df.to_csv('./holdout_metadata_df.csv' if default else alth_name, index=False)

        train_csv_path = './train_metadata_df.csv' if default else alt_name
        test_csv_path = './holdout_metadata_df.csv' if default else alth_name

        # Call the function to check for TSS intersection
        overlapping_tss = check_tss_intersection(train_csv_path, test_csv_path)
        if overlapping_tss:
            print("\nConsider modifying your dataset split to ensure no overlapping TSS between training and holdout sets.")
            raise ValueError("Overlap found in TSS. Please fix your dataset split.")
        elif len(holdout_metadata_df["project_id"].unique()) != 32:
            print("Split is too small; adjust holdout doesnt hold 32 classes")
            raise ValueError("Split is too small; adjust holdout doesnt hold 32 classes.")
        elif len(train_metadata_df["project_id"].unique()) != 32:
            print("Split is too great; adjust train doesnt hold 32 classes")
            raise ValueError("Split is too great; adjust train doesnt hold 32 classes.")
        else:
            print("\nNo overlapping TSS detected. Your dataset split is clean.")

    else:         
        train_metadata_df = pd.read_csv(alt_name)
        holdout_metadata_df = pd.read_csv(alth_name)

    # Create 'slide_id' column in metadata_df
    train_metadata_df['slide_id'] = train_metadata_df['slide_submitter_id']
    holdout_metadata_df['slide_id'] = holdout_metadata_df['slide_submitter_id']

    # Create dictionaries for quick lookup
    metadata_dict = train_metadata_df.set_index('slide_id').to_dict(orient='index')
    holdout_metadata_dict = holdout_metadata_df.set_index('slide_id').to_dict(orient='index')
    csv_data={}
    # Collect image paths and metadata for both training and holdout datasets

    base_path = data_root
    for cancer_type in sorted(os.listdir(base_path)):
        if cancer_types is not None and cancer_type not in cancer_types:
            continue
        cancer_folder = os.path.join(base_path, cancer_type)
        if os.path.isdir(cancer_folder):
            for sample_folder in os.listdir(cancer_folder):
                sample_path = os.path.join(cancer_folder, sample_folder)
                if os.path.isdir(sample_path):
                    for image_file in os.listdir(sample_path):
                        if image_file.endswith('.jpg'):
                            image_path = os.path.join(sample_path, image_file)
                            slide_id = sample_folder  # Adjust if necessary

                            if slide_id in metadata_dict:
                                metadata = metadata_dict[slide_id]
                                image_metadata_list.append((image_path, metadata))
                                if sample_type == 'cancer_type':
                                    sample_type_list.append(metadata['cancer_type'])
                                else:
                                    sample_type_list.append(metadata[holdout])
                            elif slide_id in holdout_metadata_dict:
                                metadata = holdout_metadata_dict[slide_id]
                                holdout_image_metadata_list.append((image_path, metadata))
                            else:
                                #print(f"for cancer_type: {cancer_type} lide_id {slide_id} not found in metadata_dict")
                                csv_data[slide_id] = cancer_type
    
    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['Slide ID', 'Cancer Type'])

        # Write the rows
        for slide_id, cancer_type in csv_data.items():
            writer.writerow([slide_id, cancer_type])
                    
    # Now, compute sample_weights based on sample_type_list
    sample_type_counts = Counter(sample_type_list)
    sample_weights = []

    for sample_type_value in sample_type_list:
        num_samples_in_type = sample_type_counts[sample_type_value]
        weight_per_sample = 1.0 / num_samples_in_type
        sample_weights.append(weight_per_sample)
    
    # Optionally, normalize the sample_weights so that they sum up to 1
    total_weight = sum(sample_weights)
    sample_weights = [w / total_weight for w in sample_weights]

    # Create the datasets
    if use_VAE:
        # Instantiate the VAE
        vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="vae",
            sample_size=resolution
        )
        vae.eval()  # Set to evaluation mode
        # Decide on the device (CPU or GPU)
        device = torch.device('cuda')
        vae.to(device)
    else:
        vae = None
        device = 'cuda'  # Default to CPU if not using VAE

    # Create the datasets with the new parameters
    train_dataset = CancerDataset(
        image_metadata_list,
        transform=NORM,
        #use_VAE=use_VAE,  # Pass the flag to the dataset
        #vae=vae          # Pass the VAE instance     # Pass the device
    )
    holdout_dataset = CancerDataset(
        holdout_image_metadata_list,
        transform=NORM,
        #use_VAE=use_VAE,
        #vae=vae,
    )

    # Convert sample weights to a NumPy array
    samples_weight = np.array(sample_weights)

    # Create the sampler for the training dataset
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    return train_dataset, sampler, holdout_dataset

def gather_real_images(root_dir, valid_slide_ids):
    """
    Recursively walks through `root_dir`, gathering image paths 
    only if they reside in a folder whose name is in valid_slide_ids.
    """
    real_image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # The parent directory (i.e., the "slide_submitter_id" folder) 
                # is usually the last part of 'root'
                parent_dir_name = os.path.basename(root)
                if parent_dir_name in valid_slide_ids:
                    real_image_paths.append(os.path.join(root, filename))
    return real_image_paths

class FIDDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

def compute_fid(real_root_dir, gen_root_dir, transform=None):
    """
    Recursively walks through real_root_dir and gen_root_dir
    to build two CustomImageDatasets, then computes FID.
    """
    train_csv_path="/home/daviddrexlin/TCGA/train_metadata_df_complex.csv"
    if transform is None: 
        transform = transforms.Compose([
            transforms.Resize(128),    
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8)
        ])
    #print(f"NOTE: FID COMPUTED IN {"128" if transform is None else "256"} PIXEL SPACE")
    # --- 1) Load CSV, gather valid slide IDs
    df_train = pd.read_csv(train_csv_path)
    valid_slide_ids = set(df_train["slide_submitter_id"].unique())
    
    # --- 2) Gather real image paths (only from valid slide IDs)
    real_image_paths = gather_real_images(real_root_dir, valid_slide_ids)
    
    # --- 3) Gather generated image paths (unrestricted or however you prefer)
    gen_image_paths = []
    for root, _, files in os.walk(gen_root_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                gen_image_paths.append(os.path.join(root, filename))
    
    # --- 4) Build the Datasets
    real_dataset = FIDDataset(real_image_paths, transform=transform)
    gen_dataset  = FIDDataset(gen_image_paths, transform=transform)
    
    if len(real_dataset) == 0:
        raise ValueError(f"No real images found in {real_root_dir}")
    if len(gen_dataset) == 0:
        raise ValueError(f"No generated images found in {gen_root_dir}")

    # --- 5) Compute FID
    metrics = calculate_metrics(
        input1=real_dataset,
        input2=gen_dataset,
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
        samples_find_deep=True
    )
    return metrics['frechet_inception_distance']

def check_tss_intersection(train_csv_path, test_csv_path, tss_column='tissue_source_site'):
    """
    Check if there is any intersection in the tissue_source_site (TSS) between train and test datasets.
    
    Parameters:
        train_csv_path (str): Path to the training metadata CSV file.
        test_csv_path (str): Path to the holdout/test metadata CSV file.
        tss_column (str): The column name for tissue_source_site. Default is 'tissue_source_site'.
        
    Returns:
        overlapping_tss (set): A set of TSS that are present in both datasets.
    """
    # Load the CSV files into pandas DataFrames
    try:
        train_df = pd.read_csv(train_csv_path)
        print(f"Loaded training data from: {train_csv_path}")
    except FileNotFoundError:
        print(f"Training file not found at: {train_csv_path}")
        return
    except Exception as e:
        print(f"Error loading training file: {e}")
        return
    
    try:
        test_df = pd.read_csv(test_csv_path)
        print(f"Loaded holdout data from: {test_csv_path}")
    except FileNotFoundError:
        print(f"Holdout file not found at: {test_csv_path}")
        return
    except Exception as e:
        print(f"Error loading holdout file: {e}")
        return
    
    # Check if tss_column exists in both DataFrames
    if tss_column not in train_df.columns:
        print(f"Column '{tss_column}' not found in training DataFrame.")
        return
    if tss_column not in test_df.columns:
        print(f"Column '{tss_column}' not found in holdout DataFrame.")
        return
    
    # Get unique TSS from both DataFrames
    train_tss = set(train_df[tss_column].dropna().unique())
    test_tss = set(test_df[tss_column].dropna().unique())
    
    print(f"Number of unique TSS in training set: {len(train_tss)}")
    print(f"Number of unique TSS in holdout set: {len(test_tss)}")
    
    # Find intersection
    overlapping_tss = train_tss.intersection(test_tss)
    
    if overlapping_tss:
        print(f"\nThere are {len(overlapping_tss)} overlapping TSS between training and holdout sets:")
        for tss in overlapping_tss:
            print(f"- {tss}")
    else:
        print("\nThere are no overlapping TSS between training and holdout sets.")
    
    return overlapping_tss
