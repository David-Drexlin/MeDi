## MeDi

**Metadata-Guided Diffusion Models for Histopathology**

MeDi is an end-to-end pipeline that:

1. **Trains** a diffusion UNet conditioned on both cancer subtype _and_ tissue-source-site (TSS).
2. **Generates** synthetic histopathology patches and **embeds** them—alongside real slides—via the UNI foundation model.
3. **Evaluates** downstream classifiers under realistic subpopulation shifts by linear probing in embedding space.

By explicitly modeling metadata, MeDi helps mitigate “shortcut” biases (e.g., hospital-specific staining) and improves robustness to unseen medical centers.

---

## 1. Repository Structure
MeDi/
├── Scripts/
│ ├── train_diffusion.sh # wrapper for train_diffusion.py
│ ├── sample.sh # wrapper for sample.py
│ ├── embed.sh # wrapper for embed.py
│ └── train_linear.sh # wrapper for train_linear.py
├── train_diffusion.py # diffusion model training
├── sample.py # latent sampling & raw image generation
├── embed.py # embed real + synthetic with UNI
├── train_linear.py # downstream linear‐probe evaluation
├── unet.py # UNet2DModel wrapper & loading
├── load_TCGA.py # TCGA metadata loader
└── requirements.txt # pip dependencies

- **'train_diffusion.py'**  
  Trains a 2D UNet diffusion model.  
  - Condition on `cancer_type` only (CLS) or on both `cancer_type` + `tissue_source_site` (MeDi).  
  - Supports “union” vs. “separate” TSS embedding modes.
  key settings:
  - FID_tracker: governs after how many OS a lightweight FID inference is computed
  - data_root: you must pass the path to the TCGA dataset here (not provided through repo)

- **'sample.py'**  
  Generates raw image latents (no embedding). Useful for FID or visual comparisons.

- **'embed.py'**  
  Uses the pretrained UNI model to extract embeddings from both real and synthetic images.  
  - Saves compressed `.npz` per slide for downstream use.
  key settings:
  - mode_union: if passed the union of TSS between the provided cancer types is sampled e.g. A with 1,2 and B with 4,5 will yield with mode_union synth: A: 1,2,4,5 and B: 1,2,4,5
  - n: number of synth images per cancer-tss pair 

- **'train_linear.py'**  
  Runs logistic regression (balanced) on embeddings.  
  - Reports overall balanced accuracy, TSS-averaged accuracy, and MCC.
  key settings:  
  - ratio: synthetic / real examples per class (e.g. 1.0 = equal).
  - max_real: cap number of real samples to simulate low‐data regimes.
  - sweep_number_tss: for each seed, train on k randomly chosen TSS codes -> chosen from the real dataset.

- **'unet.py'**, **'load_TCGA.py'**  
  Helper modules for model setup and metadata loading.

---

## 2. Installation

git clone https://github.com/David-Drexlin/MeDi.git
cd MeDi
pip install -r requirements.txt

Data:
Download TCGA‐UT patches yourself (https://www.cancer.gov/ccg/research/genome-sequencing/tcga)


