#!/usr/bin/env python3
import os
import glob
import argparse
import random
import math
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

def parse_args():
    p = argparse.ArgumentParser(
        description="Linear‐probe logistic regression on real + synthetic embeddings."
    )
    p.add_argument('--cancer_types', type=str, nargs='+', default=None,
                   help='If set, only train/test on this subset of cancer types (default: all)')

    p.add_argument('--ratio', type=float, default=1.0,
                   help='#synthetic per real (e.g. 1.0 = equal)')
    p.add_argument('--max_real', type=int, default=None,
                   help='Maximum number of real examples to use per class.')
    p.add_argument('--seed', type=int, default=1,
                   help='number of independent runs (seeds 0..seed-1)')
    p.add_argument('--iters', type=int, default=100,
        help='number of solver-iterations)')

    p.add_argument('--sweep_number_tss', type=int, default=None,
            help='If set to k, for each seed randomly sample k TSS codes for training (test unchanged)')

    return p.parse_args()

def gather(emb_dir, classes, two_levels=False):
    """Return dict[class] → list of (embedding, filepath)."""
    out = {}
    for c in classes:
        if two_levels:
            pattern = os.path.join(emb_dir, c, '*', '*.npz')
        else:
            pattern = os.path.join(emb_dir, c, '*.npz')
        files = sorted(glob.glob(pattern))
        arrs = []
        for f in files:
            arrs.append((np.load(f)['embedding'], f))
        out[c] = arrs
    return out

def load_holdout_tss(holdout_csv):
    df = pd.read_csv(holdout_csv)
    df = df[['slide_submitter_id','tissue_source_site']].dropna().drop_duplicates()
    return dict(zip(df['slide_submitter_id'], df['tissue_source_site']))

def main():
    args = parse_args()

    # hard-coded dirs & files
    train_dir    = './embeddings_train'
    synth_dir    = './embeddings'
    test_dir     = './embeddings_test'
    holdout_meta = './holdout_metadata_df_complex.csv'

    # 1) infer classes from train_dir
    all_classes = sorted(d for d in os.listdir(train_dir)
                         if os.path.isdir(os.path.join(train_dir, d)))
    # 2) restrict to --cancer_types if provided
    classes = all_classes if args.cancer_types is None \
              else [c for c in all_classes if c in args.cancer_types]

    # 3) gather all embeddings up-front
    train_real_all  = gather(train_dir,   classes, two_levels=False)
    train_synth_all = gather(synth_dir,   classes, two_levels=True)
    test_data_all   = gather(test_dir,    classes, two_levels=False)
    # drop any class with no test examples
    classes = [c for c in classes if len(test_data_all[c]) > 0]

    # build list of all TSS codes present in train_real_all (safe split)
    real_tss = set()
    for cls, arr in train_real_all.items():
        for _, f in arr:
            base   = os.path.basename(f)           # e.g. "TCGA-OR-A5J1-01Z-00-DX1_0.npz"
            prefix = base.split('_', 1)[0]         # "TCGA-OR-A5J1-01Z-00-DX1"
            parts  = prefix.split('-')
            if len(parts) >= 2:
                real_tss.add(parts[1])

    # synth: directory name one level above the npz
    synth_tss = {
        f.split(os.sep)[-2]
        for cls in classes for _,f in train_synth_all[cls]
    }

    all_tss = sorted(real_tss.intersection(synth_tss))

    # load per-slide TSS lookup for test grouping
    sid2tss = load_holdout_tss(holdout_meta)

    results = []
    for seed in range(args.seed):
        random.seed(seed)

        # if sweep over TSS is requested
        if args.sweep_number_tss is not None:
            k = args.sweep_number_tss
            chosen_tss = random.sample(all_tss, k)
            print(f"[seed {seed}] training only on TSS subset: {chosen_tss}")

            # filter real embeddings by extracted TSS
            train_real = {
                cls: [
                    (emb,f) for emb,f in train_real_all[cls]
                    if os.path.basename(f).split('_',1)[0].split('-')[1] in chosen_tss
                ]
                for cls in classes
            }
            # filter synth embedded by parent directory name
            train_synth = {
                cls: [
                    (emb,f) for emb,f in train_synth_all[cls]
                    if f.split(os.sep)[-2] in chosen_tss
                ]
                for cls in classes
            }
        else:
            chosen_tss = None
            train_real  = train_real_all
            train_synth = train_synth_all

        # test set remains unchanged
        test_data = test_data_all

        # --- build train set ---
        X_train, y_train = [], []
        for cls in classes:
            real = train_real[cls]
            if args.max_real is not None:
                if len(real) < args.max_real:
                    print(f"ERROR: class {cls} has only {len(real)} real samples, "
                          f"but --max_real={args.max_real}", file=sys.stderr)
                    sys.exit(1)
                real = random.sample(real, args.max_real)

            synth = train_synth[cls]
            n_real  = len(real)
            n_synth = int(args.ratio * n_real)
            if len(synth) < n_synth:
                print(f"ERROR: class {cls} has only {len(synth)} synth samples, "
                      f"cannot draw {n_synth}", file=sys.stderr)
                sys.exit(1)
            picks = random.sample(synth, n_synth)

            for emb,_ in real:
                X_train.append(emb); y_train.append(cls)
            for emb,_ in picks:
                X_train.append(emb); y_train.append(cls)

        X_train = np.stack(X_train)
        y_map   = {c:i for i,c in enumerate(classes)}
        y_train = np.array([y_map[c] for c in y_train])

        # --- build test set + group by TSS ---
        X_test, y_test, tss_groups = [], [], []
        for cls in classes:
            for emb,fpath in test_data[cls]:
                X_test.append(emb)
                y_test.append(y_map[cls])
                sid = os.path.basename(fpath).split('_')[0]
                tss_groups.append(sid2tss.get(sid, 'Unknown'))
        X_test = np.stack(X_test)
        y_test = np.array(y_test)

        # --- train & evaluate ---
        clf   = LogisticRegression(max_iter=args.iters, class_weight='balanced')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        overall = balanced_accuracy_score(y_test, preds)
        mcc     = matthews_corrcoef(y_test, preds)
        tss_accs = []
        for tss in set(tss_groups):
            idxs = [i for i,g in enumerate(tss_groups) if g == tss]
            if not idxs: continue
            tss_accs.append(balanced_accuracy_score(y_test[idxs], preds[idxs]))
        tss_avg = float(np.mean(tss_accs)) if tss_accs else float('nan')

        results.append({
            'seed':        seed,
            'tss_subset':  chosen_tss,
            'overall_bal': overall,
            'tss_avg_bal': tss_avg,
            'mcc':         mcc
        })
        print(f"[seed {seed}] overall={overall:.4f}, tss_avg={tss_avg:.4f}, mcc={mcc:.4f}")

    # --- summary ---
    df = pd.DataFrame(results)
    def mean_sem(col):
        m = df[col].mean()
        s = df[col].std(ddof=1) / math.sqrt(len(df))
        return m, s

    mean_ov, sem_ov   = mean_sem('overall_bal')
    mean_tss, sem_tss = mean_sem('tss_avg_bal')
    mean_mcc, sem_mcc = mean_sem('mcc')

    print(f"\n== Summary over {args.seed} runs ==")
    print(f"Overall    Bal Acc: {mean_ov:.3f} ± {sem_ov:.3f}")
    print(f"TSS‐Avg Bal Acc:   {mean_tss:.3f} ± {sem_tss:.3f}")
    print(f"MCC:               {mean_mcc:.3f} ± {sem_mcc:.3f}")

    df.to_csv('results.csv', index=False)
    summary = pd.DataFrame({
        'metric': ['overall_bal','tss_avg_bal','mcc'],
        'mean':   [mean_ov, mean_tss, mean_mcc],
        'sem':    [sem_ov, sem_tss, sem_mcc]
    })
    summary.to_csv('summary.csv', index=False)
    print("Results saved to results.csv and summary.csv")

if __name__=='__main__':
    main()
