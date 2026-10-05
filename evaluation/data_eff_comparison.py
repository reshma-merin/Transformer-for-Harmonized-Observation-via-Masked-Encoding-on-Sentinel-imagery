"""
Data-efficiency comparison: does MAE pretraining let you reach good
classification accuracy with LESS labeled data than starting from random
weights? This is the fast, linear-probe-based equivalent of the
data-efficiency experiment in Goulao & Oliveira's paper (which used full RL
training -- infeasible to replicate at that scale here, so we isolate the
same underlying question -- pretrained vs random features -- using frozen
encoders + linear probing at increasing labeled-data fractions).

Two encoders are compared, both FROZEN (no training of the backbone itself):
  1. PRETRAINED -- your MAE-pretrained ViT-Small encoder
  2. RANDOM     -- the exact same architecture, but with untrained,
                    randomly-initialized weights (a standard SSL baseline)

For each, a linear probe (logistic regression) is trained on increasing
fractions of the labeled training data, and evaluated on the SAME held-out
test set every time -- isolating the effect of pretraining from the effect
of data quantity.
"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

from model_mae import vit_small_mae

# ---- Config ----
CHECKPOINT_PATH = r"checkpoints\mae_vit_small_best.pt"
FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\fire\fire_resnet-data"
NO_FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
IMG_SIZE = 512
PATCH_SIZE = 16
N_PER_CLASS = 2000          # larger sample here, so small fractions still have enough images
TEST_SIZE = 0.25
SEED = 42
BATCH_SIZE = 32
TRAIN_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def list_images(folder, n):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(files)
    return [os.path.join(folder, f) for f in files[:n]]


def load_pretrained_model():
    model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded PRETRAINED checkpoint: epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?')}")
    return model


def build_random_model():
    model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE)  # fresh, untrained weights
    model.to(device)
    model.eval()
    print("Built RANDOM (untrained) baseline encoder -- same architecture, no pretraining.")
    return model


@torch.no_grad()
def extract_embeddings(model, image_paths, transform):
    embeddings = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        imgs = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)
        latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
        embeddings.append(latent[:, 0, :].cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_at_fraction(X_train_full, y_train_full, X_test, y_test, fraction, seed):
    n = max(2, int(len(X_train_full) * fraction))  # need at least 2 samples for both classes
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_train_full), size=n, replace=False)
    X_sub, y_sub = X_train_full[idx], y_train_full[idx]

    if len(np.unique(y_sub)) < 2:
        return None  # can't fit a classifier with only one class present

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_sub, y_sub)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    return accuracy_score(y_test, preds), roc_auc_score(y_test, probs), n


def main():
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    print("Gathering sample images...")
    fire_paths = list_images(FIRE_DIR, N_PER_CLASS)
    no_fire_paths = list_images(NO_FIRE_DIR, N_PER_CLASS)
    all_paths = fire_paths + no_fire_paths
    labels = np.array([1] * len(fire_paths) + [0] * len(no_fire_paths))
    print(f"Using {len(all_paths)} images total\n")

    results = {}
    for name, model_fn in [("pretrained", load_pretrained_model), ("random", build_random_model)]:
        print(f"\n--- {name.upper()} encoder ---")
        model = model_fn()
        embeddings = extract_embeddings(model, all_paths, transform)
        del model
        torch.cuda.empty_cache()

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
        )

        results[name] = []
        print(f"{'Fraction':<10}{'N train':<10}{'Accuracy':<12}{'AUC':<8}")
        for frac in TRAIN_FRACTIONS:
            r = run_at_fraction(X_train, y_train, X_test, y_test, frac, seed=SEED)
            if r is None:
                print(f"{frac:<10}skipped (not enough samples for both classes)")
                continue
            acc, auc, n = r
            results[name].append({"fraction": frac, "n": n, "acc": acc, "auc": auc})
            print(f"{frac:<10}{n:<10}{acc:<12.4f}{auc:<8.4f}")

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    for name, marker in [("pretrained", "o"), ("random", "s")]:
        fracs = [r["fraction"] * 100 for r in results[name]]
        aucs = [r["auc"] for r in results[name]]
        plt.plot(fracs, aucs, marker=marker, label=name)
    plt.xscale("log")
    plt.xlabel("% of labeled training data used")
    plt.ylabel("AUC on held-out test set")
    plt.title("Data efficiency: pretrained vs. randomly-initialized encoder")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("data_efficiency_comparison.png", dpi=150)
    print("\nSaved data_efficiency_comparison.png")


if __name__ == "__main__":
    main()