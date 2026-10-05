"""
Compare linear probe performance across MULTIPLE checkpoints (e.g. epoch 10,
20, 30, and your current best), to directly test whether extended pretraining
caused memorization/overfitting rather than continued generalization.

Logic: extract embeddings ONCE per checkpoint, using the SAME fixed sample of
images and the SAME train/test split for every checkpoint, so the only thing
that differs between runs is which checkpoint's encoder produced the embeddings.
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

from model_mae import vit_small_mae

# ---- Config ----
CHECKPOINTS = [
    ("epoch10", r"checkpoints\mae_vit_small_epoch10.pt"),
    ("epoch20", r"checkpoints\mae_vit_small_epoch20.pt"),
    ("epoch30", r"checkpoints\mae_vit_small_epoch30.pt"),
    ("best",    r"checkpoints\mae_vit_small_best.pt"),
]
FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\fire\fire_resnet-data"
NO_FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
IMG_SIZE = 512
PATCH_SIZE = 16
N_PER_CLASS = 1500
TEST_SIZE = 0.25
SEED = 42
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)


def list_images(folder, n):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(files)
    return [os.path.join(folder, f) for f in files[:n]]


def load_model(checkpoint_path):
    model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", "?")
    print(f"Loaded {checkpoint_path} -> epoch {epoch}, loss {loss}")
    return model, epoch


@torch.no_grad()
def extract_embeddings(model, image_paths, transform):
    embeddings = []
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        imgs = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)
        latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
        cls_embedding = latent[:, 0, :]
        embeddings.append(cls_embedding.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def run_linear_probe(embeddings, labels, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=TEST_SIZE, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    return acc, auc


def main():
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])

    # Fixed sample + fixed split, reused for BOTH checkpoints, so the only
    # variable between the two results is the checkpoint itself.
    print("Gathering sample images...")
    fire_paths = list_images(FIRE_DIR, N_PER_CLASS)
    no_fire_paths = list_images(NO_FIRE_DIR, N_PER_CLASS)
    all_paths = fire_paths + no_fire_paths
    labels = np.array([1] * len(fire_paths) + [0] * len(no_fire_paths))
    print(f"Using {len(all_paths)} images total ({len(fire_paths)} fire, {len(no_fire_paths)} no-fire)\n")

    results = []
    for name, ckpt_path in CHECKPOINTS:
        print(f"--- {name} ---")
        model, epoch = load_model(ckpt_path)
        embeddings = extract_embeddings(model, all_paths, transform)
        acc, auc = run_linear_probe(embeddings, labels, seed=SEED)
        results.append({"name": name, "epoch": epoch, "acc": acc, "auc": auc})
        print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}\n")
        del model
        torch.cuda.empty_cache()

    print("=" * 60)
    print("COMPARISON ACROSS ALL CHECKPOINTS")
    print("=" * 60)
    print(f"{'Checkpoint':<12} {'Epoch':<8} {'Accuracy':<10} {'AUC':<8}")
    for r in results:
        print(f"{r['name']:<12} {str(r['epoch']):<8} {r['acc']:<10.4f} {r['auc']:<8.4f}")

    print("\n" + "=" * 60)
    print("STEP-TO-STEP CHANGE (each checkpoint vs. the previous one)")
    print("=" * 60)
    for i in range(1, len(results)):
        prev, curr = results[i - 1], results[i]
        acc_diff = curr["acc"] - prev["acc"]
        auc_diff = curr["auc"] - prev["auc"]
        verdict = ("WORSE" if (acc_diff < -0.02 or auc_diff < -0.02)
                   else "better" if (acc_diff > 0.02 or auc_diff > 0.02)
                   else "~unchanged")
        print(f"{prev['name']} -> {curr['name']}: "
              f"Accuracy {acc_diff:+.4f}, AUC {auc_diff:+.4f}  [{verdict}]")

    print("\n" + "=" * 60)
    print("OVERALL: first checkpoint vs. last checkpoint")
    print("=" * 60)
    first, last = results[0], results[-1]
    acc_diff = last["acc"] - first["acc"]
    auc_diff = last["auc"] - first["auc"]
    print(f"{first['name']} (epoch {first['epoch']}) -> {last['name']} (epoch {last['epoch']}): "
          f"Accuracy {acc_diff:+.4f}, AUC {auc_diff:+.4f}")
    if acc_diff < -0.02 or auc_diff < -0.02:
        print("-> Overall trend is WORSE across training. Supports the memorization/"
              "degradation concern.")
    elif abs(acc_diff) <= 0.02 and abs(auc_diff) <= 0.02:
        print("-> Overall performance is essentially UNCHANGED. No evidence that "
              "extended training caused memorization or degradation.")
    else:
        print("-> Overall trend is BETTER across training. Extended training "
              "continued to help representation quality.")


if __name__ == "__main__":
    main()