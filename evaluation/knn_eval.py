"""
k-NN evaluation of the pretrained MAE encoder -- a standard, training-free
way to check representation quality (used in DINO, Scale-MAE, etc.):
for each test image, look at its k nearest neighbors (by embedding distance)
among the training images, and predict the majority label among them.
No classifier is trained at all -- if this works well, it means the raw
embedding space already groups same-class images close together.
"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from model_mae import vit_small_mae

# ---- Config ----
CHECKPOINT_PATH = r"checkpoints\mae_vit_small_best.pt"
FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\fire\fire_resnet-data"
NO_FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
IMG_SIZE = 512
PATCH_SIZE = 16
N_PER_CLASS = 1500
TEST_SIZE = 0.25
SEED = 42
BATCH_SIZE = 32
K_VALUES = [5, 10, 20]  # try a few k values, since the "best" k varies by dataset

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
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?')}")
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


def main():
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    print("Gathering sample images...")
    fire_paths = list_images(FIRE_DIR, N_PER_CLASS)
    no_fire_paths = list_images(NO_FIRE_DIR, N_PER_CLASS)
    all_paths = fire_paths + no_fire_paths
    labels = np.array([1] * len(fire_paths) + [0] * len(no_fire_paths))
    print(f"Using {len(all_paths)} images total\n")

    model = load_model(CHECKPOINT_PATH)
    print("Extracting embeddings...")
    embeddings = extract_embeddings(model, all_paths, transform)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )

    print(f"\n{'k':<6}{'Accuracy':<12}{'AUC':<8}")
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        probs = knn.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        print(f"{k:<6}{acc:<12.4f}{auc:<8.4f}")

    print("\nFor comparison, linear probing on this same encoder achieved ~0.93 "
          "accuracy / 0.98 AUC. k-NN typically scores a bit lower than a trained "
          "linear probe, since it makes no use of labeled training signal beyond "
          "raw distance -- that's expected, not a red flag.")


if __name__ == "__main__":
    main()