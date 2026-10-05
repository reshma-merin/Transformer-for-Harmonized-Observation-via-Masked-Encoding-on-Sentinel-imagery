"""

Two checks:
1. Linear probe: freeze the pretrained encoder, extract [CLS] embeddings,
   train ONLY a linear classifier on top. If pretraining learned good
   structure, even a simple linear model should separate fire/no-fire
   reasonably well.
2. t-SNE plot: visualize whether fire and no-fire embeddings naturally
   cluster apart in the frozen representation space.

"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model_mae import vit_small_mae

# ---- Config ----
CHECKPOINT_PATH = "checkpoints/mae_vit_small_best.pt"
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
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png"))]
    random.shuffle(files)
    return [os.path.join(folder, f) for f in files[:n]]


def load_model():
    model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
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

        # mask_ratio=0.0 -> encoder sees ALL patches, no masking.
   
        latent, _, _ = model.forward_encoder(imgs, mask_ratio=0.0)
        cls_embedding = latent[:, 0, :]  # (B, embed_dim) -- the [CLS] token output
        embeddings.append(cls_embedding.cpu().numpy())

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Extracted {i + len(batch_paths)}/{len(image_paths)}")

    return np.concatenate(embeddings, axis=0)


def main():
    model = load_model()

    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ])

    print("\nGathering sample images...")
    fire_paths = list_images(FIRE_DIR, N_PER_CLASS)
    no_fire_paths = list_images(NO_FIRE_DIR, N_PER_CLASS)
    print(f"fire: {len(fire_paths)}, no-fire: {len(no_fire_paths)}")

    all_paths = fire_paths + no_fire_paths
    labels = np.array([1] * len(fire_paths) + [0] * len(no_fire_paths))

    print("\nExtracting frozen encoder embeddings...")
    embeddings = extract_embeddings(model, all_paths, transform)
    print(f"Embeddings shape: {embeddings.shape}")

    # ---- Linear probe ----
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )

    print("\nTraining linear probe (logistic regression on frozen embeddings)...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print(f"\n--- Linear Probe Results ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["no-fire", "fire"]))

    # ---- t-SNE visualization ----
    print("\nRunning t-SNE (this can take a minute)...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    embed_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    plt.scatter(embed_2d[labels == 0, 0], embed_2d[labels == 0, 1],
                c="tab:blue", label="no-fire", alpha=0.5, s=10)
    plt.scatter(embed_2d[labels == 1, 0], embed_2d[labels == 1, 1],
                c="tab:red", label="fire", alpha=0.5, s=10)
    plt.legend()
    plt.title("t-SNE of frozen MAE encoder embeddings")
    plt.savefig("embedding_tsne.png", dpi=150)
    print("Saved embedding_tsne.png")


if __name__ == "__main__":
    main()