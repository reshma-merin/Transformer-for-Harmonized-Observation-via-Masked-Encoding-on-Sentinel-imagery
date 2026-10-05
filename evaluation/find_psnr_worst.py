"""
Find the worst-scoring (lowest PSNR) reconstructions from a sample, and
visualize them -- original vs. masked vs. reconstructed -- to check whether
poor scores come from genuinely hard imagery (e.g. unusual cloud cover) or
from a data quality issue (corrupted/mismatched image).

Requires: pip install scikit-image matplotlib --break-system-packages
"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

from model_mae import vit_small_mae

# ---- Config ----
CHECKPOINT_PATH = r"checkpoints\mae_vit_small_best.pt"
FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\fire\fire_resnet-data"
NO_FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
IMG_SIZE = 512
PATCH_SIZE = 16
MASK_RATIO = 0.75
N_SAMPLES = 500          # same sample size as your psnr_eval.py run
N_WORST = 5              # how many worst cases to visualize
SEED = 42                # same seed -> same 500-image sample as before
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)


def unpatchify(x, patch_size, in_chans=3):
    B, N, D = x.shape
    h = w = int(N ** 0.5)
    p = patch_size
    x = x.reshape(B, h, w, p, p, in_chans)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(B, in_chans, h * p, w * p)


def list_images(folders, n):
    paths = []
    for folder in folders:
        files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        paths.extend(files)
    random.shuffle(paths)
    return paths[:n]


def load_model():
    model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?')}")
    return model


def main():
    model = load_model()
    transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    image_paths = list_images([FIRE_DIR, NO_FIRE_DIR], N_SAMPLES)
    print(f"Scoring {len(image_paths)} images to find the {N_WORST} worst PSNR cases...")

    scored = []  # (path, psnr_score)

    with torch.no_grad():
        for i in range(0, len(image_paths), BATCH_SIZE):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            imgs = torch.stack([
                transform(Image.open(p).convert("RGB")) for p in batch_paths
            ]).to(device)

            loss, pred, mask = model(imgs, mask_ratio=MASK_RATIO)
            pred_imgs = unpatchify(pred, PATCH_SIZE)

            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, PATCH_SIZE * PATCH_SIZE * 3)
            mask_img = unpatchify(mask_expanded, PATCH_SIZE)
            reconstruction = imgs * (1 - mask_img) + pred_imgs * mask_img

            orig_np = imgs.cpu().numpy()
            recon_np = reconstruction.cpu().clamp(0, 1).numpy()

            for j, path in enumerate(batch_paths):
                orig_img = np.transpose(orig_np[j], (1, 2, 0))
                recon_img = np.transpose(recon_np[j], (1, 2, 0))
                score = psnr(orig_img, recon_img, data_range=1.0)
                scored.append((path, score))

            if (i // BATCH_SIZE) % 5 == 0:
                print(f"  Processed {i + len(batch_paths)}/{len(image_paths)}")

    scored.sort(key=lambda x: x[1])  # ascending -> worst first
    worst = scored[:N_WORST]

    print(f"\n--- {N_WORST} Worst-Scoring Images ---")
    for path, score in worst:
        print(f"  {score:.2f} dB  --  {os.path.basename(path)}")

    # Re-run just the worst images through the model to build a visualization
    worst_paths = [p for p, _ in worst]
    imgs = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in worst_paths
    ]).to(device)

    with torch.no_grad():
        loss, pred, mask = model(imgs, mask_ratio=MASK_RATIO)

    pred_imgs = unpatchify(pred, PATCH_SIZE)
    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, PATCH_SIZE * PATCH_SIZE * 3)
    mask_img = unpatchify(mask_expanded, PATCH_SIZE)
    masked_input = imgs * (1 - mask_img)
    reconstruction = imgs * (1 - mask_img) + pred_imgs * mask_img

    n = len(worst_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        orig = imgs[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        masked = masked_input[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        recon = reconstruction[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        fname = os.path.basename(worst_paths[i])
        score = worst[i][1]

        axes[i, 0].imshow(orig); axes[i, 0].set_title(f"Original\n{fname}"); axes[i, 0].axis("off")
        axes[i, 1].imshow(masked); axes[i, 1].set_title("Masked (75% hidden)"); axes[i, 1].axis("off")
        axes[i, 2].imshow(recon); axes[i, 2].set_title(f"Reconstructed\nPSNR={score:.2f} dB"); axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("worst_psnr_cases.png", dpi=150)
    print("\nSaved worst_psnr_cases.png -- open it and inspect visually.")


if __name__ == "__main__":
    main()