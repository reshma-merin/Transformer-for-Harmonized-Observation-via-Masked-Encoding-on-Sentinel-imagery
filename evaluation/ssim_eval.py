"""
Compute SSIM (Structural Similarity Index) for the pretrained MAE's
reconstructions -- a perceptual/structural similarity metric, 0-1,
higher is better.

Requires: pip install scikit-image --break-system-packages
"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import structural_similarity as ssim

from model_mae import vit_small_mae

# ---- Config ----
CHECKPOINT_PATH = r"checkpoints\mae_vit_small_best.pt"
FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
NO_FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
IMG_SIZE = 512
PATCH_SIZE = 16
MASK_RATIO = 0.75
N_SAMPLES = 500
SEED = 42
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
    print(f"Evaluating SSIM on {len(image_paths)} images...")

    ssim_scores = []

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

            for j in range(orig_np.shape[0]):
                orig_img = np.transpose(orig_np[j], (1, 2, 0))
                recon_img = np.transpose(recon_np[j], (1, 2, 0))
                ssim_scores.append(ssim(orig_img, recon_img, data_range=1.0, channel_axis=2))

            if (i // BATCH_SIZE) % 5 == 0:
                print(f"  Processed {i + len(batch_paths)}/{len(image_paths)}")

    arr = np.array(ssim_scores)
    print("\n--- SSIM Results ---")
    print(f"Mean: {arr.mean():.4f}")
    print(f"Std:  {arr.std():.4f}")
    print(f"Min:  {arr.min():.4f}")
    print(f"Max:  {arr.max():.4f}")


if __name__ == "__main__":
    main()