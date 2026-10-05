"""
Sanity-check the pretrained MAE by visualizing real reconstructions:
original | masked | reconstructed | reconstructed+visible-pasted-back

Run this against your best checkpoint to see whether the model is learning
genuine structure (edges, terrain texture, cloud boundaries) or just
predicting a flat mean color per patch (a known MAE degenerate shortcut,
more likely on low-complexity imagery like homogeneous forest/water scenes).
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model_mae import vit_small_mae

CHECKPOINT_PATH = "checkpoints/mae_vit_small_best.pt"
VIZ_FOLDER = r"C:\Users\reshmamerinthomas\Wildfire\pretraining\viz_recon"
IMAGE_PATHS = sorted(
    glob.glob(os.path.join(VIZ_FOLDER, "*.png")) +
    glob.glob(os.path.join(VIZ_FOLDER, "*.jpg")) +
    glob.glob(os.path.join(VIZ_FOLDER, "*.jpeg"))
)
print(f"Found {len(IMAGE_PATHS)} images in {VIZ_FOLDER}")
IMG_SIZE = 512
PATCH_SIZE = 16
MASK_RATIO = 0.75
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unpatchify(x, patch_size, in_chans=3):
    """Inverse of the model's patchify -- turns (B, N, patch*patch*C) back into (B, C, H, W)."""
    B, N, D = x.shape
    h = w = int(N ** 0.5)
    p = patch_size
    x = x.reshape(B, h, w, p, p, in_chans)
    x = torch.einsum("bhwpqc->bchpwq", x)
    imgs = x.reshape(B, in_chans, h * p, w * p)
    return imgs


def load_image(path, img_size):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (C, H, W)
    return tensor


def main():
    model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"loss {ckpt.get('loss', '?')}")

    imgs = torch.stack([load_image(p, IMG_SIZE) for p in IMAGE_PATHS]).to(device)

    with torch.no_grad():
        loss, pred, mask = model(imgs, mask_ratio=MASK_RATIO)

    print(f"Reconstruction loss on these samples: {loss.item():.4f}")

    pred_imgs = unpatchify(pred, PATCH_SIZE)  # (B, C, H, W), predicted for ALL patches

    # Build the masked-input visualization: zero out masked patches in the original
    mask_expanded = mask.unsqueeze(-1).repeat(1, 1, PATCH_SIZE * PATCH_SIZE * 3)
    mask_img = unpatchify(mask_expanded, PATCH_SIZE)  # 1 = masked, 0 = visible

    masked_input = imgs * (1 - mask_img)
    reconstruction = imgs * (1 - mask_img) + pred_imgs * mask_img  # visible patches kept real

    n = len(IMAGE_PATHS)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        orig = imgs[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        masked = masked_input[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        recon = reconstruction[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy()

        axes[i, 0].imshow(orig); axes[i, 0].set_title("Original"); axes[i, 0].axis("off")
        axes[i, 1].imshow(masked); axes[i, 1].set_title("Masked (75% hidden)"); axes[i, 1].axis("off")
        axes[i, 2].imshow(recon); axes[i, 2].set_title("Reconstructed"); axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_check.png", dpi=150)
    print("Saved reconstruction_check.png -- open it and inspect visually.")


if __name__ == "__main__":
    main()