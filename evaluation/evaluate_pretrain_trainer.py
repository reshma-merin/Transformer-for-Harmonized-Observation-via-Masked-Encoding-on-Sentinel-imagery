"""
Evaluate the pretrained MAE model using Hugging Face's Trainer, rather than
a manual loop. This reports the average reconstruction loss over a sample
of images -- the same underlying metric your training logs already showed,
just computed via Trainer.evaluate() for a formal, reusable evaluation step.

Note: this measures reconstruction quality (the MAE pretraining objective),
NOT downstream classification usefulness -- for that, see evaluate_pretraining.py
(linear probe) or compare_checkpoints.py.
"""

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import Trainer, TrainingArguments

from model_mae import vit_small_mae

# ---- Config ----
CHECKPOINT_PATH = r"checkpoints\mae_vit_small_best.pt"
FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\fire\fire_resnet-data"
NO_FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
IMG_SIZE = 512
PATCH_SIZE = 16
MASK_RATIO = 0.75
N_EVAL_SAMPLES = 2000   # how many images to evaluate on
SEED = 42
BATCH_SIZE = 32

random.seed(SEED)


class MAETrainerWrapper(nn.Module):
    """Adapts MaskedAutoencoderViT's (imgs) -> (loss, pred, mask) interface
    to Trainer's expected (pixel_values=...) -> {"loss": ...} interface."""
    def __init__(self, mae_model, mask_ratio):
        super().__init__()
        self.mae_model = mae_model
        self.mask_ratio = mask_ratio

    def forward(self, pixel_values=None, labels=None):
        loss, pred, mask = self.mae_model(pixel_values, mask_ratio=self.mask_ratio)
        return {"loss": loss}


class EvalImageDataset(Dataset):
    def __init__(self, image_paths, img_size):
        self.paths = image_paths
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            return self.__getitem__((idx + 1) % len(self.paths))
        return {"pixel_values": self.transform(img), "labels": torch.tensor(0.0)}


def list_images(folder, n):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(files)
    return [os.path.join(folder, f) for f in files[:n]]


def main():
    print("Gathering evaluation sample...")
    fire_paths = list_images(FIRE_DIR, N_EVAL_SAMPLES // 2)
    no_fire_paths = list_images(NO_FIRE_DIR, N_EVAL_SAMPLES // 2)
    eval_paths = fire_paths + no_fire_paths
    print(f"Evaluating on {len(eval_paths)} images "
          f"({len(fire_paths)} fire, {len(no_fire_paths)} no-fire)")

    eval_dataset = EvalImageDataset(eval_paths, IMG_SIZE)

    print(f"\nLoading pretrained model from {CHECKPOINT_PATH}...")
    mae_model = vit_small_mae(img_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    mae_model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, "
          f"training loss {ckpt.get('loss', '?')}")

    model = MAETrainerWrapper(mae_model, mask_ratio=MASK_RATIO)

    training_args = TrainingArguments(
        output_dir="mae_eval_output",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
    )

    print("\nRunning evaluation via Trainer.evaluate()...")
    results = trainer.evaluate()
    print("\n--- Results ---")
    print(results)
    print(f"\nMean reconstruction loss on {len(eval_paths)} held-out-sampled images: "
          f"{results['eval_loss']:.4f}")


if __name__ == "__main__":
    main()