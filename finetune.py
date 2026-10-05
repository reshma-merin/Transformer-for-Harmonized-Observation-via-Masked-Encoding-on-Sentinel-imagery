import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import Trainer, TrainingArguments

from classifier import ViTClassifier

# ---- Config ----
PRETRAINED_CHECKPOINT = r"checkpoints\mae_vit_small_best.pt"
FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\fire\fire_resnet-data"
NO_FIRE_DIR = r"C:\Users\reshmamerinthomas\Wildfire\no-fire\no-fire_resnet"
IMG_SIZE = 512
PATCH_SIZE = 16
TEST_SIZE = 0.2
SEED = 42

FREEZE_ENCODER = False   # True = linear-probe-style (only head trains, fast)
                          # False = full fine-tuning (encoder + head both train)

OUTPUT_DIR = "finetune_output"
NUM_EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 if FREEZE_ENCODER else 2e-5  # lower LR for full fine-tuning

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class ClassificationImageDataset(Dataset):
    """labels: 0 = no-fire, 1 = fire"""
    def __init__(self, samples, img_size):
        self.samples = samples  # list of (path, label)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            return self.__getitem__((idx + 1) % len(self.samples))
        return {"pixel_values": self.transform(img), "labels": label}


def build_samples(folder, label):
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    return [(os.path.join(folder, f), label) for f in files]


def stratified_split(samples, test_size, seed):
    random.seed(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)

    train, test = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n_test = int(len(items) * test_size)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs[:, 1])
    return {"accuracy": acc, "auc": auc}


def main():
    print("Gathering samples...")
    fire_samples = build_samples(FIRE_DIR, label=1)
    no_fire_samples = build_samples(NO_FIRE_DIR, label=0)
    print(f"fire: {len(fire_samples)}, no-fire: {len(no_fire_samples)}")

    all_samples = fire_samples + no_fire_samples
    train_samples, test_samples = stratified_split(all_samples, TEST_SIZE, SEED)
    print(f"train: {len(train_samples)}, test: {len(test_samples)}")

    train_dataset = ClassificationImageDataset(train_samples, IMG_SIZE)
    test_dataset = ClassificationImageDataset(test_samples, IMG_SIZE)

    print("\nBuilding model and loading pretrained encoder...")
    model = ViTClassifier(img_size=IMG_SIZE, patch_size=PATCH_SIZE, num_classes=2)
    model.load_pretrained_encoder(PRETRAINED_CHECKPOINT)

    if FREEZE_ENCODER:
        print("Freezing encoder -- only the classification head will train.")
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print("\nFinal evaluation on held-out test set:")
    results = trainer.evaluate()
    print(results)

    final_path = os.path.join(OUTPUT_DIR, "final_classifier")
    trainer.save_model(final_path)
    print(f"\nSaved fine-tuned classifier to {final_path}")


if __name__ == "__main__":
    main()