import torch
from classifier import ViTClassifier

model = ViTClassifier(img_size=512, patch_size=16, num_classes=2)
x = torch.randn(2, 3, 512, 512)
labels = torch.tensor([0, 1])
out = model(pixel_values=x, labels=labels)
print("loss:", out["loss"].item())
print("logits shape:", out["logits"].shape)