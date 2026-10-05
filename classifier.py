"""
Wraps the pretrained MAE ViT-Small encoder with a classification head, in a
form compatible with transformers.Trainer (forward() accepts pixel_values +
labels, returns a dict with 'loss' and 'logits').

The MAE decoder is intentionally NOT loaded here -- only the encoder weights
transfer into fine-tuning, per how MAE pretraining is meant to be used.
"""

import torch
import torch.nn as nn

from model_mae import PatchEmbed, TransformerBlock, get_2d_sincos_pos_embed


class ViTClassifier(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=384,
                 depth=12, num_heads=6, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer(
            "pos_embed",
            torch.cat([torch.zeros(1, embed_dim), torch.from_numpy(pos_embed).float()], dim=0).unsqueeze(0),
            persistent=False,
        )

        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # [CLS] token output

    def forward(self, pixel_values=None, labels=None):
        features = self.forward_features(pixel_values)
        logits = self.head(features)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

    def load_pretrained_encoder(self, checkpoint_path, device="cpu"):
        """Load ONLY the encoder weights from an MAE pretraining checkpoint.
        The decoder (decoder_embed, mask_token, decoder_blocks, etc.) is
        intentionally skipped -- it has no role after pretraining."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"]

        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("encoder_blocks"):
                new_state[k.replace("encoder_blocks", "blocks")] = v
            elif k.startswith("encoder_norm"):
                new_state[k.replace("encoder_norm", "norm")] = v
            elif k.startswith("patch_embed") or k == "cls_token":
                new_state[k] = v
            # decoder_*, mask_token, decoder_pos_embed: deliberately skipped

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        print(f"Loaded pretrained encoder from {checkpoint_path} "
              f"(epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?')})")
        print(f"  Missing keys (expected -- these are the new classification head): {missing}")
        if unexpected:
            print(f"  WARNING -- unexpected keys not loaded: {unexpected}")