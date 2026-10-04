from __future__ import annotations

import torch
from torch import nn
from torchvision import models

__all__ = ["LipReadingModel"]


class LipReadingModel(nn.Module):
    """MobileNetV2 frame encoder + bidirectional GRU temporal aggregator."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 512,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = backbone.classifier[-1].in_features  # type: ignore[union-attr]

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``(batch, seq, C, H, W)`` frames into ``(batch, seq, feature_dim)``."""
        batch_size, seq_len, channels, height, width = frames.shape

        flat = frames.reshape(batch_size * seq_len, channels, height, width)
        features = self.features(flat)
        features = self.pool(features)
        features = torch.flatten(features, 1)

        return features.reshape(batch_size, seq_len, -1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        sequence = self.encode_frames(frames)
        output, _ = self.gru(sequence)
        last_hidden = output[:, -1, :]
        return self.classifier(self.dropout(last_hidden))
