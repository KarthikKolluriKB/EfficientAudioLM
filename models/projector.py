# models/projector.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MelProjectorLinear(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.mel_bins = getattr(model_config, "mel_size", 80)
        self.llm_dim = getattr(model_config, "llm_dim", 3072)
        self.time_stride = int(getattr(model_config, "mel_time_stride", 1))  # 1 = no downsampling
        self.linear = nn.Linear(self.mel_bins, self.llm_dim, bias=True)

    def forward(self, mel):  # mel: [B, 80, T] or [B, T, 80]
        if mel.dim() != 3:
            raise ValueError(f"mel must be 3D [B, 80, T] or [B, T, 80], got {mel.shape}")
        # Ensure [B, 80, T]
        if mel.shape[4] != self.mel_bins and mel.shape == self.mel_bins:
            mel = mel.transpose(1, 2)  # [B, 80, T]
        if mel.shape[4] != self.mel_bins:
            raise ValueError(f"Expected {self.mel_bins} mel bins on dim=1, got {mel.shape}")
        # Optional temporal average pooling along time
        if self.time_stride > 1:
            # avg_pool1d expects [B, C, T]
            mel = F.avg_pool1d(mel, kernel_size=self.time_stride, stride=self.time_stride)  # [B, 80, T’]
        # Project per time step
        x = mel.transpose(1, 2)             # [B, T’, 80]
        x = self.linear(x)                  # [B, T’, llm_dim]
        return x
