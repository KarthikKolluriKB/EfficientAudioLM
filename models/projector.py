# models/projector.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelProjectorLinear(nn.Module):
    """
    Direct Mel -> token projector for encoder-free ASR.

    Expects Mel from the dataloader as [B, T, 80] (time-major with 80 bins),
    but also accepts [B, 80, T] and will transpose automatically.
    Optionally downsamples time with average pooling before a per-frame Linear
    mapping to the LLM embedding dimension.

    Args (from model_config via getattr):
      - mel_size: int, number of Mel bins (default 80)
      - llm_dim: int, LLM embedding dimension (e.g., 3072)
      - mel_time_stride: int, temporal avg-pool stride (1 = no downsampling)
      - mel_input_norm: bool, apply LayerNorm over Mel bins before Linear
      - mel_dropout: float, dropout after Linear (default 0.0)
    """

    def __init__(self, model_config):
        super().__init__()
        self.mel_bins = int(getattr(model_config, "mel_size", 80))
        self.llm_dim = int(getattr(model_config, "llm_dim", 3072))
        self.time_stride = int(getattr(model_config, "mel_time_stride", 1))
        self.use_input_norm = bool(getattr(model_config, "mel_input_norm", False))
        self.dropout_p = float(getattr(model_config, "mel_dropout", 0.0))

        # Optional normalization over Mel bins per time step
        self.in_norm = nn.LayerNorm(self.mel_bins) if self.use_input_norm else None

        # Per-frame/patch projection to LLM embedding space
        self.linear = nn.Linear(self.mel_bins, self.llm_dim, bias=True)

        # Optional dropout after projection
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()

    def _ensure_bct(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Ensure Mel is [B, 80, T] for pooling and internal consistency.
        Accepts [B, T, 80] or [B, 80, T].
        """
        if mel.dim() != 3:
            raise ValueError(f"mel must be 3D [B, T, {self.mel_bins}] or [B, {self.mel_bins}, T], got {mel.shape}")

        # If last dim is mel_bins -> [B, T, 80] -> transpose to [B, 80, T]
        if mel.shape[-1] == self.mel_bins and mel.shape[1] != self.mel_bins:
            mel = mel.transpose(1, 2)  # [B, 80, T]
        elif mel.shape[1] == self.mel_bins:
            # already [B, 80, T]
            pass
        else:
            raise ValueError(f"Expected Mel bins == {self.mel_bins} on dim=1 or dim=2, got {mel.shape}")

        return mel

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
          mel: [B, T, 80] or [B, 80, T]
        Returns:
          tokens: [B, T', llm_dim] where T' = ceil(T / time_stride)
        """
        # 1) Ensure [B, 80, T]
        mel = self._ensure_bct(mel)  # [B, 80, T]

        # 2) Optional temporal downsampling (average pooling over time)
        if self.time_stride > 1:
            mel = F.avg_pool1d(mel, kernel_size=self.time_stride, stride=self.time_stride)  # [B, 80, T']

        # 3) Match projector weight dtype to avoid matmul dtype errors
        proj_dtype = next(self.linear.parameters()).dtype
        if mel.dtype != proj_dtype:
            mel = mel.to(dtype=proj_dtype)

        # 4) Per-time-step processing: transpose to [B, T', 80]
        x = mel.transpose(1, 2).contiguous()  # [B, T', 80]

        # 5) Optional input LayerNorm over Mel bins
        if self.in_norm is not None:
            x = self.in_norm(x)  # LN across last dim (80)

        # 6) Linear projection to LLM embedding space + optional dropout
        x = self.linear(x)      # [B, T', llm_dim]
        x = self.dropout(x)

        return x
