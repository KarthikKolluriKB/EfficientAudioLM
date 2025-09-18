# models/projector.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjector(nn.Module):
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
    

class PatchedProjector(nn.Module):
    """
    """
    def __init__(self, model_config):
        super().__init__()
        # Core dims
        self.mel_bins = int(getattr(model_config, "mel_size", 80))
        self.llm_dim  = int(getattr(model_config, "llm_dim", 3072))

        # Token-rate control (how often we start a patch)
        self.time_stride = int(getattr(model_config, "mel_time_stride", 8))

        # Receptive field control per token (how many frames each token sees)
        self.ds_rate  = int(getattr(model_config, "projector_ds_rate", 4))

        # Optional input normalization and dropout on token embeddings
        self.use_input_norm = bool(getattr(model_config, "mel_input_norm", False))
        self.dropout_p  = float(getattr(model_config, "mel_dropout", 0.0))

        # Derive patching parameters:
        # stride = time_stride (token every `time_stride` frames)
        # patch_size = time_stride * ds_rate (each token sees ds_rate * stride frames)
        self.stride = max(1, self.time_stride)
        self.patch_size = max(self.stride, self.time_stride * self.ds_rate)

        # Linear projection per patch (80*P -> D)
        self.in_dim = self.mel_bins * self.patch_size
        self.proj   = nn.Linear(self.in_dim, self.llm_dim, bias=True)
        self.drop   = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else nn.Identity()


    @staticmethod
    def _norm_over_time(mel: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Normalize each mel bin over time: (x - mean_t) / std_t
        mean = mel.mean(dim=-1, keepdim=True)
        var  = mel.var(dim=-1, unbiased=False, keepdim=True)
        std  = (var + eps).sqrt()
        return (mel - mean) / std
    
    def count_patches(self, T: int) -> int:
        # Number of patches along time for a given T
        if T < self.patch_size:
            T = self.patch_size
        return (T - self.patch_size) // self.stride + 1


    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        mel: [B, 80, T] or [B, T, 80]
        return: [B, N, D]
        """ 
        if mel.dim() != 3:
            raise ValueError(f"mel must be 3D [B, 80, T] or [B, T, 80], got {mel.shape}")

        # Ensure [B, 80, T]
        if mel.shape[1] == self.mel_bins:
            pass
        elif mel.shape[2] == self.mel_bins:
            mel = mel.transpose(1, 2)  # [B, 80, T]
        else:
            raise ValueError(f"Expected {self.mel_bins} mel bins on dim=1 or dim=2, got {mel.shape}")

        # Input Normalization (per mel bin over time)
        if self.use_input_norm:
            mel = self._norm_over_time(mel)

        B, C, T = mel.shape  # C should be self.mel_bins

        # Right-pad time so at least one patch exists
        if T < self.patch_size:
            mel = F.pad(mel, (0, self.patch_size - T), value=0.0)
            T = mel.shape[-1]

        # Extract time patches: [B, 80, N, P]
        patches = mel.unfold(dimension=2, size=self.patch_size, step=self.stride)

        # Flatten per patch to [B, N, 80*P]
        B, C, N, P = patches.shape
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, N, C * P)

        # Project to LM dim and apply dropout: [B, N, D]
        tokens = self.proj(patches)
        tokens = self.drop(tokens)
        return tokens