import torch 
import torch.nn as nn
from torch.nn import functional as F


class CNNFuyuProjectorV2(nn.Module):
    """
    Fuyu-Inspired Encoder-Free Audio Projector V2.

    Adapts the Fuyu-8B philosophy for audio with one key modification:
    a single Conv1d replaces Fuyu's fixed patching. This is necessary
    because unlike image patches (which are already visually meaningful),
    raw mel patches lack acoustic structure — audio needs learned local
    feature extraction before projection.

        Fuyu-8B (vision):  image → fixed patch → Linear → LLM
        This    (audio):   mel   → Conv1d      → Linear → LLM

    The Conv1d acts as a "learned patcher" that extracts local acoustic
    features (phoneme-level patterns across ~70ms) while downsampling.
    The linear projection then maps to LLM space, as in Fuyu.

    Architecture:
        mel [B, T, 80]
            → Conv1d(80, C, k=7, s=2) → GELU → LN   [learned local features + 2x downsample]
            → Linear(C → llm_dim)                     [projection to LLM space]
            → + Positional Embeddings → LayerNorm
            → LLM

    Args:
        config: Configuration with:
            - mel_size: Mel frequency bins (e.g., 80)
            - llm_dim: LLM embedding dimension (e.g., 3072)
            - cnn_channels: Conv output channels (default: 384)
            - cnn_kernel_size: Conv kernel size (default: 7)
            - max_audio_length: Maximum audio frames (default: 3000)
            - mel_dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, config):
        super().__init__()

        mel_dim = config.mel_size
        llm_dim = config.llm_dim
        C = getattr(config, "cnn_channels", 384)
        kernel_size = getattr(config, "cnn_kernel_size", 7)
        dropout = getattr(config, "mel_dropout", 0.1)

        max_audio_length = getattr(config, "max_audio_length", 3000)
        self.max_positions = max_audio_length // 2  # 2x downsampling

        # Conv1d: learned local feature extraction + 2x downsampling
        # Replaces Fuyu's fixed patching — audio needs this because
        # raw mel patches lack acoustic structure unlike image patches
        self.conv = nn.Conv1d(mel_dim, C, kernel_size=kernel_size,
                              stride=2, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.conv_norm = nn.LayerNorm(C)

        # Linear projection to LLM space (same as Fuyu)
        self.projection = nn.Linear(C, llm_dim)
        self.proj_dropout = nn.Dropout(dropout)

        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(self.max_positions, llm_dim)

        # Final LayerNorm
        self.norm_out = nn.LayerNorm(llm_dim)

        self._init_weights()

        # Parameter counting
        conv_params = sum(p.numel() for p in self.conv.parameters()) + \
                      sum(p.numel() for p in self.conv_norm.parameters())
        proj_params = sum(p.numel() for p in self.projection.parameters())
        pos_params = self.max_positions * llm_dim
        total_params = sum(p.numel() for p in self.parameters())

        print(f"\n{'='*60}")
        print(f"CNNFuyuProjectorV2 (Conv + Linear Projector, 2x Down)")
        print(f"  Mel Dim: {mel_dim}")
        print(f"  Conv Channels: {C}, Kernel: {kernel_size}")
        print(f"  Downsampling: 2x (single stride-2 conv)")
        print(f"  Projection: Linear({C} → {llm_dim})")
        print(f"  Max Positions: {self.max_positions} (~{self.max_positions * 2 * 10 / 1000:.1f}s audio)")
        print(f"  Dropout: {dropout}")
        print(f"  Architecture: mel[{mel_dim}] → Conv[{C}] → Linear → +PosEmb → [{llm_dim}]")
        print(f"  Parameters: {total_params:,}")
        print(f"    - Conv + Norm: {conv_params:,}")
        print(f"    - Projection: {proj_params:,}")
        print(f"    - PosEmb: {pos_params:,}")
        print(f"{'='*60}\n")

    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, T, mel_dim] - mel spectrogram
        Returns:
            [B, T/2, llm_dim] - projected features
        """
        # Conv1d: learned local features + 2x downsample
        x = x.transpose(1, 2)           # [B, mel_dim, T]
        x = self.conv(x)                # [B, C, T/2]
        x = self.act(x)
        x = x.transpose(1, 2)           # [B, T/2, C]
        x = self.conv_norm(x)

        # Linear projection to LLM space
        x = self.projection(x)          # [B, T/2, llm_dim]
        x = self.proj_dropout(x)

        # Add positional embeddings
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.clamp(max=self.max_positions - 1)
        x = x + self.pos_embedding(positions)

        # Final norm
        x = self.norm_out(x)

        return x


class CNNFuyuProjector(nn.Module):
    """
    CNN Downsampler + Fuyu-Style Projector with Positional Embeddings.
    
    Architecture:
        mel [B, T, 80] 
            → CNN Downsampler (2 layers, stride 2 each) → [B, T/4, cnn_channels]
            → Linear Projection → [B, T/4, llm_dim]
            → + Positional Embeddings
            → LayerNorm
            → LLM
    
    Design Philosophy:
        - CNN provides local acoustic pattern extraction (what pure Fuyu lacks)
        - 4x temporal downsampling reduces sequence length for LLM efficiency
        - Positional embeddings preserve temporal order information
        - Minimal complexity: only 2 CNN layers + 1 Linear
    
    Why This Works When Pure Fuyu Fails:
        - CNN kernels capture phoneme-level patterns across frames
        - Overlapping receptive fields handle phoneme boundaries
        - Downsampling creates more information-dense representations
    
    Args:
        config: Configuration with:
            - mel_size: Mel frequency bins (e.g., 80)
            - llm_dim: LLM embedding dimension (e.g., 3072)
            - cnn_channels: CNN output channels (default: 256)
            - cnn_kernel_size: Convolution kernel size (default: 5)
            - max_audio_length: Maximum audio frames (default: 3000)
            - mel_dropout: Dropout rate (default: 0.1)
    """
    def __init__(self, config):
        super().__init__()
        
        # Dimensions
        mel_dim = config.mel_size
        llm_dim = config.llm_dim
        cnn_channels = getattr(config, "cnn_channels", 256)
        kernel_size = getattr(config, "cnn_kernel_size", 5)
        dropout = getattr(config, "mel_dropout", 0.1)
        
        # Max positions after 4x downsampling
        max_audio_length = getattr(config, "max_audio_length", 3000)
        self.max_positions = max_audio_length // 4
        
        # 2-Layer CNN Downsampler (4x total downsampling)
        # Each layer: stride=2 → 2x downsampling
        self.cnn = nn.Sequential(
            # Layer 1: [B, mel_dim, T] → [B, cnn_channels, T/2]
            nn.Conv1d(mel_dim, cnn_channels, kernel_size=kernel_size, 
                      stride=2, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
            
            # Layer 2: [B, cnn_channels, T/2] → [B, cnn_channels, T/4]
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, 
                      stride=2, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
        )
        
        # Linear projection to LLM dimension
        self.projection = nn.Linear(cnn_channels, llm_dim)
        
        # Positional embeddings for temporal awareness
        self.pos_embedding = nn.Embedding(self.max_positions, llm_dim)
        
        # LayerNorm for stability
        self.norm = nn.LayerNorm(llm_dim)
        
        self._init_weights()
        
        # Parameter counting
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        proj_params = cnn_channels * llm_dim + llm_dim
        pos_params = self.max_positions * llm_dim
        norm_params = 2 * llm_dim
        total_params = cnn_params + proj_params + pos_params + norm_params
        
        print(f"\n{'='*60}")
        print(f"CNNFuyuProjector (CNN Downsampler + Fuyu + PosEmb)")
        print(f"  Mel Dim: {mel_dim}")
        print(f"  CNN Channels: {cnn_channels}")
        print(f"  Kernel Size: {kernel_size}")
        print(f"  Downsampling: 4x (stride 2 × 2 layers)")
        print(f"  LLM Dim: {llm_dim}")
        print(f"  Max Positions: {self.max_positions} (~{self.max_positions * 4 * 10 / 1000:.1f}s audio)")
        print(f"  Dropout: {dropout}")
        print(f"  Architecture: mel[{mel_dim}] → CNN[{cnn_channels}] → Linear → +PosEmb → [{llm_dim}]")
        print(f"  Parameters: {total_params:,}")
        print(f"    - CNN: {cnn_params:,}")
        print(f"    - Projection: {proj_params:,}")
        print(f"    - PosEmb: {pos_params:,}")
        print(f"{'='*60}\n")

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Xavier for CNN and Linear
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Normal initialization for positional embeddings
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Forward pass: CNN downsample → project → add positional embeddings.
        
        Args:
            x: [B, T, mel_dim] - mel spectrogram
        
        Returns:
            [B, T/4, llm_dim] - projected features with positional info
        """
        B, T, mel_dim = x.shape
        
        # CNN expects [B, C, T] format
        x = x.transpose(1, 2)  # [B, mel_dim, T]
        
        # CNN downsampling: [B, mel_dim, T] → [B, cnn_channels, T/4]
        x = self.cnn(x)
        
        # Back to [B, T/4, cnn_channels]
        x = x.transpose(1, 2)
        
        # Linear projection to LLM dim
        x = self.projection(x)
        
        # Add positional embeddings
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.clamp(max=self.max_positions - 1)
        x = x + self.pos_embedding(positions)
        
        # Normalize
        x = self.norm(x)
        
        return x


class FuyuAudioProjector(nn.Module):
    """
    Fuyu-Style Audio Projector V2 with 2-Layer MLP and Positional Embeddings.
    
    Enhanced version that adds non-linearity to learn acoustic patterns
    while staying true to the encoder-free Fuyu philosophy.
    
    Architecture:
        mel spectrogram [B, T, F] 
            -> patch [B, N, patch_length * F]
            -> 2-layer MLP [B, N, llm_dim]
            -> + positional embedding [N, llm_dim]
            -> LayerNorm
    
    Why 2-Layer MLP?
        - Single linear may be too weak for mel→text mapping
        - GELU adds non-linearity to learn acoustic patterns
        - Hidden layer provides capacity for feature transformation
        - Still much simpler than full audio encoders (Whisper: 24 layers)
    
    Args:
        config: Configuration with:
            - patch_length: Number of frames per patch (e.g., 16 = 160ms)
            - mel_size: Mel frequency bins (e.g., 80)
            - llm_dim: LLM embedding dimension (e.g., 3072)
            - hidden_dim: MLP hidden dimension (default: 2048)
            - input_type: "mel", "mfcc", or "combined"
            - max_audio_patches: Maximum patches (default: 512)
    """
    def __init__(self, config):
        super().__init__()
        self.patch_length = config.patch_length
        
        # Determine feature dimension based on input type
        self.feature_dim = self._get_feature_dim(config)
        
        # Dimensions
        patch_dim = self.patch_length * self.feature_dim
        hidden_dim = getattr(config, "hidden_dim", 2048)
        llm_dim = config.llm_dim
        dropout = getattr(config, "mel_dropout", 0.1)
        
        # Maximum audio patches for positional embeddings
        self.max_patches = getattr(config, "max_audio_patches", 512)
        
        # 2-Layer MLP Projection (adds non-linearity for acoustic patterns)
        # patch_dim -> hidden_dim -> llm_dim
        self.projection = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim),
        )
        
        # Learned positional embeddings for temporal awareness
        self.pos_embedding = nn.Embedding(self.max_patches, llm_dim)
        
        # LayerNorm for training stability
        self.norm = nn.LayerNorm(llm_dim)
        
        # Final dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
        # Logging
        proj_params = (patch_dim * hidden_dim + hidden_dim + 
                       hidden_dim * llm_dim + llm_dim)
        pos_params = self.max_patches * llm_dim
        norm_params = 2 * llm_dim
        total_params = proj_params + pos_params + norm_params
        
        print(f"\n{'='*60}")
        print(f"FuyuAudioProjector V2 (2-Layer MLP + Positional)")
        print(f"  Input Type: {getattr(config, 'input_type', 'mel')}")
        print(f"  Feature Dim: {self.feature_dim}")
        print(f"  Patch Length: {self.patch_length} frames ({self.patch_length * 10}ms)")
        print(f"  Patch Dim: {patch_dim}")
        print(f"  Hidden Dim: {hidden_dim}")
        print(f"  Output Dim: {llm_dim}")
        print(f"  Max Patches: {self.max_patches} (~{self.max_patches * self.patch_length * 10 / 1000:.1f}s audio)")
        print(f"  Dropout: {dropout}")
        print(f"  Architecture: [{patch_dim}] -> MLP({hidden_dim}) -> + PosEmb -> [{llm_dim}]")
        print(f"  Parameters: {total_params:,}")
        print(f"    - MLP: {proj_params:,}")
        print(f"    - PosEmb: {pos_params:,}")
        print(f"{'='*60}\n")

    def _get_feature_dim(self, config):
        """Determine feature dimension based on input type."""
        input_type = getattr(config, "input_type", "mel").lower()
        
        if input_type == "mfcc":
            return config.n_mfcc
        elif input_type == "combined":
            return config.mel_size + config.n_mfcc
        else:  # "mel" (default)
            return config.mel_size

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Xavier for MLP layers
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        # Normal initialization for positional embeddings
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Forward pass: Patch, MLP project, add positional embeddings.
        
        Args:
            x: [B, T, feat_dim] - mel spectrogram or MFCC features
        
        Returns:
            [B, N, llm_dim] - projected patches with positional info
        """
        B, T, feat_dim = x.shape
        
        # Pad to multiple of patch_length
        remainder = T % self.patch_length
        if remainder != 0:
            pad_len = self.patch_length - remainder
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            T = x.shape[1]
        
        # Reshape into patches: [B, T, feat_dim] -> [B, N, patch_length * feat_dim]
        N = T // self.patch_length
        x = x.view(B, N, self.patch_length * feat_dim)
        
        # 2-Layer MLP projection
        x = self.projection(x)
        
        # Add positional embeddings
        positions = torch.arange(N, device=x.device)
        positions = positions.clamp(max=self.max_patches - 1)
        x = x + self.pos_embedding(positions)
        
        # Normalize and dropout
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class MelProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.projector_ds_rate
        self.mel_dim = config.mel_size
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.mel_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x    

class PatchedLinearProjectorV1(nn.Module):
    """
    Patched Linear Projector for mel spectrograms.  
    Splits the input mel spectrogram into non-overlapping patches of fixed length along the time dimension,
    and applies a linear projection to each patch to map it to the desired llm_dim.

    Args:
        config: Configuration object with attributes:
            - patch_length: Length of each patch in frames (e.g., 16)
            - mel_size: Number of mel frequency bins (e.g., 80)
            - llm_dim: Dimension of the output LLM embeddings (e.g., 3072)   

    Input:
        x: Tensor of shape [B, T, 80] representing mel spectrograms 

    Output:
        Tensor of shape [B, num_patches, llm_dim] representing projected patches

    Returns: 
        x: Projected tensor of shape [B, num_patches, llm_dim] 
    """
    def __init__(self, config):
        super().__init__()
        self.patch_length = config.patch_length  # 16 frames = 160ms
        patch_size = self.patch_length * config.mel_size  # 16 * 80 = 1280
        
        # Linear projection from patch_size to llm_dim
        self.projection = nn.Linear(patch_size, config.llm_dim)

        print(f"PatchedLinearProjector: {patch_size} -> {config.llm_dim}")

    def forward(self, x):
        """
        Input: [B, T, 80] mel spectrograms
        Output: [B, num_patches, llm_dim]
        """
        B, T, N_MELS = x.shape
        
        # Handle variable length by padding to multiple of patch_length
        remainder = T % self.patch_length
        if remainder != 0:
            pad_length = self.patch_length - remainder
            x = F.pad(x, (0, 0, 0, pad_length), value=0.0)
            T = x.shape[1]
        
        # Create patches and project
        num_patches = T // self.patch_length
        patches = x.view(B, num_patches, self.patch_length * N_MELS)

        x = self.projection(patches)
        return x
    

class PatchedLinearProjectorV2(nn.Module):
    """
    Optimized V2 Projector: MLP + LayerNorm
    Structure: Linear -> GELU -> Linear -> Dropout -> LayerNorm
    """
    def __init__(self, config):
        super().__init__()
        patch_size = config.patch_length * config.mel_size
        hidden_dim = getattr(config, "hidden_dim", 1024) 
        self.patch_length = config.patch_length 
        self.patch_stride = config.patch_stride

        # MLP Projection
        self.projection = nn.Sequential(
            nn.Linear(patch_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.llm_dim),
            nn.Dropout(config.mel_dropout)
        )
        
        # LayerNorm for stability
        self.norm = nn.LayerNorm(config.llm_dim)
        
        print(f"PatchedLinearProjectorV2 (Optimized): {patch_size} -> {hidden_dim} -> {config.llm_dim}")
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (Batch, Time, Mel)
        """
        B, T, F_dim = x.shape
        
        # 1. Pad if needed
        remainder = T % self.patch_length
        if remainder != 0:
            pad_length = self.patch_length - remainder
            x = F.pad(x, (0, 0, 0, pad_length), value=0.0)
            T = x.shape[1]

        # 2. Optimized Patch Extraction (Unfold is faster than loop)
        # x.unfold(dimension, size, step)
        # Output: (B, Num_Patches, Mel, Patch_Len)
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        
        # Rearrange to (B, Num_Patches, Patch_Len * Mel)
        patches = patches.permute(0, 1, 3, 2) 
        patches = patches.reshape(B, patches.size(1), -1)

        # 3. Project + Norm
        x = self.projection(patches)
        x = self.norm(x)  # Final stability check
        
        return x

class ContextAwareProjector(nn.Module):
    """
    Context-Aware Projector: Utilizes neighboring patches for richer context.
    
    Structure: For each patch at time t, concatenate patches at t-1, t, t+1
    before projection. This provides temporal context to the model.

    Args:
        config: Configuration object with attributes:
            - patch_length: Length of each patch in frames (e.g., 16)
            - patch_stride: Stride between patches (e.g., 8)
            - mel_size: Number of mel frequency bins (e.g., 80)
            - llm_dim: Dimension of the output LLM embeddings (e.g., 3072)
    Input:
        x: Tensor of shape [B, T, 80] representing mel spectrograms

    Output:
        Tensor of shape [B, num_patches, llm_dim] representing projected patches with context
    """    
    def __init__(self, config):
        super().__init__()
        self.patch_length = config.patch_length 
        self.patch_stride = config.patch_stride
        
        # Standard patch dim
        raw_patch_dim = config.patch_length * config.mel_size
        
        # Triple the context by concatenating neighboring patches
        context_dim = raw_patch_dim * 3 
        
        # Use Wide Hidden Dim (e.g., 2048) to handle the extra info
        hidden_dim = getattr(config, "hidden_dim", 2048) 

        # MLP Projection (Deep & Wide)
        self.projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),  # Compresses 3x context
            nn.GELU(),
            nn.Dropout(config.mel_dropout),
            nn.Linear(hidden_dim, config.llm_dim)
        )
        
        self.norm = nn.LayerNorm(config.llm_dim)
        
        print(f"ProjectorV3: {context_dim} -> {hidden_dim} -> {config.llm_dim}")
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        B, T, F_dim = x.shape
        
        # 1. Pad for patching
        remainder = T % self.patch_length
        if remainder != 0:
            pad_length = self.patch_length - remainder
            x = F.pad(x, (0, 0, 0, pad_length), value=0.0)
            T = x.shape[1]

        # 2. Extract Patches
        # Shape: (B, Num_Patches, raw_patch_dim)
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).reshape(B, -1, self.patch_length * F_dim)
        
        # 3. Create Context Windows [t-1, t, t+1]
        # Padding for boundaries (so t=0 has a 'zero' t-1)
        # Pad 1 patch left and 1 patch right
        # (B, Num_Patches + 2, D)
        padded_patches = F.pad(patches, (0, 0, 1, 1), value=0.0) 
        
        # Unfold along the 'patches' dimension to get window of 3
        # (B, Num_Patches, D, 3)
        context_windows = padded_patches.unfold(1, 3, 1) 
        
        # Flatten the context: (B, Num_Patches, 3*D)
        # We want [t-1, t, t+1] concatenated
        B_new, N_new, D, Window = context_windows.shape
        context_input = context_windows.permute(0, 1, 3, 2).reshape(B_new, N_new, D * Window)

        # 4. Project
        x = self.projection(context_input)
        x = self.norm(x)
        
        return x


class ContextAwareProjectorGLU(nn.Module):
    """
    Context-Aware Projector: Uses GLU (Gated Linear Unit) for smarter context filtering.
    Supports mel, mfcc, and combined feature types.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_length = config.patch_length 
        self.patch_stride = config.patch_stride
        self.context_size = 3  # Fixed to 3 for [t-1, t, t+1]
        
        # Determine feature dimension based on input type
        feature_dim = self._get_feature_dim(config)

        # Standard patch dim
        raw_patch_dim = self.patch_length * feature_dim
        
        # Context dimension (3 patches for temporal context)
        context_dim = raw_patch_dim * self.context_size
        
        # Hidden dim
        hidden_dim = getattr(config, "hidden_dim", 2048)

        # GLU Projection Logic
        # 1. Project to 2 * hidden_dim (because GLU consumes half for gating)
        # 2. Apply GLU (outputs hidden_dim)
        # 3. Dropout for regularization
        # 4. Project to llm_dim
        self.projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim * 2),  # double width for GLU
            nn.GLU(dim=-1),                          # GLU activation
            nn.Dropout(config.mel_dropout),
            nn.Linear(hidden_dim, config.llm_dim)
        )
        
        self.norm = nn.LayerNorm(config.llm_dim)
        
        # Logging for debugging
        input_type = getattr(config, "input_type", "mel")
        print(f"\n{'='*60}")
        print(f"ContextAwareProjectorGLU Configuration:")
        print(f"  Input Type: {input_type}")
        print(f"  Feature Dimension: {feature_dim}")
        print(f"  Patch Length: {self.patch_length}")
        print(f"  Patch Stride: {self.patch_stride}")
        print(f"  Raw Patch Dim: {raw_patch_dim}")
        print(f"  Context Size: {self.context_size}")
        print(f"  Context Dim: {context_dim}")
        print(f"  Architecture: {context_dim} -> GLU({hidden_dim*2}->{hidden_dim}) -> {config.llm_dim}")
        print(f"{'='*60}\n")
        
        self._init_weights()

    def _get_feature_dim(self, config):
        """
        Determine feature dimension based on input type.
        
        Supports:
        - "mel": mel_size only
        - "mfcc": n_mfcc only  
        - "combined": mel_size + n_mfcc
        
        Args:
            config: Configuration object with input_type, mel_size, and/or n_mfcc
            
        Returns:
            int: Feature dimension
        """
        input_type = getattr(config, "input_type", "mel").lower()
        
        if input_type == "mfcc":
            # MFCC only
            if not hasattr(config, "n_mfcc"):
                raise ValueError(
                    "Config must have 'n_mfcc' attribute when input_type='mfcc'"
                )
            return config.n_mfcc
        
        elif input_type == "combined":
            # Mel + MFCC
            if not hasattr(config, "mel_size") or not hasattr(config, "n_mfcc"):
                raise ValueError(
                    "Config must have both 'mel_size' and 'n_mfcc' "
                    "attributes when input_type='combined'"
                )
            return config.mel_size + config.n_mfcc
        
        else:  # "mel" or default
            # Mel only (default behavior)
            if not hasattr(config, "mel_size"):
                raise ValueError(
                    "Config must have 'mel_size' attribute when input_type='mel'"
                )
            return config.mel_size
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass with context-aware patching and GLU projection.
        
        Args:
            x: (B, T, F_dim) - batch, time, feature dimension
               - For mel: F_dim = mel_size (e.g., 80)
               - For mfcc: F_dim = n_mfcc (e.g., 40)
               - For combined: F_dim = mel_size + n_mfcc (e.g., 120)
        
        Returns:
            (B, Num_Patches, LLM_dim) - projected context-aware patches
        """
        B, T, F_dim = x.shape
        
        # Pad for patching (handle remainder)
        remainder = T % self.patch_length
        if remainder != 0:
            pad_length = self.patch_length - remainder
            x = F.pad(x, (0, 0, 0, pad_length), value=0.0)
            T = x.shape[1]

        # Extract Patches
        # unfold: (B, T, F_dim) -> (B, Num_Patches, patch_length, F_dim)
        # Shape: (B, Num_Patches, raw_patch_dim)
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).reshape(B, -1, self.patch_length * F_dim)
        
        # Create Context Windows [t-1, t, t+1]
        # Pad 1 patch left and 1 patch right for temporal context
        padded_patches = F.pad(patches, (0, 0, 1, 1), value=0.0) 
        
        # Window size 3 for context [previous, current, next]
        context_windows = padded_patches.unfold(1, 3, 1) 
        
        # Flatten the context: (B, Num_Patches, 3*raw_patch_dim)
        B_new, N_new, D, Window = context_windows.shape
        context_input = context_windows.permute(0, 1, 3, 2).reshape(B_new, N_new, D * Window)

        # Apply GLU projection and normalization
        x = self.projection(context_input)
        x = self.norm(x)
        
        return x



class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        
        return query_proj
    

