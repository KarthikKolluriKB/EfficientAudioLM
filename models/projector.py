import torch 
import torch.nn as nn
from torch.nn import functional as F


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
        hidden_dim = 1024  
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
    

class PatchedLinearProjectorV3(nn.Module):
    """
    Patched Linear Projector V3 (Fuyu-Style) for mel spectrograms.  
    Splits the input mel spectrogram into overlapping patches of fixed length along the time dimension, 
    and applies a linear projection followed by LayerNorm to each patch to map it to the desired llm_dim.

    Args:
        config: Configuration object with attributes:
            - patch_length: Length of each patch in frames (e.g., 16)
            - patch_stride: Stride between patches in frames (e.g., 8)  
            - mel_size: Number of mel frequency bins (e.g., 80)
            - llm_dim: Dimension of the output LLM embeddings (e.g., 3072)
    """
    def __init__(self, config):
        super().__init__()
        patch_size = config.patch_length * config.mel_size
        
        self.patch_length = config.patch_length 
        self.patch_stride = config.patch_stride

        # Fuyu-Style: Single Linear Layer + LayerNorm
        # Direct projection: patch_size -> llm_dim
        self.projection = nn.Linear(patch_size, config.llm_dim)
        
        # Critical for encoder-free training stability
        self.norm = nn.LayerNorm(config.llm_dim)
        
        # Initialize weights properly (Xavier)
        self._init_weights()

        print(f"PatchedLinearProjectorV3 (Fuyu-Style): {patch_size} -> {config.llm_dim} (Linear+LN)")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        B, T, N_MELS = x.shape
        
        # 1. Padding Logic (Same as V2)
        remainder = T % self.patch_length
        if remainder != 0:
            pad_length = self.patch_length - remainder
            x = F.pad(x, (0, 0, 0, pad_length), value=0.0)
            T = x.shape[1]

        # 2. Patch Extraction Logic (Same as V2)
        # Using the exact loop structure from V2 for consistency
        patches = []
        for i in range(0, T - self.patch_length + 1, self.patch_stride):
            patch = x[:, i:i+self.patch_length, :].reshape(B, -1)
            patches.append(patch)
        
        if len(patches) == 0:
             # Safety for very short audio
             return torch.zeros(B, 0, self.projection.out_features, device=x.device)

        patches = torch.stack(patches, 1)

        # 3. Projection (Linear -> Norm)
        x = self.projection(patches)
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
    

