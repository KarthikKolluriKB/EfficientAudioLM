import torch 
import torch.nn as nn


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
        B, T, F = x.shape
        
        # Handle variable length by padding to multiple of patch_length
        remainder = T % self.patch_length
        if remainder != 0:
            pad_length = self.patch_length - remainder
            x = F.pad(x, (0, 0, 0, pad_length), value=0.0)
            T = x.shape[1]
        
        # Create patches and project
        num_patches = T // self.patch_length
        patches = x.view(B, num_patches, self.patch_length * F)

        x = self.projection(patches)
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
    

