import torch
import torch.nn as nn
import whisper


class WhisperEncoder(nn.Module):
    """
    Whisper encoder that outputs features compatible with LinearProjector.
    
    Input:  mel [B, T, 80] or [B, 80, T] 
    Output: [B, T', whisper_dim] where T' depends on Whisper's internal downsampling
    """
    def __init__(
        self,
        whisper_model_name="base",  # tiny, base, small, medium, large
        freeze_encoder=True,
    ):
        super().__init__()
        
        # Load pretrained Whisper model
        self.whisper = whisper.load_model(whisper_model_name)
        
        # Get encoder dimension 
        self.whisper_dim = self.whisper.encoder.ln_post.normalized_shape[0]
        
        # Freeze whisper parameters if requested
        if freeze_encoder:
            for param in self.whisper.parameters():
                param.requires_grad = False
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, T, 80] or [B, 80, T] mel spectrogram
        Returns:
            [B, T', whisper_dim] encoded features
        """
        # Ensure correct format for Whisper (expects [B, 80, T])
        if mel.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {mel.shape}")
        
        if mel.shape[2] == 80:  # [B, T, 80] -> [B, 80, T]
            mel = mel.transpose(1, 2)
        elif mel.shape[1] != 80:
            raise ValueError(f"Expected 80 mel bins, got shape {mel.shape}")
        
        # Whisper encoder forward pass
        x = self.whisper.encoder(mel)  # [B, T', whisper_dim]
        
        return x
