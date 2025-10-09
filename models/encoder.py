import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import WhisperModel


class WhisperWrappedEncoder:

     @classmethod
     def load(cls, model_config):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x
    

        encoder = WhisperModel.from_pretrained(model_config.encoder_model,torch_dtype=torch.bfloat16).encoder
        
        return encoder