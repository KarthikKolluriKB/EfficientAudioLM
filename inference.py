import argparse
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np 

from transformers import AutoProcessor
from models.model import model_builder 
from omegaconf import OmegaConf



def main():
    #parser = argparse.ArgumentParser()
    # parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    # parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    # parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file")
    # parser.add_argument("--prompt", type=str, default="<|ASR|>", help="(optional) Conditioning Prompt")
    # parser.add_argument("--max_new_tokens", type=int, default=64)
    #args = parser.parse_args()

    args = argparse.Namespace(config='configs/config.yaml', 
                              ckpt_path="outputs/ASRLLM_enc_lin_20h/projector_best_wer.pt",
                              prompt="Transcribe speech to text.",
                              audio_path="audio/test.flac",
                              max_new_tokens=64)

    # Load config and build model/tokenizer
    cfg = OmegaConf.load(args.config)
    model, tokenizer = model_builder(cfg.train, cfg.model, ckpt_path=args.ckpt_path, data_config=cfg.data)
    # Setting model to eval
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text = model.inference(
        audio_path=args.audio_path, 
        prompt=args.prompt, 
        max_new_tokens=args.max_new_tokens,
        device=device, 
        data_config=cfg.data
    )
    print(f"Transcription: {text}")

if __name__ == "__main__":
    main()


#Example usage: 

# python inference.py --config configs/test_config.yaml --ckpt_path outputs/projector_best.pt --audio_path path/to/file.wav --prompt "Transcribe speech to text." --max_new_tokens 64
