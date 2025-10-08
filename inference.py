import argparse
import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import OmegaConf
from models.model import model_builder


def load_wave_16k_mono(path: str, target_sr: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(path)              # [C, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)      # [1, T]
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)     # [1, T]
    return wav.squeeze(0).contiguous().float()   # [T]

def pad_or_trim_seconds(wav: torch.Tensor, sr: int = 16000, max_sec: int = 30) -> torch.Tensor:
    max_len = sr * max_sec
    n = wav.numel()
    if n < max_len:
        wav = torch.nn.functional.pad(wav, (0, max_len - n))
    elif n > max_len:
        wav = wav[:max_len]
    return wav

def log_mel_80(wav: torch.Tensor, sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    mel_spec = T.MelSpectrogram(
        sample_rate=sr, n_fft=400, win_length=400, hop_length=160,
        f_min=0.0, f_max=None, n_mels=n_mels, mel_scale="htk", power=2.0
    )(wav)                                         # [80, T]
    log_mel = T.AmplitudeToDB(stype="power", top_db=80)(mel_spec)  # [80, T]
    return log_mel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--audio_path", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Transcribe speech to text.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--pad_to_30s", action="store_true", help="Pad/trim to 30s like dataset.py")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)  # attribute access to cfg.model/cfg.train/cfg.data [web:135]
    model, tokenizer = model_builder(cfg.train, cfg.model, ckpt_path=args.ckpt_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1) Raw audio -> 16 kHz mono
    wav = load_wave_16k_mono(args.audio_path, target_sr=16000)  # [T] [web:174]
    if args.pad_to_30s:
        wav = pad_or_trim_seconds(wav, sr=16000, max_sec=30)    # 30 s window [web:39][web:173]

    # 2) 80-bin log-Mel (Whisper params)
    mel = log_mel_80(wav.unsqueeze(0), sr=16000, n_mels=80).to(device)  # [1,80,T] [web:174]
    # Note: dataset returns [T,80], but projector accepts either layout; here we use [B,80,T] directly [web:156]

    # 3) Inference
    text = model.inference(
        audio_mel=mel,                    # [B,80,T]
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(f"Transcription: {text}")

if __name__ == "__main__":
    main()