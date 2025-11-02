import numpy as np
from omegaconf import OmegaConf
import whisper
import json
from tqdm import tqdm
from pathlib import Path


def compute_and_save_mel_stats(
    train_jsonl_path: str,
    mel_size: int,
    output_path: str,
    clamp_epsilon: float = 1e-8,
    verbose: bool = True,
):
    """
    Compute global per-mel-bin mean and std from a JSONL list of audio files
    and save results to mean_out and std_out.
    Returns (mel_mean, mel_std) as numpy arrays (float32).
    """
    total_sum = np.zeros(mel_size, dtype=np.float64)
    total_sq_sum = np.zeros(mel_size, dtype=np.float64)
    total_count = 0
    num_lines = sum(1 for _ in open(train_jsonl_path, "r", encoding="utf8"))
    with open(train_jsonl_path, "r", encoding="utf8") as fin:
        for line in tqdm(fin, desc="Computing mel stats", total=num_lines):
            data = json.loads(line)
            audio_path = data.get("source")
            if not audio_path:
                if verbose:
                    print("Skipping entry with no source")
                continue
            try:
                audio_raw = whisper.load_audio(audio_path)
                audio_raw = whisper.pad_or_trim(audio_raw)
                audio_mel = whisper.log_mel_spectrogram(audio_raw, n_mels=mel_size).permute(1, 0)  # [T, mel_size]
                mel_np = audio_mel.numpy()
            except Exception as e:
                if verbose:
                    print(f"Skipping {audio_path} due to error: {e}")
                continue

            total_sum += mel_np.sum(axis=0)
            total_sq_sum += (mel_np ** 2).sum(axis=0)
            total_count += mel_np.shape[0]

    if total_count == 0:
        raise RuntimeError("No frames processed; check dataset paths and file accessibility.")

    mel_mean = total_sum / total_count
    mel_var = (total_sq_sum / total_count) - (mel_mean ** 2)
    mel_std = np.sqrt(np.maximum(mel_var, clamp_epsilon))  # clamp for stability

    # Save to files
    output_dir = Path(output_path)
    data_set_name = "train_100h"

    output_dir.mkdir(parents=True, exist_ok=True)

    mean_file = output_dir / f"mel_means_{data_set_name}.npy"
    std_file = output_dir / f"mel_stds_{data_set_name}.npy"

    np.save(mean_file, mel_mean.astype(np.float32))
    np.save(std_file, mel_std.astype(np.float32))
    if verbose:
        print(f"Global mel means/stds saved: {mean_file} (mean), {std_file} (std)")
        print(f"Global mel means/stds saved: {output_dir / 'mel_means.npy'} (mean), {output_dir / 'mel_stds.npy'} (std)")

    return mel_mean, mel_std


if __name__ == "__main__":

    cfg = OmegaConf.load("configs/train_lin_proj.yaml")

    compute_and_save_mel_stats(
        train_jsonl_path="data/test-clean.jsonl",
        mel_size=cfg.data.mel_size,
        output_path=cfg.data.mel_stats_path,
        clamp_epsilon=cfg.data.clamp_epsilon,
        verbose=cfg.data.verbose,
    )
