"""
Compute mel-spectrogram mean and std statistics for normalization.
Matches the exact feature extraction in datasets.py using librosa.
"""
import numpy as np
from omegaconf import OmegaConf
import librosa
import json
from tqdm import tqdm
from pathlib import Path


def extract_log_mel_spectrogram(
    audio_path: str,
    mel_size: int = 80,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 0,
    fmax: float = 8000,
) -> np.ndarray:
    """
    Extract log-mel spectrogram using librosa (EXACT match to datasets.py).
    
    Args:
        audio_path: Path to audio file
        mel_size: Number of mel bins
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length
        fmin: Minimum frequency
        fmax: Maximum frequency
    
    Returns:
        numpy array of shape [time, mel_size]
    """
    # Load audio using librosa (same as datasets.py)
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Compute mel spectrogram (EXACT same parameters as datasets.py)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=mel_size,
        fmin=fmin,
        fmax=fmax,
        power=2.0  # Power Spectrogram (IMPORTANT!)
    )
    
    # Convert to log scale (EXACT same as datasets.py)
    log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Transpose to (time, freq) format (EXACT same as datasets.py)
    log_mel_spec = log_mel_spec.T
    
    return log_mel_spec


def compute_and_save_mel_stats(
    train_jsonl_path: str,
    mel_size: int,
    output_path: str,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    fmin: float = 0,
    fmax: float = 8000,
    clamp_epsilon: float = 1e-8,
    verbose: bool = True,
):
    """
    Compute global per-mel-bin mean and std from a JSONL list of audio files.
    Uses EXACT same feature extraction as datasets.py.
    
    Args:
        train_jsonl_path: Path to training JSONL file
        mel_size: Number of mel bins
        output_path: Directory to save statistics
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        win_length: Window length
        fmin: Minimum frequency
        fmax: Maximum frequency
        clamp_epsilon: Epsilon for numerical stability
        verbose: Print progress information
    
    Returns:
        (mel_mean, mel_std) as numpy arrays (float32)
    """
    total_sum = np.zeros(mel_size, dtype=np.float64)
    total_sq_sum = np.zeros(mel_size, dtype=np.float64)
    total_count = 0
    
    # Count total lines for progress bar
    num_lines = sum(1 for _ in open(train_jsonl_path, "r", encoding="utf8"))
    
    print(f"Computing mel statistics from {num_lines} samples...")
    print(f"Parameters (matching datasets.py):")
    print(f"  mel_size: {mel_size}")
    print(f"  sample_rate: {sample_rate}")
    print(f"  n_fft: {n_fft}")
    print(f"  hop_length: {hop_length}")
    print(f"  win_length: {win_length}")
    print(f"  fmin: {fmin}, fmax: {fmax}")
    print(f"  power: 2.0 (Power Spectrogram)")
    print()
    
    with open(train_jsonl_path, "r", encoding="utf8") as fin:
        for line in tqdm(fin, desc="Computing mel stats", total=num_lines):
            data = json.loads(line)
            audio_path = data.get("source")
            
            if not audio_path:
                if verbose:
                    print("Skipping entry with no source")
                continue
            
            try:
                # Extract mel using EXACT same function as datasets.py
                mel_np = extract_log_mel_spectrogram(
                    audio_path=audio_path,
                    mel_size=mel_size,
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    fmin=fmin,
                    fmax=fmax,
                )
                
            except Exception as e:
                if verbose:
                    print(f"Skipping {audio_path} due to error: {e}")
                continue
            
            # Accumulate statistics
            total_sum += mel_np.sum(axis=0)
            total_sq_sum += (mel_np ** 2).sum(axis=0)
            total_count += mel_np.shape[0]
    
    if total_count == 0:
        raise RuntimeError("No frames processed; check dataset paths and file accessibility.")
    
    # Compute mean and std
    mel_mean = total_sum / total_count
    mel_var = (total_sq_sum / total_count) - (mel_mean ** 2)
    mel_std = np.sqrt(np.maximum(mel_var, clamp_epsilon))  # clamp for stability
    
    # Save to files
    output_dir = Path(output_path)
    data_set_name = "train_50h"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean_file = output_dir / f"mel_means_{data_set_name}.npy"
    std_file = output_dir / f"mel_stds_{data_set_name}.npy"
    
    np.save(mean_file, mel_mean.astype(np.float32))
    np.save(std_file, mel_std.astype(np.float32))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Mel Statistics Computed Successfully!")
        print(f"{'='*60}")
        print(f"Total frames processed: {total_count}")
        print(f"Mel mean shape: {mel_mean.shape}")
        print(f"Mel std shape: {mel_std.shape}")
        print(f"Mean range: [{mel_mean.min():.4f}, {mel_mean.max():.4f}]")
        print(f"Std range: [{mel_std.min():.4f}, {mel_std.max():.4f}]")
        print(f"\nSaved to:")
        print(f"  Mean: {mean_file}")
        print(f"  Std:  {std_file}")
        print(f"{'='*60}\n")
    
    return mel_mean, mel_std


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compute and save global mel spectrogram mean and std from training data."
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to configuration file (YAML) containing data paths and parameters.",
    )
    
    args = parser.parse_args()
    config_path = args.config_path
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Get parameters from config (matching datasets.py defaults)
    mel_size = cfg.data.get('mel_size', 80)
    sample_rate = cfg.data.get('sample_rate', 16000)
    n_fft = cfg.data.get('n_fft', 400)
    hop_length = cfg.data.get('hop_length', 160)
    win_length = cfg.data.get('win_length', 400)
    fmin = cfg.data.get('fmin', 0)
    fmax = cfg.data.get('fmax', 8000)
    clamp_epsilon = cfg.data.get('clamp_epsilon', 1e-8)
    verbose = cfg.data.get('verbose', True)
    
    compute_and_save_mel_stats(
        train_jsonl_path=cfg.data.train_data_path,
        mel_size=mel_size,
        output_path=cfg.data.mel_stats_path,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        clamp_epsilon=clamp_epsilon,
        verbose=verbose,
    )
