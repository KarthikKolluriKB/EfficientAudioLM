import os
import json
import copy
from typing import List, Dict, Any

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

def load_wave_16k_mono(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file as mono at target_sr. Returns float32 tensor [T].
    """
    wav, sr = torchaudio.load(path)  # [C, T]  # torchaudio load 
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)  # [1, T]  # resample 
    return wav.squeeze(0).contiguous().float()  # [T]

def pad_or_trim_seconds(wav: torch.Tensor, sr: int = 16000, max_sec: int = 30) -> torch.Tensor:
    """
    Pad with zeros or truncate to exactly max_sec seconds (default 30 s).
    """
    max_len = sr * max_sec
    n = wav.numel()
    if n < max_len:
        wav = torch.nn.functional.pad(wav, (0, max_len - n))
    elif n > max_len:
        wav = wav[:max_len]
    return wav

def log_mel_80(wav: torch.Tensor, sr: int = 16000, n_mels: int = 80) -> torch.Tensor:
    """
    Compute 80-bin log-Mel spectrogram using Whisper-like frontend:
      - n_fft=400 (25 ms), hop_length=160 (10 ms), win_length=400
      - mel_scale='htk', power=2.0, AmplitudeToDB with top_db=80
    Returns [T, 80].
    """
    mel_spec = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=400,
        win_length=400,
        hop_length=160,
        f_min=0.0,
        f_max=None,
        n_mels=n_mels,
        mel_scale="htk",
        power=2.0,
    )(wav)  # [80, T]  # MelSpectrogram 
    log_mel = T.AmplitudeToDB(stype="power", top_db=80)(mel_spec)  # [80, T]  # AmplitudeToDB 
    return log_mel.transpose(0, 1).contiguous()  # [T, 80]

class SpeechDatasetJsonl(Dataset):
    """
    JSONL dataset for projector-only ASR (mel -> projector -> LLM).
    Produces:
      - input_ids: concat(audio placeholder, prompt+answer+eos)
      - labels: ignore audio+prompt, keep answer+eos; IGNORE_INDEX elsewhere
      - attention_mask: boolean mask over input_ids
      - audio_mel: [T, 80] when self.input_type == 'mel'
      - audio_length: placeholder int used by the model to build masks
    """
    def __init__(self, dataset_config, tokenizer=None, split: str = "train"):
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer

        # Fixed settings to match existing pipeline
        self.IGNORE_INDEX = -100
        self.mel_size = 80
        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = -1
        self.inference_mode = False
        self.normalize = True
        self.input_type = "mel"
        assert self.input_type in ["raw", "mel"]

        # Load jsonl lines
        self.data_list: List[Dict[str, Any]] = []
        path = dataset_config.train_data_path if split == "train" else dataset_config.val_data_path
        with open(path, encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line.strip())
                self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def get_source_len(self, data_dict):
        return data_dict.get("source_len", 0)

    def get_target_len(self, data_dict):
        return data_dict.get("target_len", 0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data_list[index]
        audio_path = data.get("source")
        target = data.get("target", "")
        key = data.get("key", None)

        # 1) Load audio 16 kHz mono
        wav = load_wave_16k_mono(audio_path, target_sr=16000)  # [T]  # load/resample [web:1003][web:590]

        # 2) Build input features
        if self.input_type == "raw":
            audio_raw = wav
            if self.normalize:
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
            # Legacy placeholder sizing to keep masks consistent
            audio_length = (audio_raw.numel() // 320) // 5
            audio_mel = None
        else:
            wav_30s = pad_or_trim_seconds(wav, sr=16000, max_sec=30)  # 30s framing [web:562]
            audio_mel = log_mel_80(wav_30s, sr=16000, n_mels=self.mel_size)  # [T, 80] [web:995][web:999]
            # Legacy placeholder sizing assumed by masks downstream
            audio_length = ((audio_mel.shape[0] + 1) // 2) // 5
            audio_raw = None

        if self.fix_length_audio > 0:
            audio_length = self.fix_length_audio
        audio_pseudo = torch.full((audio_length,), -1, dtype=torch.int64)

        # 3) Prompt / labels
        prompt_text = (
            "Transcribe speech to text. Output the transcription directly without redundant content. "
            "Ensure that the output is not duplicated."
        )
        prompt = self.prompt_template.format(prompt_text)
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        if self.inference_mode:
            prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.int64)
            input_ids = torch.cat((audio_pseudo, prompt_ids_t))  # [audio,prompt]
            attn_mask = input_ids.ge(-1)
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_len,
            }

        # Train view: prompt + answer + eos
        answer = self.answer_template.format(target)
        text = prompt + answer
        text_ids = self.tokenizer.encode(text)
        text_ids.append(self.tokenizer.eos_token_id)
        text_ids = torch.tensor(text_ids, dtype=torch.int64)
        input_ids = torch.cat((audio_pseudo, text_ids))

        labels = copy.deepcopy(input_ids)
        labels[: audio_length + prompt_len] = -1  # ignore audio + prompt
        attn_mask = input_ids.ge(-1)
        label_mask = labels.ge(0)
        input_ids[~attn_mask] = 0
        labels[~label_mask] = self.IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,  # [T, 80]
            "audio_mel_post_mask": None,
            "audio_length": audio_length,
            "prompt_length": prompt_len,
        }


    def pad(self, sequence, max_length: int, padding_idx=0):
        """
        Right-pad/truncate along time dimension to max_length.
        Works for torch.Tensor or np.ndarray shaped [T, ...] or 1D sequences.
        Returns a torch.Tensor.
        """
        if isinstance(sequence, (list, tuple, int)):
            seq = list(sequence) if not isinstance(sequence, int) else [sequence]
            L = len(seq)
            if L < max_length:
                seq = seq + [padding_idx] * (max_length - L)
            else:
                seq = seq[:max_length]
            return torch.tensor(seq)

        if isinstance(sequence, np.ndarray):
            arr = sequence
            L = arr.shape[0]
            if L < max_length:
                pad_shape = (max_length - L,) + arr.shape[1:]
                pad = np.full(pad_shape, padding_idx, dtype=arr.dtype)
                arr = np.concatenate((arr, pad), axis=0)
            else:
                arr = arr[:max_length]
            return torch.from_numpy(arr)

        if isinstance(sequence, torch.Tensor):
            L = sequence.shape[0]
            if L < max_length:
                pad_shape = [max_length - L] + list(sequence.size())[1:]
                pad = torch.full(pad_shape, padding_idx, dtype=sequence.dtype, device=sequence.device)
                sequence = torch.cat((sequence, pad), dim=0)
            else:
                sequence = sequence[:max_length]
            return sequence

        raise TypeError("Unsupported type for padding")

    @classmethod
    def padding(cls, sequence, padding_length: int, padding_idx=0, padding_side="right"):
        """
        Add padding_length items either to left or right along time for sequences/tensors.
        Returns a torch.Tensor.
        """
        if isinstance(sequence, (list, tuple, int)):
            seq = list(sequence) if not isinstance(sequence, int) else [sequence]
            if padding_length >= 0:
                pad = [padding_idx] * padding_length
                seq = pad + seq if padding_side == "left" else seq + pad
            else:
                seq = seq[:padding_length]
            return torch.tensor(seq)

        if isinstance(sequence, np.ndarray):
            arr = sequence
            if padding_length >= 0:
                pad_shape = (padding_length,) + arr.shape[1:]
                pad = np.full(pad_shape, padding_idx, dtype=arr.dtype)
                arr = np.concatenate((pad, arr), axis=0) if padding_side == "left" else np.concatenate((arr, pad), axis=0)
            else:
                arr = arr[:padding_length]
            return torch.from_numpy(arr)

        if isinstance(sequence, torch.Tensor):
            if padding_length >= 0:
                pad_shape = [padding_length] + list(sequence.size())[1:]
                pad = torch.full(pad_shape, padding_idx, dtype=sequence.dtype, device=sequence.device)
                sequence = torch.cat((pad, sequence), dim=0) if padding_side == "left" else torch.cat((sequence, pad), dim=0)
            else:
                sequence = sequence[:padding_length]
            return sequence

        raise TypeError("Unsupported type for padding")

    def collator(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert samples, "Empty batch"

        input_prompt_lengths = [s["audio_length"] + s["prompt_length"] for s in samples]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s["prompt_length"] for s in samples]
        prompt_max = max(input_prompt_lengths)
        answer_max = max(input_answer_lengths)

        input_ids = torch.stack([
            self.padding(
                self.padding(samples[i]["input_ids"], prompt_max - input_prompt_lengths[i],
                             self.tokenizer.pad_token_id, padding_side="left"),
                answer_max - input_answer_lengths[i], self.tokenizer.pad_token_id
            ) for i in range(len(samples))
        ])

        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[i]["attention_mask"], prompt_max - input_prompt_lengths[i],
                             False, padding_side="left"),
                answer_max - input_answer_lengths[i], False
            ) for i in range(len(samples))
        ])

        audio_raw = audio_mask = None
        audio_mel = audio_mel_post_mask = None

        if self.input_type == "raw":
            max_len = max([s["audio"].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s["audio"], max_len, 0) for s in samples])
            audio_mask = torch.zeros(len(samples), max_len, dtype=torch.bool)
            for i, s in enumerate(samples):
                audio_mask[i, : s["audio"].shape[0]] = True
        else:
            max_len = max([s["audio_mel"].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s["audio_mel"], max_len, 0) for s in samples])  # [B, T_max, 80]
            # Keep legacy (T//2) post-mask if downstream expects it
            audio_mel_post_mask = torch.zeros(len(samples), (max_len + 1) // 2, dtype=torch.bool)
            for i, s in enumerate(samples):
                L = (s["audio_mel"].shape[0] + 1) // 2
                audio_mel_post_mask[i, :L] = True

        modality_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for i, s in enumerate(samples):
            pad_left = prompt_max - input_prompt_lengths[i]
            modality_mask[i, pad_left : pad_left + s["audio_length"]] = True

        if "labels" not in samples[0]:
            keys = [s.get("key") for s in samples]
            targets = [s.get("target") for s in samples]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mask": audio_mask if self.input_type == "raw" else None,
                "audio_mel": audio_mel if self.input_type == "mel" else None,
                "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets,
            }

        labels = torch.stack([
            self.padding(
                self.padding(samples[i]["labels"], prompt_max - input_prompt_lengths[i],
                             self.IGNORE_INDEX, padding_side="left"),
                answer_max - input_answer_lengths[i], self.IGNORE_INDEX
            ) for i in range(len(samples))
        ])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mask": audio_mask if self.input_type == "raw" else None,
            "audio_mel": audio_mel if self.input_type == "mel" else None,
            "audio_mel_post_mask": audio_mel_post_mask if self.input_type == "mel" else None,
            "modality_mask": modality_mask,
        }

# Dataset Factory
def get_speech_dataset(dataset_config, tokenizer, split):
    return SpeechDatasetJsonl(dataset_config, tokenizer, split)