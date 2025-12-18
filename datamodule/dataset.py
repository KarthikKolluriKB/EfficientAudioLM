import json, yaml
import math
import copy 

import numpy as np
import random

import torch
import librosa  
import soundfile as sf

class SpeechDatasetJsonl(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_config: dict,
                 tokenizer=None,
                 split='train',
        )-> None:
        """
        A dataset class for loading speech data and corresponding text from JSONL files.
        """
        super().__init__()
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        data_parallel_size = 1

        self.IGNORE_INDEX = -100
        self.prompt = dataset_config.get("prompt", None)

        # Audio feature parameters
        self.input_type = dataset_config.get("input_type", "mel")
        assert self.input_type in ["raw", "mel", "mfcc", "combined"], "input_type must be one of [raw, mel, mfcc, combined]"

         # Mel spectrogram parameters
        self.mel_size = dataset_config.get("mel_size", 80)
        self.sample_rate = dataset_config.get("sample_rate", 16000)
        self.n_fft = dataset_config.get("n_fft", 400)
        self.hop_length = dataset_config.get("hop_length", 160)
        self.win_length = dataset_config.get("win_length", 400)
        self.fmin = dataset_config.get("fmin", 0)
        self.fmax = dataset_config.get("fmax", 8000)

        # MFCC parameters
        self.n_mfcc = dataset_config.get("n_mfcc", 40)
        
        # Projector parameters (needed for audio_length calculation)
        self.patch_length = dataset_config.get("patch_length", 16)
        self.patch_stride = dataset_config.get("patch_stride", 8)
        
        # Audio processing
        self.max_audio_length = dataset_config.get("max_audio_length", 30.0)  # seconds

        self.prompt_template = "USER: {}\n ASSISTANT:"
        self.answer_template = "{}"
        self.fix_length_audio = dataset_config.get("fix_length_audio", -1)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.normalize = dataset_config.get("normalize", False)
        self.mel_input_norm = dataset_config.get("mel_input_norm", False)
        self.mel_stats_path = dataset_config.get("mel_stats_path", None)
        self.clamp_epsilon = dataset_config.get("clamp_epsilon", 1e-8)

        # mfcc stats
        self.mfcc_input_norm = dataset_config.get("mfcc_input_norm", False)
        self.mfcc_stats_path = dataset_config.get("mfcc_stats_path", None)

        # Loading normalization stats if provided
        if self.mel_input_norm:
            assert self.mel_stats_path is not None
            mel_mean_file = dataset_config.get("mel_mean_file", "mel_means.npy")
            mel_std_file = dataset_config.get("mel_std_file", "mel_stds.npy")
            mel_mean_file = f"{self.mel_stats_path}/{mel_mean_file}"
            mel_std_file = f"{self.mel_stats_path}/{mel_std_file}"
            self.mel_means = torch.from_numpy(np.load(mel_mean_file)).float()
            self.mel_stds = torch.from_numpy(np.load(mel_std_file)).float()

        # MFCC normalization stats
        if self.mfcc_input_norm:
            assert self.mfcc_stats_path is not None
            mfcc_mean_file = dataset_config.get("mfcc_mean_file", "mfcc_means.npy")
            mfcc_std_file = dataset_config.get("mfcc_std_file", "mfcc_stds.npy")
            mfcc_mean_file = f"{self.mfcc_stats_path}/{mfcc_mean_file}"
            mfcc_std_file = f"{self.mfcc_stats_path}/{mfcc_std_file}"
            self.mfcc_means = torch.from_numpy(np.load(mfcc_mean_file)).float()
            self.mfcc_stds = torch.from_numpy(np.load(mfcc_std_file)).float()

        # Load dataset from jsonl files
        self.data_list = []
        if split == "train":
            with open(dataset_config.train_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        elif split == "test":
            with open(dataset_config.test_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)
        else:
            with open(dataset_config.val_data_path, encoding='utf-8') as fin:
                for line in fin:
                    data_dict = json.loads(line.strip())
                    self.data_list.append(data_dict)

    
    def get_source_len(self, data_dict) -> int: 
        """
        Get the length of the source (input) sequence.

        Args: 
            data_dict: dict containing data sample information.

        Returns:
            int: length of the source sequence.
        """
        return data_dict["source_len"]


    def get_target_len(self, data_dict) -> int:
        """
        Get the length of the target (output) sequence.

        Args:
            data_dict: dict containing data sample information.

        Returns:
            int: length of the target sequence.
        """
        return data_dict["target_len"] if "target_len" in data_dict else 0

    def __len__(self) -> int: 
        """
        Return the total number of samples in the dataset.

        Args: 
            None    

        Returns:
            int: total number of samples.
        """
        return len(self.data_list)

    def load_audio(self, audio_path) -> np.ndarray: 
        """
        Load audio file using librosa.

        Args:
            audio_path: path to the audio file.

        Returns:
            numpy array of audio samples.
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio

    def pad_or_trim_audio(self, audio) -> np.ndarray: 
        """
        Pad or trim the audio to a fixed length if specified.

        Args:
            audio: numpy array of audio samples.

        Returns:
            numpy array of audio samples with fixed length.
        """ 
        target_length = int(self.max_audio_length * self.sample_rate) 
        # Pad or trim audio
        if len(audio) > target_length: 
            # Trim audio
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # pad with zeros 
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        return audio

    def extract_log_mel_spectrogram(self, audio) -> torch.Tensor: 
        """
        Extract log-mel spectrogram from audio.

        Args: 
            audio: numpy array of audio samples.

        Returns:
            torch.Tensor of shape (time, n_mels)
        """
        # compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.mel_size,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0 # Power Spectrogram
        )

        # convert to log scale 
        log_mel_spec = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Transpose to (time, freq) format 
        log_mel_spec = torch.from_numpy(log_mel_spec.T).float()

        return log_mel_spec

    def extract_mfcc(self, audio) -> torch.Tensor: 
        """
        Extract MFCC features from audio.

        Args: 
            audio: numpy array of audio samples.

        Returns:
            torch.Tensor of shape (time, n_mfcc)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax
        )

        # Transpose to (time, n_mfcc) format 
        mfcc = torch.from_numpy(mfcc.T).float()

        return mfcc

    def calculate_audio_length(self, num_frames) -> int:
        """Calculate the audio length after patching."""
        if self.fix_length_audio > 0:
            return self.fix_length_audio
    
        # Account for padding that happens in projector
        padded_frames = math.ceil(num_frames / self.patch_length) * self.patch_length
        audio_length = (padded_frames - self.patch_length) // self.patch_stride + 1
        return max(1, audio_length)

    def __getitem__(self, index: int) -> dict: 
        """
        """ 
        data_dict = self.data_list[index]
        audio_path = data_dict.get("source")
        target = data_dict.get("target", None)
        task = data_dict.get("prompt", None)
        key = data_dict.get("key", None)

        # Load audio 
        audio_raw = self.load_audio(audio_path)

        if self.input_type == "raw": 
            audio_raw = torch.from_numpy(audio_raw) 
            if self.normalize: 
                audio_raw = torch.nn.functional.layer_norm(audio_raw, audio_raw.shape)
                audio_length = len(audio_raw) // 320  # assuming 20ms frames at 16kHz
                audio_length = audio_length // 5 
                audio_mel = None 

        elif self.input_type == "mel":
            # Pad or trim audio 
            audio_raw = self.pad_or_trim_audio(audio_raw)
             
            # Extract log-mel spectrogram
            audio_mel = self.extract_log_mel_spectrogram(audio_raw)

            # Apply normalization if enabled
            if self.mel_input_norm:
                audio_mel = (audio_mel - self.mel_means) / (self.mel_stds + self.clamp_epsilon)

            # Calculate audio length for your projector
            audio_length = self.calculate_audio_length(audio_mel.shape[0])

        elif self.input_type == "mfcc":
            # Pad or trim audio 
            audio_raw = self.pad_or_trim_audio(audio_raw)
             
            # Extract MFCC features
            audio_mel = self.extract_mfcc(audio_raw)

            # Apply normalization if enabled
            if self.mfcc_input_norm:
                audio_mel = (audio_mel - self.mfcc_means) / (self.mfcc_stds + self.clamp_epsilon)

            # Calculate audio length for your projector
            audio_length = self.calculate_audio_length(audio_mel.shape[0])

        elif self.input_type == "combined":
            # pad or trim audio 
            audio_raw = self.pad_or_trim_audio(audio_raw)

            # Extract log-mel spectrogram
            mel_features = self.extract_log_mel_spectrogram(audio_raw)

            # Apply normalization if enabled
            if self.mel_input_norm:
                mel_features = (mel_features - self.mel_means) / (self.mel_stds + self.clamp_epsilon)

            # Extract MFCC features
            mfcc_features = self.extract_mfcc(audio_raw)

            # Ensure both features have the same time dimension
            min_length = min(mel_features.shape[0], mfcc_features.shape[0])
            mel_features = mel_features[:min_length, :]
            mfcc_features = mfcc_features[:min_length, :]

            # Concatenate along feature dimension
            audio_mel = torch.cat((mel_features, mfcc_features), dim=1)

            # calculate Sequence length for your projector
            audio_length = self.calculate_audio_length(audio_mel.shape[0])

        audio_pseudo = torch.full((audio_length,), -1)

        # Prompt 
        prompt = self.prompt 
        if prompt is None: 
            prompt = "Transcribe speech to text. Output the transcription directly without redundant content. Ensure that the output is not duplicated. "

        prompt = self.prompt_template.format(prompt.strip())
        prompt_ids = self.tokenizer.encode(prompt) 
        prompt_length = len(prompt_ids)

        if self.inference_mode: 
            prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64) 
            example_ids = torch.cat((audio_pseudo, prompt_ids))
            example_mask = example_ids.ge(-1)

            return {
                "input_ids": example_ids,
                "attention_mask": example_mask,
                "audio": audio_raw if self.input_type == "raw" else None,
                "audio_mel": audio_mel, 
                "audio_length": audio_length,
                "key": key,
                "target": target,
                "prompt_length": prompt_length,
            }

        # Answer 
        answer = self.answer_template.format(target)
        example = prompt + answer 
        example_ids = self.tokenizer.encode(example)
        example_ids.append(self.tokenizer.eos_token_id)
        example_ids = torch.tensor(example_ids, dtype=torch.int64)
        example_ids = torch.cat((audio_pseudo, example_ids))

        # Create attention mask
        labels_ids = copy.deepcopy(example_ids)
        labels_ids[:audio_length + prompt_length] = -1 
        example_mask = example_ids.ge(-1)

        label_mask = labels_ids.ge(0)
        example_ids[~example_mask] = 0  # padding token id is 0
        labels_ids[~label_mask] = self.IGNORE_INDEX

        return {
            "input_ids": example_ids,
            "attention_mask": example_mask,
            "labels": labels_ids,
            "audio": audio_raw if self.input_type == "raw" else None,
            "audio_mel": audio_mel, 
            "audio_length": audio_length,
            "prompt_length": prompt_length,
        }

    def pad(self, sequence, max_length, padding_idx=0):
        """
        Pad or trim a sequence to max_length
        """
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                sequence = sequence + [padding_idx] * (max_length - len(sequence))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                sequence = torch.cat(
                    (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                sequence = np.concatenate(
                    (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    @classmethod
    def padding(cls, sequence, padding_length, padding_idx=0, padding_side="right"):
        """
        Add padding to a sequence (or trim if negative padding_length)
        Supports left and right padding
        """
        if isinstance(sequence, (int, list, tuple)):
            if padding_length >= 0:
                sequence = sequence + [padding_idx] * padding_length
            else:
                sequence = sequence[:padding_length]
        elif isinstance(sequence, torch.Tensor):
            if sequence.ndimension() == 2:
                if padding_length >= 0:
                    sequence = torch.nn.functional.pad(sequence, (0, 0, 0, padding_length))
                else:
                    sequence = sequence[:, :padding_length]
            else:
                if padding_length >= 0:
                    if padding_side == "left":
                        sequence = torch.cat((torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx), sequence))
                    else:
                        sequence = torch.cat((sequence, torch.full(([padding_length] + list(sequence.size())[1:]), padding_idx)))
                else:
                    sequence = sequence[:padding_length]
        elif isinstance(sequence, np.ndarray):
            if padding_length >= 0:
                sequence = np.concatenate(
                    (sequence, np.full((padding_length,) + sequence.shape[1:], padding_idx)))
            else:
                sequence = sequence[:padding_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def collator(self, samples):
        """
        Collate batch samples with proper padding
        Supports: raw audio, mel, mfcc, and combined features
        """
        assert samples is not None 
        
        # Calculate lengths
        input_prompt_lengths = [s["audio_length"] + s['prompt_length'] for s in samples]
        input_answer_lengths = [len(s["input_ids"]) - s["audio_length"] - s['prompt_length'] for s in samples]

        input_prompt_max_length = max(input_prompt_lengths)
        input_answer_max_length = max(input_answer_lengths)
        
        # Pad input_ids
        input_ids = torch.stack([
            self.padding(
                self.padding(samples[index]["input_ids"], input_prompt_max_length - input_prompt_lengths[index], self.tokenizer.pad_token_id, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.tokenizer.pad_token_id
            ) for index in range(len(samples))
        ])

        # Pad attention_mask
        attention_mask = torch.stack([
            self.padding(
                self.padding(samples[index]["attention_mask"], input_prompt_max_length - input_prompt_lengths[index], False, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], False
            ) for index in range(len(samples))
        ])

        # Handle different feature types
        if self.input_type == "raw":
            # Raw audio processing
            audio_raw_max_length = max([s['audio'].shape[0] for s in samples])
            audio_raw = torch.stack([self.pad(s['audio'], audio_raw_max_length, 0)
                                    for s in samples])
            audio_mask = torch.zeros(len(samples), audio_raw_max_length)
            for line, sample in enumerate(samples):
                audio_mask[line, :sample['audio'].shape[0]] = 1
            
            audio_mel = None
            audio_mel_post_mask = None
            
        else:
            # Feature processing (mel, mfcc, or combined)
            audio_mel_max_length = max([s['audio_mel'].shape[0] for s in samples])
            audio_mel = torch.stack([self.pad(s['audio_mel'], audio_mel_max_length, 0)
                                for s in samples])
            
            # Create post-mask based on actual audio_length from projector
            max_audio_length = max([s['audio_length'] for s in samples])
            audio_mel_post_mask = torch.zeros(len(samples), max_audio_length)
            for line, sample in enumerate(samples):
                audio_mel_post_mask[line, :sample['audio_length']] = 1
            
            audio_raw = None
            audio_mask = None

        # Create modality mask
        modality_mask = torch.zeros_like(attention_mask)
        for index in range(len(samples)):
            padding_left = input_prompt_max_length - input_prompt_lengths[index]
            modality_mask[index, padding_left:padding_left+samples[index]["audio_length"]] = True

        # Return for inference mode
        if self.inference_mode:
            keys = [s['key'] for s in samples]
            targets = [s['target'] for s in samples]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "audio": audio_raw,
                "audio_mask": audio_mask,
                "audio_mel": audio_mel,
                "audio_mel_post_mask": audio_mel_post_mask,
                "modality_mask": modality_mask,
                "keys": keys,
                "targets": targets
            }

        # Pad labels for training
        labels = torch.stack([
            self.padding(
                self.padding(samples[index]['labels'], input_prompt_max_length - input_prompt_lengths[index], self.IGNORE_INDEX, padding_side="left"),
                input_answer_max_length - input_answer_lengths[index], self.IGNORE_INDEX)
            for index in range(len(samples))
        ])
        
        # Return for training mode
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "audio": audio_raw,
            "audio_mask": audio_mask,
            "audio_mel": audio_mel,
            "audio_mel_post_mask": audio_mel_post_mask,
            "modality_mask": modality_mask
        }


def get_speech_dataset(dataset_config, tokenizer, split):
    """
    Factory function to create dataset
    """
    dataset = SpeechDatasetJsonl(dataset_config, tokenizer, split)
    return dataset