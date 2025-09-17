# load_config.py
from dataclasses import dataclass
from omegaconf import OmegaConf
from typing import Optional

@dataclass
class ModelCfg:
    file: str
    llm_name: str
    llm_path: str
    llm_type: str
    llm_dim: int
    encoder_name: str
    encoder_ds_rate: int
    encoder_path: str
    encoder_dim: int
    encoder_projector: str
    encoder_projector_ds_rate: int
    modal: str
    normalize: bool
    encoder_type: str

@dataclass
class TrainCfg:
    model_name: str
    run_validation: bool
    batch_size_training: int
    batching_strategy: str
    context_length: int
    gradient_accumulation_steps: int
    num_epochs: int
    num_workers_dataloader: int
    warmup_steps: int
    total_steps: int
    validation_interval: int
    lr: float
    weight_decay: float
    gamma: float
    seed: int
    use_fp16: bool
    mixed_precision: bool
    val_batch_size: int
    use_peft: bool
    output_dir: str
    freeze_layers: bool
    num_freeze_layers: int
    quantization: bool
    one_gpu: bool
    save_model: bool
    use_fast_kernels: bool
    freeze_llm: bool
    freeze_encoder: bool

@dataclass
class DataCfg:
    dataset: str
    file: str
    val_data_path: str
    test_split: str
    prompt: Optional[str]
    data_path: Optional[str]
    max_words: Optional[int]
    max_mel: Optional[int]
    fix_length_audio: int
    inference_mode: bool
    input_type: str
    mel_size: int
    normalize: bool

@dataclass
class LogCfg:
    use_wandb: bool
    wandb_dir: str
    wandb_entity_name: str
    wandb_project_name: str
    wandb_exp_name: str
    log_file: str
    log_interval: int

@dataclass
class RootCfg:
    model: ModelCfg
    train: TrainCfg
    data: DataCfg
    log: LogCfg

def load_yaml(path: str) -> RootCfg:
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(RootCfg)
    cfg = OmegaConf.merge(schema, raw)  # type-checked merge
    return cfg  # access as cfg.model.llm_path, etc.
