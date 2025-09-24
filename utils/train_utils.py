import os
import json
from omegaconf import OmegaConf
import torch
import transformers
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from omegaconf import DictConfig, OmegaConf, ListConfig

# internal imports
from utils.log_config import get_logger

# Initialize logger
logger = get_logger(log_dir="logs")


def print_model_size(model, config) -> None:
    """
    Logs model name and the number of parameters of the model. 

    Args: 
        model (torch.nn.Module): The model to be evaluated.
        config (dict): Configuration dictionary containing model details.
    """
    logger.info(f"Model: {config.model_name}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{config.model_name} has {total_params/ 1e6} Million trainable parameters")


def print_module_size(module, module_name: str) -> None:
    """
    Logs the module name, the number of parameters of a specific module.

    Args:
        module (torch.nn.Module): The module to be evaluated.
        module_name (str): Name of the module.
    """
    logger.info(f"Module: {module_name}")
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info(f"{module_name} has {total_params/ 1e6} Million trainable parameters")

def convert_to_dict(cfg): 
    if OmegaConf.is_config(cfg): # DictConfig or ListConfig
        # Convert to plain dict or list
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg # already a plain dict or list

def save_training_config(cfg, output_dir):
    """
    Saves the training configuration to a JSON file in the specified output directory.

    Args:
        cfg (omegaconf.DictConfig): The configuration object containing training settings.
        output_dir (str): The directory where the configuration file will be saved.
    
    Returns:
        str: The path to the saved configuration file.
    """
    # mkdir if not exists
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "train_config": convert_to_dict(cfg.get("train")),
        "model_config": convert_to_dict(cfg.get("model")),
        "data_config":  convert_to_dict(cfg.get("data")),
        "log_config":   convert_to_dict(cfg.get("log")),
        "scheduler_config": convert_to_dict(cfg.get("scheduler", {})),
    }

    path = os.path.join(output_dir, "training_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Training configuration saved to {path}")
    return path


def get_lr_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        num_training_steps: int,
        num_warmup_steps: int = 0, 
        num_cycles: float = 0.5,
        **kwargs,
    ) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Returns a learning rate scheduler based on the specified type.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        scheduler_type (str): The type of scheduler to use ("linear" or "cosine").
        num_training_steps (int): Total number of training steps.
        num_warmup_steps (int, optional): Number of warmup steps. Defaults to 0.
        num_cycles (float, optional): Number of cycles for cosine scheduler. Defaults to 0.5.
        **kwargs: Additional keyword arguments for future extensions.

    Returns:
        transformers.PreTrainedScheduler: The configured learning rate scheduler.
    """
    if scheduler_type == "linear_warmup":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    