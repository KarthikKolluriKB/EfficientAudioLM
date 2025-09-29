import os
import json
import random
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
    

def save_and_print_examples(
        hyp_texts: list[str], 
        ref_texts: list[str],
        output_path: str,
        epoch: int,
        n_save: int = 10,
        n_print: int = 5,
        run=None,
        seed: int = 42
) -> None:
    """
    Save random hypothesis/reference pairs to JSONL and print a few examples.

    Args:
        hyp_texts: List of hypothesis (predicted) texts
        ref_texts: List of reference (ground truth) texts
        output_dir: Directory to save the JSONL file
        epoch: Current epoch number
        n_save: Number of examples to save to file (default: 10)
        n_print: Number of examples to print to console (default: 3)
        run: Optional wandb run object for logging
        seed: Random seed for reproducible sampling
    """
    if len(hyp_texts) == 0 or len(ref_texts) == 0:
        print(f"[Epoch {epoch}] No examples to save/print.")
        return

    assert len(hyp_texts) == len(ref_texts), "Hypothesis and reference lists must have same length"

    # set seed for reproducibility sampling 
    random.seed(seed + epoch) # different seed for each epoch

    # Samples random indices (without replacement)
    n_available = len(hyp_texts)
    n_to_sample = min(n_save, n_available)
    sample_indices = random.sample(range(n_available), n_to_sample)

    # Create example records
    examples = []
    for idx in sample_indices: 
        example = {
            "epoch": epoch,
            "sample_index": idx,
            "hyp": hyp_texts[idx],
            "ref": ref_texts[idx]
        }
        examples.append(example)

    # Save to JSONL file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    jsonl_path = os.path.join(output_path, f"epoch_{epoch:03d}_examples.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"[Epoch {epoch}] Saved {n_to_sample} examples to {jsonl_path}")

    # Print a few examples to console
    n_to_print = min(n_print, n_to_sample)
    print_indices = random.sample(range(n_to_sample), n_to_print)

    print(f"\n{'='*80}")
    print(f"Epoch {epoch} - ASR Examples (showing {n_to_print} of {n_to_sample} saved):")
    print(f"{'='*80}")
    
    for i, example in enumerate(print_indices, 1):
        print(f"\n[{i}] Sample #{example['sample_index']}:")
        print(f"  REF: {example['ref']}")
        print(f"  HYP: {example['hyp']}")
    
    # Optional: log to wandb as a table
    if run is not None:
        try: 
            import wandb
            table = wandb.Table(columns=["epoch", "sample_index", "hyp", "ref"])
            for example in examples:
                table.add_data(
                    example["epoch"], 
                    example["sample_index"], 
                    example["hyp"], 
                    example["ref"]
                )
                run.log({f"examples/epoch_{epoch:03d}": table}, commit=False) # avoid creating extra step
                logger.info(f"[Epoch {epoch}] Logged examples to wandb")
        except ImportError:
            logger.warning("wandb is not installed. Skipping wandb logging.")
        except Exception as e:
            logger.warning(f"Failed to log examples to wandb: {e}")
    print(f"{'='*80}\n")