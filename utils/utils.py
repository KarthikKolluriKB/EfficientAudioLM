import json
import os 
import random
from types import SimpleNamespace 
import torch 
from typing import Any, Dict, Optional

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# TODO: Need to check if this works correctly (single GPU case) (cuda vs cuda:0)
def get_device() -> str:
    """Get the available device (GPU0 or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

# TODO: need to understand this function
def resolve_pad_token(tokenizer, model) -> int:
    """
    Ensure the tokenizer.pad_token_id is set and syncs it to model configs. 
    If tokenizer.pad_token_id is None, set it to tokenizer.eos_token_id (int). 
    
    Args:
        tokenizer: The tokenizer to check and set the pad_token_id.
        model: The model whose config needs to be updated with the pad_token_id.

    Returns:
        pad_id (int): The resolved pad token ID.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None: 
        # Use tokenizer.eos_token_id as pad_token_id if pad_token_id is not set
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = tokenizer.pad_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            pad_id = tokenizer.pad_token_id
            if hasattr(model, 'resize_token_embeddings'):
                model.resize_token_embeddings(len(tokenizer))
        
        # propagate the pad_token_id to model config
        if hasattr(model, 'config'):
            model.config.pad_token_id = pad_id
        if hasattr(model , 'generation_config'):
            model.generation_config.pad_token_id = pad_id
    return pad_id


def ensure_dir(dir_path: str):
    """Ensure that a directory exists; if not, create it."""
    os.makedirs(dir_path, exist_ok=True)

def save_projector(model, path: str, step: int):
    torch.save({"step": step, "projector": model.projector.state_dict()}, path)
    print(f"Projector saved at step {step} to {path}")

# TODO: Need to implement this function
def load_projector(model, path: str) -> Optional[int]:
    pass

def dict_to_ns(d: Dict[str, Any]) -> SimpleNamespace: 
    """Convert a dictionary to a SimpleNamespace, recursively converting nested dictionaries."""
    out = {} 
    for k, v in d.items():
        out[k] = dict_to_ns(v) if isinstance(v, dict) else v
    return SimpleNamespace(**out)

def import_from_path(spec_str: str): 
    """Import a module or object from a file path."""
    file, name = spec_str.split(":")
    import importlib.util
    spec = importlib.util.spec_from_file_location("dynmod", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)

def load_jsonl(path: str): 
    """Load a JSONL file and return a list of dictionaries."""
    data = [] 
    with open(path, "r", encoding="utf-8") as f: 
        for line in f: 
            data.append(json.loads(line))
    return data