import os 
import random 
import torch 
from typing import Optional

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
    """
    Save the projector weights to `path` under the key "projector".

    This function prefers `model.projector` but falls back to
    `model.encoder_projector` for backward compatibility with older code.
    """
    proj = None
    if hasattr(model, "projector") and model.projector is not None:
        proj = model.projector
    elif hasattr(model, "encoder_projector") and model.encoder_projector is not None:
        proj = model.encoder_projector
    else:
        raise AttributeError("Model has no attribute 'projector' or 'encoder_projector' to save.")

    torch.save({"step": step, "projector": proj.state_dict()}, path)
    print(f"Projector saved at step {step} to {path}")
