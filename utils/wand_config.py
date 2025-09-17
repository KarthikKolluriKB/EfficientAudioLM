from typing import Optional, Dict, Any 
import wandb 

def init_wandb(
        use_wand: bool, 
        project: str,
        run_name: str,
        tags, 
        config: Optional[Dict[str, Any]] = None
        ) -> Optional[wandb.wandb_sdk.wandb_run.Run]:
    """
    Initialize a Weights & Biases (wandb) run if use_wand is True.

    Args:
        use_wand (bool): Whether to initialize wandb.
        project (str): The name of the wandb project.
        run_name (str): The name of the wandb run.
        tags (list): List of tags for the wandb run.
        config (Optional[Dict[str, Any]]): Configuration dictionary to log with wandb.

    Returns:
        Optional[wandb.wandb_sdk.wandb_run.Run]: The initialized wandb run or None if not used.
    """
    # if not using wandb, return None
    if not use_wand:
        return None
    wandb_run = wandb.init(
        project=project,        
        name=run_name,
        tags=tags,
        config=config
    )
    return wandb_run    