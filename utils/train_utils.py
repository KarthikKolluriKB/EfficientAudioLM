import logging

logger = logging.getLogger(__name__)


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
