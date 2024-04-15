"""
Utility functions for model training and saving
"""

from pathlib import Path
import torch


def save_model(model: torch.nn.Module, model_name: str, target_dir: str):
    """
    Save the model to the given path
    """
    # create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(model.state_dict(), model_save_path)
