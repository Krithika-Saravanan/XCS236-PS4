import sys
sys.path.append("..") # Adds higher directory to python modules path.
import torch
from torch import Tensor
from typing import Dict
from sampling_config import get_config

def add_forward_tnoise(
    image: Tensor, timestep: int, scheduler_data: Dict[str, Tensor]
) -> Tensor:
    """Add forward timestep noise to the image.

    Args:
        image (Tensor): The input image tensor.
        timestep (int): Current timestep.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.

    Returns:
        x_t (Tensor): The image tensor with added noise.
    """
    config = get_config()
    alpha_bar_at_t = scheduler_data["alphas_bar"][timestep]
    noise = torch.randn(image.shape, device=config.device)
    ### START CODE HERE ###
    #pass
    x_t = torch.sqrt(alpha_bar_at_t) * image + torch.sqrt(1 - alpha_bar_at_t) * noise
    return x_t
    ### END CODE HERE ###

def apply_inpainting_mask(
        original_image: Tensor, 
        noisy_image: Tensor,
        mask: Tensor, 
        timestep, 
        scheduler_data) -> Tensor:
    """Apply the inpainting mask to the image.

    Args:
        image (Tensor): The input image tensor.
        noisy_image (Tensor): The noisy image tensor.
        mask (Tensor): The inpainting mask tensor.
        timestep (int): Current timestep.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.
    Returns:
        Tensor: The inpainted image tensor.
    
    HINT: use add_forward_tnoise to add noise to the original image.
    """
    ### START CODE HERE ###
    #pass
    # Create the noised original image for this timestep
    original_noisy = add_forward_tnoise(original_image, timestep, scheduler_data)

    # Blend: mask=1 keeps noisy_image, mask=0 keeps original_noisy
    x_t = mask * noisy_image + (1 - mask) * original_noisy
    return x_t
    ### END CODE HERE ###

def get_mask(image: Tensor) -> Tensor:
    """Generate a mask for the given image.

    Args:
        image (Tensor): The input image tensor.

    Returns:
        Tensor: The generated mask tensor.
    """
    # Suppose your image is [1, 3, H, W]
    config = get_config() # useful to get torch device details
    ### START CODE HERE ###
    #pass
    B, C, H, W = image.shape
    mask = torch.ones((1, 3, H, W), device=config.device)

    side = H // 2
    start = (H - side) // 2
    end = start + side

    # Center square = known region → mask = 0
    mask[:, :, start:end, start:end] = 0

    return mask
    ### END CODE HERE ###