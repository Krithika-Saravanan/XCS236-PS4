from typing import Dict, Tuple
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from sampling_config import get_config
from utils import get_alpha_bar_prev

def get_timesteps(training_timesteps: int, num_steps: int) -> Tuple[Tensor, Tensor]:
    """
    Generate timesteps for the diffusion process.

    Args:
        training_timesteps (int): Total number of training timesteps.
        num_steps (int): Number of inference steps.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the timesteps and previous timesteps.

    Note: make sure you return -1 for previous timesteps that do not exist
    """
    config = get_config() # useful to get torch device details
    ### START CODE HERE ###
    #pass
    #config = get_config()
    device = config.device

    # Compute evenly spaced timesteps using floor
    step = training_timesteps / num_steps
    timesteps = torch.floor(torch.arange(0, num_steps, device=device) * step).long()

    # Ensure last timestep is exactly training_timesteps - 1
    timesteps[-1] = training_timesteps - 1

    # Previous timesteps: first is -1
    t_prev = torch.full_like(timesteps, -1)
    t_prev[1:] = timesteps[:-1]

    return timesteps, t_prev
    ### END CODE HERE ###

def predict_x0(
    predicted_noise: torch.Tensor,
    t: int,
    sample_t: torch.Tensor,
    scheduler_params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Predict the original image from the noisy sample.

    Args:
        predicted_noise (torch.Tensor): The predicted noise tensor.
        t (int): Current timestep.
        sample_t (torch.Tensor): The noisy sample tensor.
        scheduler_params (Dict[str, torch.Tensor]): Scheduler parameters.

    Returns:
        torch.Tensor: The predicted original image tensor.
    
    Note:
        Make sure you clamp all the returned values in the [-1. 1] range.
    """
    ### START CODE HERE ###
    #pass
    # Get alpha_bar for current timestep
    alphas_bar_t = scheduler_params["alphas_bar"][t]  # scalar tensor

    # Compute x0
    x0_pred = (sample_t - torch.sqrt(1 - alphas_bar_t) * predicted_noise) / torch.sqrt(alphas_bar_t)

    # Clamp values to [-1, 1]
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

    return x0_pred
    ### END CODE HERE ###

def compute_forward_posterior_mean(
    predicted_x0: Tensor,
    noisy_image: Tensor,
    scheduler_params: Dict[str, Tensor],
    t: int,
    t_prev: int,
) -> Tensor:
    """Compute the mean of the forward posterior distribution.

    Args:
        predicted_x0 (Tensor): The predicted original image tensor.
        noisy_image (Tensor): The noisy image tensor.
        scheduler_params (Dict[str, Tensor]): Scheduler parameters.
        t (int): Current timestep.
        t_prev (int): Previous timestep.

    Returns:
        Tensor: The computed mean of the forward posterior distribution.
    """
    ### START CODE HERE ###
    #pass
    if t_prev == -1:
        # First step has no previous, just return x0
        return predicted_x0

    # Scheduler parameters
    alpha_t = scheduler_params["alphas"][t]               
    alpha_t_prev = scheduler_params["alphas"][t_prev]     
    alpha_bar_t = scheduler_params["alphas_bar"][t]       
    alpha_bar_t_prev = scheduler_params["alphas_bar"][t_prev]  

    # Compute forward posterior mean
    mean = (
        torch.sqrt(alpha_bar_t_prev) * (1 - alpha_t) / (1 - alpha_bar_t) * predicted_x0
        + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * noisy_image
    )

    return mean
    ### END CODE HERE ###

def compute_forward_posterior_variance(
    scheduler_params: Dict[str, Tensor], t: int, t_prev: int
) -> Tensor:
    """Compute the variance of the forward posterior distribution.

    Args:
        scheduler_params (Dict[str, Tensor]): Scheduler parameters.
        t (int): Current tim estep.
        t_prev (int): Previous timestep.

    Returns:
        Tensor: The computed variance of the forward posterior distribution.
    """
    ### START CODE HERE ###
    #pass
    if t_prev == -1:
        # First step has no previous timestep; variance is 0
        return torch.tensor(0.0, device=scheduler_params["alphas"].device)

    beta_t = scheduler_params["betas"][t]                  # scalar tensor
    alpha_bar_t = scheduler_params["alphas_bar"][t]       # scalar tensor
    alpha_bar_t_prev = scheduler_params["alphas_bar"][t_prev]  # scalar tensor

    variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
    return variance
    ### END CODE HERE ###

def get_stochasticity_std(
    eta: float, t: int, t_prev: int, alphas_bar: torch.Tensor
) -> torch.Tensor:
    """Calculate the stochasticity standard deviation for DDIM sampling.

    Args:
        eta (float): The DDIM stochasticity parameter (0 = deterministic, 1 = full stochastic).
        t (int): Current timestep.
        t_prev (int): Previous timestep.
        alphas_bar (torch.Tensor): Cumulative product of (1 - beta).

    Returns:
        Tensor: The computed standard deviation.
    """
    ### START CODE HERE ###
    #pass
    if t_prev == -1:
        return torch.tensor(0.0, device=alphas_bar.device)

    abar_t = alphas_bar[t]
    abar_prev = alphas_bar[t_prev]

    # DDIM sigma_t (Eq 16)
    std = eta * torch.sqrt(
        (1 - abar_prev) / (1 - abar_t) * (1 - abar_t / abar_prev)
    )

    return std
    ### END CODE HERE ###

def predict_sample_direction(
    alphas_bars_prev: float, predicted_noise: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Predict the direction for the next sample in DDIM.

    Args:
        alphas_bars_prev (float): Alpha bar value for previous timestep.
        predicted_noise (torch.Tensor): Predicted noise from the model.
        std (torch.Tensor): Standard deviation for the step.

    Returns:
        Tensor: The predicted sample direction.
    """
    ### START CODE HERE ###
    #pass
    direction = torch.sqrt(1 - alphas_bars_prev - std ** 2) * predicted_noise
    return direction
    ### END CODE HERE ###


def stochasticity_term(std: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Compute the stochasticity term for DDIM sampling.

    Args:
        std (torch.Tensor): The computed standard deviation.
        noise (torch.Tensor): Random noise tensor.

    Returns:
        Tensor: The stochasticity term to be added to the sample.
    """
    ### START CODE HERE ###
    #pass
    return std * noise
    ### END CODE HERE ###
