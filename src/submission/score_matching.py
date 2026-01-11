from typing import Dict

import torch

from .score_matching_utils import (
    add_noise,
    compute_divergence,
    compute_gaussian_score,
    compute_l2norm_squared,
    compute_score,
    compute_target_score,
    log_p_theta,
)


# Objective Function for Denoising Score Matching
def denoising_score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor], noise_std: float = 0.1
) -> torch.Tensor:
    """Objective function for denoising score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.
        noise_std (float): Standard deviation of the noise to add.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]
    ### START CODE HERE ###
    #pass
    # Add noise
    noisy = add_noise(x, noise_std)
    noisy.requires_grad_(True)

    # Predicted score at noisy points
    log_p = log_p_theta(noisy, mean, log_var)
    pred_score = compute_score(log_p, noisy)

    # Target score
    target_score = (x - noisy) / (noise_std ** 2)

    # DSM loss: mean squared error over batch and dimensions
    loss = torch.mean(torch.sum((pred_score - target_score) ** 2, dim=-1))

    return loss                
    ### END CODE HERE ###


# Objective Function for Score Matching
def score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Objective function for score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]

    ### START CODE HERE ###

    # x must require gradient
    x = x.clone().detach().requires_grad_(True)

    # log p(x)
    log_p = log_p_theta(x, mean, log_var)

    # compute score 
    score = compute_score(log_p, x)

    # compute divergence 
    divergence = compute_divergence(score, x)

    # classical score matching loss
    loss = torch.mean(
        0.5 * compute_l2norm_squared(score) + divergence
    )

    return loss
    #pass
    ### END CODE HERE ###
