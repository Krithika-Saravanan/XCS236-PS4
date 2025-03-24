import torch
import torch.autograd as autograd


def create_log_p_theta(
    x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """
    Computes the log probability of a Gaussian distribution with diagonal variance based on the input and parameters.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor for which the log probability is evaluated.
    mean : torch.Tensor
        Mean of the Gaussian distribution.
    log_var : torch.Tensor
        Logarithm of the variance for each component in the Gaussian. The actual variance
        is obtained by exponentiating this value (variance = exp(log_var)).

    Returns
    -------
    torch.Tensor
        The element-wise log probability of x under the specified Gaussian distribution.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def create_log_p_theta_closure(
    theta: Dict[str, torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a closure for the log probability function using a dictionary of parameters.

    Parameters
    ----------
    theta : Dict[str, torch.Tensor]
        A dictionary containing 'mean' and 'log_var' tensors.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        A closure that computes the log probability for a given input tensor.
    """
    ### START CODE HERE ###
    pass
    return log_p_theta_closure
    ### END CODE HERE ###


def compute_score_function(
    log_p_theta: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Jacobian matrix of the log probability function with respect to the input tensor x.

    Parameters
    ----------
    log_p_theta : Callable[[torch.Tensor], torch.Tensor]
        The log probability function for which the Jacobian is computed.
    x : torch.Tensor
        Input tensor with respect to which the Jacobian is computed.

    Returns
    -------
    torch.Tensor
        The Jacobian matrix of the log probability function. The output of calling
        this function will represent the score function.

    Hint: please use autograd.functional.jacobian
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_trace_jacobian(jacobian: torch.Tensor) -> torch.Tensor:
    """
    Computes the trace of the Jacobian matrix.

    Parameters
    ----------
    jacobian : torch.Tensor
        The Jacobian matrix.

    Returns
    -------
    torch.Tensor
        The trace of the Jacobian matrix.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_frobenius_norm_squared(jacobian: torch.Tensor) -> torch.Tensor:
    """
    Computes the Frobenius norm squared of the Jacobian matrix.

    Parameters
    ----------
    jacobian : torch.Tensor
        The Jacobian matrix.

    Returns
    -------
    torch.Tensor
        The Frobenius norm squared of the Jacobian matrix.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the input tensor.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to which noise is added.
    noise_std : float
        The standard deviation of the Gaussian noise.

    Returns
    -------
    torch.Tensor
        The input tensor with added Gaussian noise.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


def compute_score(
    x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor
) -> torch.Tensor:
    """
    Computes the score function, which is the gradient of the log probability of a Gaussian distribution.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor for which the score is computed.
    mean : torch.Tensor
        Mean of the Gaussian distribution.
    log_var : torch.Tensor
        Logarithm of the variance for each component in the Gaussian.

    Returns
    -------
    torch.Tensor
        The score function evaluated at x.
    """
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###
    ### END CODE HERE ###
