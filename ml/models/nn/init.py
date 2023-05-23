import torch
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped


@typechecker
def calculate_gain(non_linearity: str) -> float:
    """
    Calculates gain for scaling the standart deviation of the weights

    :param non_linearity: Name of non_linearity
    :return: gain scale factor for weights
    """
    if non_linearity == "relu":
        return 2.0**0.5
    elif non_linearity == "tanh":
        return 5 / 3
    else:
        return 1.0


def kaiming_normal(input: Float[torch.Tensor, "..."], non_linearity: str):
    """
    Re-initializes weights due to `kaiming normal`

    :param input: Tensor of shape `{fan_out, fan_in}`
    :param non_linearity: Name of non_linearity
    :return: Re-initialized tensor
    """
    # We always have weight tensors with {in_features, out_features}
    # So number of fan_in is in .shape[1]
    fan_in = input.shape[1]
    gain = calculate_gain(non_linearity)
    std = gain / fan_in**0.5
    return input.normal_(0, std)


def kaiming_uniform(input: Float[torch.Tensor, "..."], non_linearity: str):
    """
    Re-initializes weights due to `kaiming uniform`

    :param input: Tensor of shape `{fan_out, fan_in}`
    :param non_linearity: Name of non_linearity
    :return: Re-initialized tensor
    """
    fan_in = input.shape[1]
    gain = calculate_gain(non_linearity)
    std = gain / fan_in**0.5
    bound = 3.0**0.5 * std
    return input.uniform_(-bound, bound)
