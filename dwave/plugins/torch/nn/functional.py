# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functional interface."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.nn.modules.kernels import Kernel

__all__ = [
    "bit2spin_soft",
    "gumbel_spins",
    "maximum_mean_discrepancy_loss",
    "pseudo_kl_divergence_loss",
    "spin2bit_soft",
]


def _validate_sample_pair(x: torch.Tensor, y: torch.Tensor) -> None:
    """Checks that ``x`` and ``y`` are two samples of at least two items each with equal feature
    shapes, as required by kernels and the maximum mean discrepancy.

    Args:
        x (torch.Tensor): A (n_x, f1, f2, ..., fk) tensor.
        y (torch.Tensor): A (n_y, f1, f2, ..., fk) tensor.

    Raises:
        ValueError: If shape of ``x`` and ``y`` mismatch (excluding batch size).
        ValueError: If the sample size of ``x`` or ``y`` is less than two.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(
            "Input dimensions must match. You are trying to compute "
            f"the kernel between tensors of shape {x.shape} and {y.shape}."
        )
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError(
            "Sample size of ``x`` and ``y`` must be at least two. "
            f"Got, respectively, {x.shape} and {y.shape}."
        )


def maximum_mean_discrepancy_loss(x: torch.Tensor, y: torch.Tensor, kernel: Kernel) -> torch.Tensor:
    r"""Estimates the squared maximum mean discrepancy (MMD) given two samples ``x`` and ``y``.

    The `squared MMD <https://dl.acm.org/doi/abs/10.5555/2188385.2188410>`_ is defined as

    .. math::
        MMD^2(X, Y) = |E_{x\sim p}[\varphi(x)] - E_{y\sim q}[\varphi(y)] |^2,

    where :math:`\varphi` is a feature map associated with the kernel function
    :math:`k(x, y) = \langle \varphi(x), \varphi(y) \rangle`, and :math:`p` and :math:`q` are the
    distributions of the samples. It follows that, in terms of the kernel function, the squared MMD
    can be computed as

    .. math::
        E_{x, x'\sim p}[k(x, x')] + E_{y, y'\sim q}[k(y, y')] - 2E_{x\sim p, y\sim q}[k(x, y)].

    If :math:`p = q`, then :math:`MMD^2(X, Y) = 0`. This motivates the squared MMD as a loss
    function for minimizing the distance between the model distribution and data distribution.

    For more information, see
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. The journal of machine learning research, 13(1), 723-773.

    Args:
        x (torch.Tensor): A (n_x, f1, f2, ..., fk) tensor of samples from distribution p.
        y (torch.Tensor): A (n_y, f1, f2, ..., fk) tensor of samples from distribution q.
        kernel (Kernel): A kernel function object.

    Raises:
        ValueError: If the sample size of ``x`` or ``y`` is less than two.
        ValueError: If shape of ``x`` and ``y`` mismatch (excluding batch size)

    Returns:
        torch.Tensor: The squared maximum mean discrepancy estimate.
    """
    _validate_sample_pair(x, y)
    num_x = x.shape[0]
    num_y = y.shape[0]
    xy = torch.cat([x, y], dim=0)
    kernel_matrix = kernel(xy, xy)
    kernel_xx = kernel_matrix[:num_x, :num_x]
    kernel_yy = kernel_matrix[num_x:, num_x:]
    kernel_xy = kernel_matrix[:num_x, num_x:]
    xx = (kernel_xx.sum() - kernel_xx.trace()) / (num_x * (num_x - 1))
    yy = (kernel_yy.sum() - kernel_yy.trace()) / (num_y * (num_y - 1))
    xy = kernel_xy.sum() / (num_x * num_y)
    return xx + yy - 2 * xy


def pseudo_kl_divergence_loss(
    spins: torch.Tensor,
    logits: torch.Tensor,
    samples: torch.Tensor,
    boltzmann_machine: GraphRestrictedBoltzmannMachine,
) -> torch.Tensor:
    """A pseudo Kullback-Leibler divergence loss function for a discrete autoencoder with a
    Boltzmann machine prior.

    This is not the true KL divergence, but the gradient of this function is the same as
    the KL divergence gradient. See https://arxiv.org/abs/1609.02200 for more details.

    The loss is the average, over the batch, of the energy of the encoder's spins under the
    Boltzmann machine (up to a constant that does not depend on the encoder) minus the entropy of
    the encoder's factorized distribution over the spins of each data point.

    Args:
        spins (torch.Tensor): A tensor of spins of shape (batch_size, n_spins) or shape
            (batch_size, n_samples, n_spins) obtained from a stochastic function that
            maps the output of the encoder (logit representation) to a spin
            representation, e.g. :func:`gumbel_spins`.
        logits (torch.Tensor): A tensor of logits of shape (batch_size, n_spins). These
            logits are the raw output of the encoder.
        samples (torch.Tensor): A tensor of samples from the Boltzmann machine, of shape
            (num_samples, n_spins).
        boltzmann_machine (GraphRestrictedBoltzmannMachine): The Boltzmann machine prior. Any
            object with a ``quasi_objective(s_data, s_model)`` method is accepted.

    Returns:
        torch.Tensor: The computed pseudo KL divergence loss.
    """
    probabilities = torch.sigmoid(logits)
    # Entropy of the factorized encoder distribution of each data point is the *sum* of the
    # per-spin binary entropies; like the energy term below it is then averaged over the batch.
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, probabilities, reduction="none"
    ).flatten(1).sum(-1).mean()
    cross_entropy = boltzmann_machine.quasi_objective(spins, samples)
    return cross_entropy - entropy


def gumbel_spins(logits: torch.Tensor, n_samples: int = 1, tau: float = 1 / 7) -> torch.Tensor:
    r"""Samples spins from the factorized distribution of encoder logits with the straight-through
    Gumbel-softmax estimator.

    Every logit :math:`\ell` defines a spin with :math:`P(s = +1) = \sigma(\ell)`. Spins are drawn
    with a hard two-class Gumbel-softmax over :math:`(\ell, 0)` at temperature ``tau`` (see
    :func:`torch.nn.functional.gumbel_softmax`): the forward pass yields exact spins and the
    backward pass uses the gradient of the softmax relaxation. This is the default
    ``latent_to_discrete`` map of
    :class:`~dwave.plugins.torch.models.DiscreteVariationalAutoencoder`; the default temperature
    is the one used in https://iopscience.iop.org/article/10.1088/2632-2153/aba220.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, l1, l2, ...).
        n_samples (int): Number of spin configurations drawn per row of logits. Defaults to 1.
        tau (float): Temperature of the Gumbel-softmax relaxation. Defaults to ``1/7``.

    Returns:
        torch.Tensor: Spins in ``{-1, +1}`` of shape (batch_size, n_samples, l1, l2, ...),
        differentiable with respect to ``logits``.
    """
    expanded = logits.unsqueeze(1).expand(-1, n_samples, *logits.shape[1:])
    two_class = torch.stack((expanded, torch.zeros_like(expanded)), dim=-1)
    one_hot = torch.nn.functional.gumbel_softmax(two_class, tau=tau, hard=True)
    # The first class indicates s = +1. The straight-through estimator can leave the indicator a
    # rounding error away from {0, 1}, so the exact range check of bit2spin_soft is not used here.
    return 2 * one_hot[..., 0] - 1


def bit2spin_soft(b: torch.Tensor) -> torch.Tensor:
    """Maps input :math:`b` to :math:`2b-1`.

    The mapping does not require :math:`b` to be binary, only that it is in the interval :math:`[0, 1]`.

    Args:
        b (torch.Tensor): Input tensor of values in :math:`[0, 1]`.

    Raises:
        ValueError: If not all ``b`` values are in :math:`[0, 1]`.

    Returns:
        torch.Tensor: A tensor with values :math:`2b-1`.
    """
    if not ((b >= 0) & (b <= 1)).all():
        raise ValueError(f"Not all inputs are in [0, 1]: {b}")
    return b * 2 - 1


def spin2bit_soft(s: torch.Tensor) -> torch.Tensor:
    """Maps input :math:`s` to :math:`(s+1)/2`.

    The mapping does not require :math:`s` to be spin-valued, only that it is in the interval :math:`[-1, 1]`.

    Args:
        s (torch.Tensor): Input tensor of values in :math:`[-1, 1]`.

    Raises:
        ValueError: If not all ``s`` values are in `[-1, 1]`.

    Returns:
        torch.Tensor: A tensor with values :math:`(s+1)/2`.
    """
    if (s.abs() > 1).any():
        raise ValueError(f"Not all inputs are in [-1, 1]: {s}")
    return (s + 1) / 2
