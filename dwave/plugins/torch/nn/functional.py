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

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dwave.plugins.torch.graph import GraphIndex
    from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.nn.modules.kernels import Kernel

__all__ = [
    "expectation",
    "gumbel_spins",
    "leave_one_out",
    "maximum_mean_discrepancy_loss",
    "pseudo_kl_divergence_loss",
]


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
        kernel (Kernel): A kernel function object, evaluated on the pooled sample with the
            features of every item flattened to one dimension.

    Raises:
        ValueError: If ``x`` or ``y`` holds fewer than two samples, or their feature shapes differ.

    Returns:
        torch.Tensor: The squared maximum mean discrepancy estimate.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(
            f"Feature shapes of x and y must match, got {tuple(x.shape)} and {tuple(y.shape)}."
        )
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError(
            "x and y must each hold at least two samples, got shapes "
            f"{tuple(x.shape)} and {tuple(y.shape)}."
        )
    num_x = x.shape[0]
    num_y = y.shape[0]
    xy = torch.cat([x.flatten(1), y.flatten(1)], dim=0)
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
    backward pass uses the gradient of the softmax relaxation. This is the discretisation step
    of a discrete variational autoencoder with a Boltzmann machine prior (see
    ``examples/discrete_variational_autoencoder.py``); the default temperature
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
    # The first class indicates s = +1; the straight-through estimator can leave the indicator a
    # rounding error away from {0, 1}.
    return 2 * one_hot[..., 0] - 1


def leave_one_out(
    statistic: Callable[..., torch.Tensor], spins: torch.Tensor, *inputs: torch.Tensor
) -> torch.Tensor:
    """The values of a statistic of a set of samples on every set of all samples but one.

    Args:
        statistic (Callable): A statistic of a set of samples, see :func:`expectation`.
        spins (torch.Tensor): Samples of shape ``(..., M, N)``.
        *inputs (torch.Tensor): Further inputs of the statistic, passed on unchanged.

    Returns:
        torch.Tensor: A tensor of shape ``(..., M, D)`` whose entry ``m`` is the statistic of all
        samples other than sample ``m``. If ``statistic`` has a ``leave_one_out`` method, its
        result is returned; otherwise ``statistic`` is evaluated once per sample.
    """
    if hasattr(statistic, "leave_one_out"):
        return statistic.leave_one_out(spins, *inputs)
    n_samples = spins.shape[-2]
    keep = ~torch.eye(n_samples, dtype=torch.bool, device=spins.device)
    return torch.stack(
        [statistic(spins[..., keep[m], :], *inputs) for m in range(n_samples)], -2
    )


def expectation(
    graph: GraphIndex,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    spins: torch.Tensor,
    statistic: Callable[..., torch.Tensor],
    *inputs: torch.Tensor,
) -> torch.Tensor:
    r"""A statistic of samples of Ising models, with an unbiased gradient of its expectation with
    respect to the biases.

    Let :math:`s_1, \dots, s_M` be independent samples of the Boltzmann distribution
    :math:`p_\theta(s) \propto \exp\{-E_\theta(s)\}` of an Ising model on ``graph`` with biases
    :math:`\theta = (h, J)`, and let :math:`F` be a statistic of the set of samples that is
    symmetric in them. The value returned is :math:`F(s_1, \dots, s_M)`. Its gradient with respect
    to :math:`\theta` is the score-function estimator with leave-one-out weights,

    .. math::

        -\sum_m \big(d_m - \bar d\big)\, T(s_m),
        \qquad d_m = F(s_1, \dots, s_M) - F(s_1, \dots, s_{m-1}, s_{m+1}, \dots, s_M),

    where :math:`T(s) = \nabla_\theta E_\theta(s)` are the sufficient statistics
    (:meth:`~dwave.plugins.torch.graph.GraphIndex.statistics`) and :math:`\bar d` is the mean of
    the :math:`d_m`, so that the weights sum to zero and the intractable
    :math:`\nabla_\theta \log Z` cancels. The estimator is unbiased for
    :math:`\nabla_\theta\,\mathbb E[F(s_1, \dots, s_{M-1})]` for every symmetric :math:`F`; for a
    sample mean or a U-statistic of any degree, such as the squared maximum mean discrepancy, that
    equals :math:`\nabla_\theta\,\mathbb E[F(s_1, \dots, s_M)]`. Behind it is the identity
    :math:`\nabla_\theta\,\mathbb E[F] = -\sum_m \mathrm{Cov}\big(h_m(s_m), T(s_m)\big)` with
    :math:`h_m(s) = \mathbb E[F \mid s_m = s]`: only the first-order influence function of
    :math:`F` enters, whatever the interaction order of :math:`F`, and the leave-one-out
    differences estimate it with the right normalisation. For the mean of a per-sample transform
    :math:`g` the weights are :math:`(g(s_m) - \bar g) / (M - 1)` and the gradient is the unbiased
    sample covariance :math:`-\mathrm{Cov}(g, T)`.

    The gradient with respect to everything else---the parameters of ``statistic`` and the further
    ``inputs``---is the ordinary gradient of the value, i.e. the sample mean of the pathwise
    derivative, which is the gradient of the expectation because the distribution of the samples
    does not depend on them. The samples themselves receive no gradient.

    Args:
        graph (GraphIndex): The graph of the Ising models.
        linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``, or ``(n_nodes,)`` for
            a single model.
        quadratic (torch.Tensor): Quadratic biases of the edges of shape ``(*batch, n_edges)``, or
            ``(n_edges,)``, in the order of the graph's edges.
        spins (torch.Tensor): Samples of shape ``(*batch, M, n_nodes)`` drawn from the models at
            unit inverse temperature, ``M`` per model.
        statistic (Callable): Maps ``spins`` and ``inputs`` to a tensor of shape ``(*batch, D)``
            and is symmetric in the ``M`` samples; for example
            :class:`~dwave.plugins.torch.nn.Mean` or :class:`~dwave.plugins.torch.nn.SquaredMMD`.
            A ``leave_one_out(spins, *inputs)`` method returning the statistics of all samples but
            one, of shape ``(*batch, M, D)``, is used if present; otherwise the statistic is
            evaluated once per sample (see :func:`leave_one_out`).
        *inputs (torch.Tensor): Further inputs of ``statistic``, such as the reference sample of
            :class:`~dwave.plugins.torch.nn.SquaredMMD`.

    Raises:
        ValueError: If the shapes are inconsistent, fewer than two samples per model are given, or
            the statistic does not return one row per model.

    Returns:
        torch.Tensor: ``statistic(spins, *inputs)``, of shape ``(*batch, D)``.
    """
    n_nodes = graph.n_nodes
    if spins.ndim < 2 or spins.shape[-1] != n_nodes:
        raise ValueError(f"spins must have shape (..., M, {n_nodes}), got {tuple(spins.shape)}.")
    batch_shape, n_samples = spins.shape[:-2], spins.shape[-2]
    if n_samples < 2:
        raise ValueError(f"At least two samples per model are required, got {n_samples}.")
    if linear.ndim > 1 and linear.shape[:-1] != batch_shape:
        raise ValueError(
            f"linear must have shape ({n_nodes},) or be batched like spins, "
            f"{(*batch_shape, n_nodes)}, got {tuple(linear.shape)}."
        )
    spins = spins.detach()
    value = statistic(spins, *inputs)
    if value.ndim < 1 or value.shape[:-1] != batch_shape:
        raise ValueError(
            f"The statistic must return one row of values per model, shape {(*batch_shape, 'D')}, "
            f"got {tuple(value.shape)}."
        )
    with torch.no_grad():
        others = leave_one_out(statistic, spins, *inputs)
        expected = (*batch_shape, n_samples, value.shape[-1])
        if tuple(others.shape) != expected:
            raise ValueError(
                f"The leave-one-out statistics must have shape {expected}, got "
                f"{tuple(others.shape)}."
            )
        weights = value.unsqueeze(-2) - others
        weights = weights - weights.mean(-2, keepdim=True)
    energy = graph.energy(spins, linear, quadratic)
    surrogate = -(weights * energy.unsqueeze(-1)).sum(-2)
    return value + surrogate - surrogate.detach()
