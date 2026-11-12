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

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable

import torch

from dwave.plugins.torch.graph import GraphIndex
from dwave.plugins.torch.nn.functional import expectation
from dwave.plugins.torch.nn.modules.statistics import Mean
from dwave.plugins.torch.utils import estimate_beta

__all__ = ["Ising"]


class Ising(GraphIndex):
    r"""An Ising layer: the inputs are the biases of Ising models, the output is a statistic of
    samples of the models, and the gradient is that of the statistic's expectation.

    An Ising model on the graph :math:`G = (V, E)` with linear biases :math:`h` and quadratic
    biases :math:`J`, one per node and one per edge in the order of :attr:`nodes` and
    :attr:`edges`, has the Boltzmann distribution

    .. math::

        p_{h, J}(s) = \frac{\exp\{-E_{h, J}(s)\}}{Z(h, J)},
        \qquad E_{h, J}(s) = \sum_{i \in V} h_i s_i + \sum_{(i, j) \in E} J_{ij} s_i s_j,
        \qquad s \in \{\pm 1\}^{|V|}.

    The layer maps the biases to :math:`\mathbb E_{p_{h, J}}[F]`, the expectation of a
    ``statistic`` :math:`F` of :math:`M` independent samples of the model. The expectation is
    intractable, so the layer takes samples drawn from the models by any
    :class:`~dwave.plugins.torch.samplers.TorchSampler` bound to it, returns the statistic of the
    samples, and backpropagates an unbiased estimate of the gradient of the expectation with
    respect to the biases (see :func:`~dwave.plugins.torch.nn.functional.expectation`):

    .. code-block:: python

        ising = Ising(nodes, edges)                          # statistic: the mean spins
        sampler = BlockSampler(ising, schedule=[1.0] * 10)   # or DimodSampler(ising, qpu, ...)

        spins = sampler.sample_biases(linear, quadratic, num_samples=100)  # (B, M, N)
        y = ising(linear, quadratic, spins)                                # (B, N)

    Inputs ``linear`` and ``quadratic`` have shapes ``(*batch, |V|)`` and ``(*batch, |E|)``, one
    model per batch element, and ``spins`` has shape ``(*batch, M, |V|)``; the output has shape
    ``(*batch, D)`` for a statistic with ``D`` outputs. The gradient with respect to ``quadratic``
    has one entry per edge; ``spins`` receive no gradient; the parameters of the statistic and any
    further inputs of the layer receive the ordinary gradient of the value, which is the gradient
    of the expectation with respect to them.

    The statistic is a :class:`torch.nn.Module` (or any callable) of the *set* of samples,
    symmetric in them, mapping ``(*batch, M, |V|)`` spins and optional further inputs to
    ``(*batch, D)`` values: :class:`~dwave.plugins.torch.nn.Mean` for the sample mean of a
    per-sample transform, the default with the identity transform;
    :class:`~dwave.plugins.torch.nn.SquaredMMD` for the squared maximum mean discrepancy to a
    reference sample; or one of your own. Where a loss is placed decides what is differentiated: a
    loss applied to the output of a ``Mean`` layer gives the gradient of the loss of the
    expectation, a loss written as the statistic gives the gradient of the expectation of the loss.

    The gradient estimator assumes that the spins are Boltzmann distributed at unit inverse
    temperature under the given biases. In practice, when sampling using a quantum annealer,
    samples are not guaranteed to be Boltzmann---which is an assumption this implementation leans
    on. Furthermore, a quantum annealer samples at an effective inverse temperature (beta) that is
    not guaranteed to be 1. This poses a problem when the Ising module is used in conjunction with
    other modules, where---if beta is not accounted for---gradient estimates will be off by a
    factor equal to beta. In other words, the effective learning rate of this layer will differ
    from other parameters by a factor of beta. To account for beta, the sampler has to be given
    the biases scaled by ``1/beta``, which is the ``prefactor`` of a
    :class:`~dwave.plugins.torch.samplers.DimodSampler`. To estimate beta from a set of samples,
    use the method :meth:`estimate_betas`. See
    `Global Warming: Temperature Estimation in Annealers <https://doi.org/10.3389/fict.2016.00023>`_
    for more on estimating beta.

    The graph attributes, buffers and the batched :meth:`energy`, :meth:`effective_field` and
    :meth:`statistics` are those of :class:`~dwave.plugins.torch.graph.GraphIndex`.

    Args:
        nodes (Iterable[Hashable]): Nodes of the model.
        edges (Iterable[tuple[Hashable, Hashable]]): Edges of the model.
        statistic (torch.nn.Module, optional): The statistic of a set of samples. Defaults to
            ``Mean()``, the mean spins.

    Attributes:
        statistic (torch.nn.Module): The statistic, a submodule.
    """

    def __init__(
        self,
        nodes: Iterable[Hashable],
        edges: Iterable[tuple[Hashable, Hashable]],
        statistic: torch.nn.Module | Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__(nodes, edges)
        self.statistic = Mean() if statistic is None else statistic

    def forward(
        self,
        linear: torch.Tensor,
        quadratic: torch.Tensor,
        spins: torch.Tensor,
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        """The statistic of the samples, with the gradient of its expectation under the models.

        Args:
            linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``, or ``(n_nodes,)``
                for a single model.
            quadratic (torch.Tensor): Quadratic biases of the edges of shape
                ``(*batch, n_edges)``, or ``(n_edges,)``, in the order of :attr:`edges`.
            spins (torch.Tensor): Spins of shape ``(*batch, M, n_nodes)`` sampled from the Boltzmann
                distributions of the models, ``M`` per model, at unit inverse temperature.
            *inputs (torch.Tensor): Further inputs of the statistic, for example the reference
                sample of :class:`~dwave.plugins.torch.nn.SquaredMMD`.

        Raises:
            ValueError: If the inputs do not have consistent shapes, or fewer than two samples per
                model are given.

        Returns:
            torch.Tensor: The statistic of the samples, of shape ``(*batch, D)``.
        """
        return expectation(self, linear, quadratic, spins, self.statistic, *inputs)

    def estimate_betas(
        self, linear: torch.Tensor, quadratic: torch.Tensor, spins: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the inverse temperatures at which spins were sampled from the models, by
        maximum pseudolikelihood (see :func:`~dwave.plugins.torch.utils.estimate_beta`).

        Args:
            linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)`` or ``(n_nodes,)``.
            quadratic (torch.Tensor): Quadratic biases of the edges of shape ``(*batch, n_edges)``
                or ``(n_edges,)``.
            spins (torch.Tensor): Spins of shape ``(*batch, M, n_nodes)``, ``M`` samples per model.

        Raises:
            ValueError: If the shapes are inconsistent.

        Returns:
            torch.Tensor: The estimated inverse temperature of every model, shape ``(*batch)``.
        """
        n_nodes, n_edges = self.n_nodes, self.n_edges
        if spins.ndim < 2 or spins.shape[-1] != n_nodes:
            raise ValueError(f"spins must have shape (..., M, {n_nodes}), got {tuple(spins.shape)}.")
        batch_shape = spins.shape[:-2]
        try:
            linear = linear.detach().expand(*batch_shape, n_nodes).reshape(-1, n_nodes)
            quadratic = quadratic.detach().expand(*batch_shape, n_edges).reshape(-1, n_edges)
        except RuntimeError:
            raise ValueError(
                f"linear and quadratic must have shapes ({n_nodes},) and ({n_edges},), or be "
                f"batched like spins, got {tuple(linear.shape)}, {tuple(quadratic.shape)} and "
                f"{tuple(spins.shape)}."
            ) from None
        samples = spins.reshape(-1, spins.shape[-2], n_nodes)
        betas = [estimate_beta(self, h, J, s) for h, J, s in zip(linear, quadratic, samples)]
        return torch.tensor(betas).reshape(batch_shape)
