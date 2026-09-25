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

from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING

import torch
from dimod import BinaryQuadraticModel
from torch import nn

from dwave.plugins.torch.nn.modules.ising.spin_statistic import IdentityStatistic
from dwave.plugins.torch.utils import GraphIndex, sampleset_to_tensor
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

if TYPE_CHECKING:
    import dimod
    from dwave.plugins.torch.nn.modules.ising.spin_statistic import SpinStatistic

__all__ = ["Ising", "IsingExpectation"]


class IsingExpectation(torch.autograd.Function):
    r"""Computes the sample average statistic of the Ising model.

    This is a helper function to facilitate backpropagation through an Ising layer. Details of the
    Ising layer are in :class:`.Ising`.

    The gradient of the expected output statistic (``statistic`` or :math:`g` mapping from
    :math:`\{\pm 1\} ^N` to real numbers), is the negative covariance between the sufficient statistic
    and :math:`g`. See
    `Graphical Models, Exponential Families, and Variational Inference <https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf>`_
    for a more rigorous treatment.

    .. math::

        \nabla_\theta \mathbb{E} \left[ g(S)\right]  & = \sum_{s\in\{\pm 1\}^d} \nabla_\theta g(s) \frac{\exp \Big \{ -\langle T(s), \theta \rangle \Big \}}{Z(\theta)}

         & = \sum_{s\in\{\pm 1\}^d} \frac{g(s)}{Z(\theta)^2} \Bigg [ -T(s)\exp \Big \{ -\langle T(s), \theta \rangle  \Big\} Z(\theta) + \exp \Big \{ -\langle T(s), \theta\rangle  \Big\} \mathbb{E}\left[ T(S) \right]Z(\theta) \Bigg]

         & = \sum_{s\in\{\pm 1\}^d} \frac{g(s)}{Z(\theta)} \Bigg[ -T(s)\exp \Big \{ -\langle T(s),\theta\rangle  \Big\} + \exp \Big \{ -\langle T(s), \theta\rangle  \Big\} \mathbb{E}\left[ T(S) \right]\Bigg]

         & = -\mathbb{E}\left[g(S)T(S)\right] + \mathbb{E}\left[g(S)\right] \mathbb{E}\left[ T(S) \right]

         & = -\text{Covariance}\left[g(S), T(S)\right].

    The sufficient statistics are the spins :math:`s_i` (paired with the linear biases) and the
    pairwise products :math:`s_i s_j` (paired with the quadratic biases). The latter covariance is
    accumulated as a dense ``(N, N)`` matrix per batch element and masked to the edges of the model.
    """

    @staticmethod
    def forward(
        ctx,
        spins: torch.Tensor,
        statistics: torch.Tensor,
        adjacency: torch.Tensor,
        linear: torch.Tensor,
        quadratic: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the sample average of ``statistics``.

        Args:
            spins: Spins of shape (B, M, N) where B is a batch size, M is the sample size of spin
                vectors per observation in batch, and N is the number of nodes in the Ising model.
            statistics: Output statistic of spins of shape (B, M, D) where D is the dimension of
                output statistics.
            adjacency: Boolean tensor of shape (N, N) that is ``True`` at the entries of the
                quadratic biases that correspond to edges of the Ising model.
            linear: Linear biases of the Ising model with shape (B, N).
            quadratic: Quadratic biases of the Ising model with shape (B, N, N).

        Raises:
            ValueError: If ``statistics.ndim`` != 3.

        Returns:
            torch.Tensor: Sample average of statistics with shape (B, D).
        """
        if statistics.ndim != 3:
            raise ValueError(f"statistics.ndim should be 3. statistics.ndim is {statistics.ndim}")
        ctx.save_for_backward(spins, statistics, adjacency)
        return statistics.mean(-2)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[None, None, None, torch.Tensor, torch.Tensor]:
        """Backpropagation of gradients approximated via sample covariance.

        Args:
            grad_output: Gradients of the loss with respect to the outputs, shape (B, D).

        Returns:
            tuple[None, None, None, torch.Tensor, torch.Tensor]: Gradients with respect to the
            linear and quadratic biases with shapes (B, N) and (B, N, N) respectively. Entries of
            the latter outside ``adjacency`` are zero.
        """
        spins, statistics, adjacency = ctx.saved_tensors
        sample_size = spins.shape[-2]
        # Weight of every sample: the gradient projected onto the centred output statistics.
        # Because the weights sum to zero over samples, covariances with the sufficient statistics
        # can be accumulated from uncentred spins.
        centred = statistics - statistics.mean(-2, keepdim=True)
        weights = torch.einsum("bmy,by->bm", centred, grad_output) / (sample_size - 1)
        grad_linear = -torch.einsum("bm,bmi->bi", weights, spins)
        grad_quadratic = -torch.bmm((spins * weights.unsqueeze(-1)).mT, spins) * adjacency
        return None, None, None, grad_linear, grad_quadratic


class Ising(nn.Module):
    r"""An Ising layer in which inputs are interpreted as Hamiltonian parameters and outputs are
    expected statistics of the system.

    An Ising model is defined by a graph :math:`G = (V, E)` or, equivalently, a set of nodes and
    edges. In implementation, the model is defined by an ordered list of nodes and edges. Inputs
    ``linear`` and ``quadratic`` are interpreted as the linear biases of :math:`V` and the
    quadratic biases of :math:`E`. The quadratic biases are given as a dense ``(B, N, N)`` tensor
    in canonical orientation: the bias of the edge between the nodes with indices ``i < j`` is read
    from ``quadratic[:, i, j]`` (see :attr:`adjacency`); all other entries are ignored. The output
    of the model is the expected output statistic (``statistic`` or :math:`g` below). That is,

    .. math::

        f(x, y) = \sum_{s\in\{\pm 1\}^d} g(s) \frac{\exp \Big \{ -\langle T(s), (x, y) \rangle \Big \}}{Z(x, y)}.

    where

    .. math::

        Z(x, y) = \sum_{s\in\{\pm 1\}^d} \exp \Big \{ -\langle T(s), (x, y) \rangle \Big \}

    is the partition function.

    Model outputs are, in theory, deterministic. In practice, in this implementation, model outputs
    are stochastic due to its computational intractability and thus the need to employ a Monte Carlo
    approximation.

    Inputs ``linear`` and ``quadratic`` should have shape ``(B, |V|)`` and ``(B, |V|, |V|)``
    respectively where ``B`` indicates a batch size. Outputs have shape ``(B, D)`` where ``D`` is
    the output dimension of ``statistic``. Gradients with respect to ``quadratic`` are nonzero only
    at the edges of the model.

    In practice, when sampling using a quantum annealer, samples are not guaranteed to be
    Boltzmann---which is an assumption this implementation leans on. Furthermore, a quantum annealer
    samples at an effective inverse temperature (beta) that is not guaranteed to be 1. This poses a
    problem when the Ising module is used in conjunction with other modules, where---if beta is not
    accounted for---gradient estimates will be off by a factor equal to beta. In other words, the
    effective learning rate of this layer will differ from other parameters by a factor of beta.
    To account for beta, use the method ``set_beta``. To estimate beta, use the method
    ``estimate_betas``. See
    `Global Warming: Temperature Estimation in Annealers <https://doi.org/10.3389/fict.2016.00023>`_.
    for more on estimating beta.

    Args:
        nodes: Nodes of the model.
        edges: Edges of the model.
        sampler: The sampler used to sample from the model.
        sample_params: Keyword arguments used in the ``sampler.sample`` method.
        beta: Effective inverse temperature of the sampler.
        statistic: Function mapping spins to statistics. If None, the statistic corresponds
            to the input nodes and input edges. Defaults to None.
    """

    def __init__(
        self,
        nodes: Iterable[Hashable],
        edges: Iterable[tuple[Hashable, Hashable]],
        sampler: dimod.Sampler,
        sample_params: dict,
        beta: float,
        statistic: SpinStatistic | None = None,
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"Effective inverse temperature beta must be positive. Got {beta}.")

        graph = GraphIndex.from_graph(nodes, edges)
        self._nodes = graph.nodes
        self._edges = graph.edges
        self._sampler = sampler
        self._sample_params = dict(sample_params)

        self.register_buffer("_beta", torch.tensor(float(beta)))
        self.register_buffer("_edge_idx_i", graph.edge_idx_i)
        self.register_buffer("_edge_idx_j", graph.edge_idx_j)
        self.register_buffer("_adjacency", graph.adjacency())

        # Default to identity
        if statistic is None:
            statistic = IdentityStatistic(self.num_nodes)
        self.statistic = statistic
        self.dim_out = statistic.dim_out

    def set_sampler(self, sampler: dimod.Sampler) -> None:
        """Set the sampler to ``sampler``.

        Args:
            sampler: The sampler used to sample from the model.
        """
        self._sampler = sampler

    @property
    def sampler(self) -> dimod.Sampler:
        """Sampler used to sample from the model."""
        return self._sampler

    def set_sample_params(self, sample_params: dict) -> None:
        """Set sampling parameters.

        Args:
            sample_params: Keyword arguments used in the ``sampler.sample`` method.
        """
        self._sample_params = dict(sample_params)

    @property
    def sample_params(self) -> dict:
        """Sampling parameters used to sample from the model."""
        return self._sample_params

    @property
    def nodes(self) -> list[Hashable]:
        """Nodes of the model."""
        return self._nodes

    @property
    def edges(self) -> list[tuple[Hashable, Hashable]]:
        """Edges of the model."""
        return self._edges

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the model."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the model."""
        return len(self._edges)

    @property
    def beta(self) -> torch.Tensor:
        """The effective inverse temperature of the sampler (a scalar tensor)."""
        return self._beta

    def set_beta(self, beta: float) -> None:
        """Set the effective inverse temperature of the sampler.

        Args:
            beta: The effective inverse temperature of the sampler.
        """
        if beta <= 0:
            raise ValueError(f"Effective inverse temperature beta must be positive. Got {beta}.")
        with torch.no_grad():
            self._beta.fill_(float(beta))

    @property
    def edge_idx_i(self) -> torch.Tensor:
        """The smaller node index of each edge, in the order of :attr:`edges`."""
        return self._edge_idx_i

    @property
    def edge_idx_j(self) -> torch.Tensor:
        """The larger node index of each edge, in the order of :attr:`edges`."""
        return self._edge_idx_j

    @property
    def adjacency(self) -> torch.Tensor:
        """Strictly upper-triangular boolean tensor of shape ``(N, N)`` that is ``True`` at the
        entries of the quadratic biases that correspond to edges."""
        return self._adjacency

    def edge_biases(self, quadratic: torch.Tensor) -> torch.Tensor:
        """Extract the per-edge biases from dense quadratic biases.

        Args:
            quadratic: Dense quadratic biases of shape (..., N, N).

        Returns:
            Per-edge biases of shape (..., E) in the order of :attr:`edges`.
        """
        return quadratic[..., self._edge_idx_i, self._edge_idx_j]

    def dense_quadratic(self, edge_biases: torch.Tensor) -> torch.Tensor:
        """Build dense quadratic biases from per-edge biases.

        Args:
            edge_biases: Per-edge biases of shape (..., E) in the order of :attr:`edges`.

        Returns:
            Dense quadratic biases of shape (..., N, N) with ``edge_biases`` at the canonical edge
            positions and zeros elsewhere.
        """
        if edge_biases.shape[-1] != self.num_edges:
            raise ValueError(
                f"Expected {self.num_edges} edge biases, got {edge_biases.shape[-1]}."
            )
        quadratic = edge_biases.new_zeros(*edge_biases.shape[:-1], self.num_nodes, self.num_nodes)
        quadratic[..., self._edge_idx_i, self._edge_idx_j] = edge_biases
        return quadratic

    def _sample(self, linear: torch.Tensor, quadratic: torch.Tensor) -> list[dimod.SampleSet]:
        """Sample from a batch of models defined by ``linear/self.beta`` and ``quadratic/self.beta``
        biases.

        .. note:: Linear and quadratic biases are scaled by ``1/self.beta`` prior to sampling, thus
            extra caution should be taken when estimating beta.

        Args:
            linear: Linear biases of shape (B, N).
            quadratic: Dense quadratic biases of shape (B, N, N).

        Returns:
            A corresponding list of B sample sets.
        """
        linear = (linear.detach() / self._beta).cpu()
        edge_biases = (self.edge_biases(quadratic.detach()) / self._beta).cpu()
        return [
            self._sampler.sample_ising(
                dict(zip(self._nodes, h.tolist())),
                dict(zip(self._edges, J.tolist())),
                **self._sample_params,
            )
            for h, J in zip(linear, edge_biases)
        ]

    def _to_tensor(
        self, sample_sets: Iterable[dimod.SampleSet], device: torch.device | None = None
    ) -> torch.Tensor:
        """Converts a list of sample sets to a tensor.

        Args:
            sample_sets: A list of sample sets.
            device: The device of the constructed tensor. If None, then the resulting tensor is
                constructed on the device of :attr:`beta`. Defaults to None.

        Returns:
            A tensor of shape (B, M, N) where B is the number of sample sets, M is the number of
            samples per sample set and N is the number of nodes in the model.
        """
        if device is None:
            device = self._beta.device
        return torch.stack([sampleset_to_tensor(self._nodes, ss, device) for ss in sample_sets])

    def forward(self, linear: torch.Tensor, quadratic: torch.Tensor) -> torch.Tensor:
        """Approximate the expected output statistics of the Ising model.

        Args:
            linear: Linear biases with shape (B, N) where N is the number of nodes in the model.
            quadratic: Dense quadratic biases with shape (B, N, N); only the entries at
                :attr:`adjacency` are used.

        Raises:
            ValueError: If the inputs do not have the expected shapes.

        Returns:
            Sample-approximation of expected output statistics with shape (B, D) where D is the
            output dimension of the output statistic (the ``statistic`` parameter in the constructor).
        """
        if linear.ndim != 2 or linear.shape[1] != self.num_nodes:
            raise ValueError(
                f"linear should have shape (B, {self.num_nodes}), got {tuple(linear.shape)}."
            )
        if quadratic.shape != (linear.shape[0], self.num_nodes, self.num_nodes):
            raise ValueError(
                f"quadratic should have shape (B, {self.num_nodes}, {self.num_nodes}) = "
                f"{(linear.shape[0], self.num_nodes, self.num_nodes)}, got {tuple(quadratic.shape)}."
            )

        sample_sets = self._sample(linear, quadratic)
        spins = self._to_tensor(sample_sets, linear.device)
        statistics = self.statistic(spins)
        return IsingExpectation.apply(spins, statistics, self._adjacency, linear, quadratic)

    def estimate_betas(self, linear: torch.Tensor, quadratic: torch.Tensor) -> torch.Tensor:
        """Estimate the maximum pseudolikelihood temperature using
        ``dwave.system.temperatures.maximum_pseudolikelihood_temperature``.

        See
        `Global Warming: Temperature Estimation in Annealers <https://doi.org/10.3389/fict.2016.00023>`_.
        for more on estimating beta.

        Args:
            linear: Linear biases of shape (B, N).
            quadratic: Dense quadratic biases of shape (B, N, N).

        Returns:
            Tensor of length B estimates of inverse temperature of the model where B is batch size.
        """
        edge_biases = self.edge_biases(quadratic.detach()).cpu()
        # NOTE: Notice `self.beta` is not used to scale when sampling, c.f., `_sample`.
        bqms = [
            BinaryQuadraticModel.from_ising(
                dict(zip(self._nodes, h.tolist())), dict(zip(self._edges, J.tolist()))
            )
            for h, J in zip(linear.detach().cpu(), edge_biases)
        ]
        sample_sets = [self._sampler.sample(bqm, **self._sample_params) for bqm in bqms]
        return torch.tensor(
            [1 / float(mple(bqm, ss)[0]) for bqm, ss in zip(bqms, sample_sets)]
        )
