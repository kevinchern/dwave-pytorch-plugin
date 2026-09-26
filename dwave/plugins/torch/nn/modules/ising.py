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

from dwave.plugins.torch.nn.modules.spin_statistic import IdentityStatistic
from dwave.plugins.torch.utils import GraphIndex, estimate_beta

if TYPE_CHECKING:
    from dwave.plugins.torch.nn.modules.spin_statistic import SpinStatistic

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


class Ising(GraphIndex):
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
    approximation: the layer takes spins sampled from the Ising models of its inputs, averages the
    statistic over them, and backpropagates with the sample covariance of the statistic and the
    sufficient statistics (see :class:`IsingExpectation`).

    The layer does not sample. Spins are an input of :meth:`forward` and are drawn with the
    ``sample_biases`` method of any :class:`~dwave.plugins.torch.samplers.TorchSampler` bound to
    the layer, for example

    .. code-block:: python

        ising = Ising(nodes, edges)
        sampler = BlockSampler(ising, schedule=[1.0] * 10)  # or DimodSampler(ising, qpu, ...)

        spins = sampler.sample_biases(linear, quadratic, num_samples=100)  # (B, M, N)
        y = ising(linear, quadratic, spins)  # (B, D)

    Inputs ``linear`` and ``quadratic`` should have shape ``(B, |V|)`` and ``(B, |V|, |V|)``
    respectively where ``B`` indicates a batch size, and ``spins`` should have shape
    ``(B, M, |V|)`` with ``M`` samples per model. Outputs have shape ``(B, D)`` where ``D`` is
    the output dimension of ``statistic``. Gradients with respect to ``quadratic`` are nonzero only
    at the edges of the model; ``spins`` receive no gradient.

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
    `Global Warming: Temperature Estimation in Annealers <https://doi.org/10.3389/fict.2016.00023>`_.
    for more on estimating beta.

    The graph attributes, buffers and the batched :meth:`energy` and :meth:`effective_field` are
    those of :class:`~dwave.plugins.torch.utils.GraphIndex`.

    Args:
        nodes: Nodes of the model.
        edges: Edges of the model.
        statistic: Function mapping spins to statistics. If None, the statistic corresponds
            to the input nodes and input edges. Defaults to None.

    Attributes:
        statistic (SpinStatistic): The output statistic, a submodule.
        dim_out (int): The output dimension of ``statistic``.
    """

    def __init__(
        self,
        nodes: Iterable[Hashable],
        edges: Iterable[tuple[Hashable, Hashable]],
        statistic: SpinStatistic | None = None,
    ) -> None:
        super().__init__(nodes, edges)
        # Default to identity
        if statistic is None:
            statistic = IdentityStatistic(self.n_nodes)
        self.statistic = statistic
        self.dim_out = statistic.dim_out

    def _validate_inputs(
        self, linear: torch.Tensor, quadratic: torch.Tensor, spins: torch.Tensor
    ) -> None:
        """Check the shapes of a batch of biases and their samples.

        Raises:
            ValueError: If the inputs do not have the expected shapes.
        """
        n_nodes = self.n_nodes
        if linear.ndim != 2 or linear.shape[1] != n_nodes:
            raise ValueError(
                f"linear should have shape (B, {n_nodes}), got {tuple(linear.shape)}."
            )
        batch_size = linear.shape[0]
        if quadratic.shape != (batch_size, n_nodes, n_nodes):
            raise ValueError(
                f"quadratic should have shape (B, {n_nodes}, {n_nodes}) = "
                f"{(batch_size, n_nodes, n_nodes)}, got {tuple(quadratic.shape)}."
            )
        if spins.ndim != 3 or spins.shape[0] != batch_size or spins.shape[2] != n_nodes:
            raise ValueError(
                f"spins should have shape (B, M, {n_nodes}) = ({batch_size}, M, {n_nodes}), got "
                f"{tuple(spins.shape)}."
            )

    def forward(
        self, linear: torch.Tensor, quadratic: torch.Tensor, spins: torch.Tensor
    ) -> torch.Tensor:
        """Approximate the expected output statistics of the Ising models from their samples.

        Args:
            linear: Linear biases with shape (B, N) where N is the number of nodes in the model.
            quadratic: Dense quadratic biases with shape (B, N, N); only the entries at
                :attr:`adjacency` are used.
            spins: Spins of shape (B, M, N) sampled from the Boltzmann distributions of the B
                models, M per model, at unit inverse temperature.

        Raises:
            ValueError: If the inputs do not have the expected shapes.

        Returns:
            Sample-approximation of expected output statistics with shape (B, D) where D is the
            output dimension of the output statistic (the ``statistic`` parameter in the constructor).
        """
        self._validate_inputs(linear, quadratic, spins)
        statistics = self.statistic(spins)
        return IsingExpectation.apply(spins, statistics, self.adjacency, linear, quadratic)

    def estimate_betas(
        self, linear: torch.Tensor, quadratic: torch.Tensor, spins: torch.Tensor
    ) -> torch.Tensor:
        """Estimate the inverse temperatures at which spins were sampled from the models, by
        maximum pseudolikelihood (see :func:`~dwave.plugins.torch.utils.estimate_beta`).

        Args:
            linear: Linear biases of shape (B, N).
            quadratic: Dense quadratic biases of shape (B, N, N).
            spins: Spins of shape (B, M, N), M samples per model.

        Returns:
            Tensor of length B with the estimated inverse temperature of every model.
        """
        self._validate_inputs(linear, quadratic, spins)
        edge_biases = self.edge_biases(quadratic.detach())
        return torch.tensor([
            estimate_beta(self.nodes, self.edges, h, J, s)
            for h, J, s in zip(linear.detach(), edge_biases, spins)
        ])
