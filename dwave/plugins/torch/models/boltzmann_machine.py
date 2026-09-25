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
#
# The use of the Boltzmann Machine implementations below (including the
# GraphRestrictedBoltzmannMachine) with a quantum computing system is
# protected by the intellectual property rights of D-Wave Quantum Inc.
# and its affiliates.
#
# The use of the Boltzmann Machine implementations below (including the
# GraphRestrictedBoltzmannMachine) with D-Wave's quantum computing
# system will require access to D-Wave’s LeapTM quantum cloud service and
# will be governed by the Leap Cloud Subscription Agreement available at:
# https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/
#

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Hashable, Iterable, Literal, Optional

import torch
from dimod import BinaryQuadraticModel
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

from dwave.plugins.torch.utils import GraphIndex

if TYPE_CHECKING:
    from dwave.plugins.torch.samplers.base import TorchSampler

__all__ = ["GraphRestrictedBoltzmannMachine"]


class GraphRestrictedBoltzmannMachine(torch.nn.Module):
    r"""Creates a graph-restricted Boltzmann machine.

    A graph-restricted Boltzmann machine (GRBM) is an Ising model on the nodes and edges of a
    graph with energy

    .. math::

        E(s) = \sum_{i} h_i s_i + \sum_{(i, j) \in \mathcal{E}} J_{ij} s_i s_j,
        \qquad s \in \{-1, +1\}^N.

    The quadratic biases are stored in a dense ``(n_nodes, n_nodes)`` matrix :attr:`quadratic`
    in *canonical orientation*: the bias of edge ``(u, v)`` lives at ``[i, j]`` with
    ``i = min(idx(u), idx(v))`` and ``j = max(idx(u), idx(v))``, so the matrix is strictly upper
    triangular. Entries that do not correspond to an edge are structurally zero: a fixed boolean
    :attr:`adjacency` mask is applied whenever the matrix is used, so those entries never
    contribute to the energy and receive exactly zero gradient. All computations---energies,
    batch statistics for learning and effective fields for sampling---are dense matrix products,
    which makes them fast on GPUs and independent of the orientation and order of the edge list.

    The initialization strategy is grounded in `Hinton's practical guide for RBM training
    <https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf>`_, which recommends sampling weights
    from a Gaussian distribution with mean 0 and small standard deviation. The quadratic weights
    are initialized with graph-connectivity-dependent standard deviations so the energy remains
    extensive on sparse graphs as well as dense graphs. In particular, for edge :math:`(u, v)`,
    we set the standard deviation of its J value as :math:`ß / (\deg(u)\deg(v))^{1/4}`, where
    :math:`ß=2.5` is half of a representative QPU inverse sampling-temperature scale. This
    initializes the GRBM in a paramagnetic regime, consistent with the `Sherrington-Kirkpatrick
    model <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.35.1792>`_.
    The linear biases are initialized to zero to avoid introducing any initial preference for spin
    configurations.

    Hidden units are nodes that are not observed in the data. Observed spins of a model with
    hidden units have one column per *visible* node, in the order of :attr:`visible_idx`; use
    :meth:`pad_visible` to embed them into the full node order with ``torch.nan`` marking the
    hidden units.

    Args:
        nodes (Iterable[Hashable]): List of nodes.
        edges (Iterable[tuple[Hashable, Hashable]]): List of edges. Self-loops and duplicate
            edges (in either orientation) are rejected.
        hidden_nodes (Iterable[Hashable], optional): List of hidden nodes. Each hidden node must
            also be listed in ``nodes``.
        linear (dict[Hashable, float], optional): A dictionary mapping from nodes of the
            model to its corresponding linear bias.
        quadratic (dict[tuple[Hashable, Hashable], float], optional): A dictionary mapping from
            edges of the model to its corresponding quadratic bias.

    Raises:
        ValueError: If ``nodes`` contains duplicates, an edge references an unknown node, an edge
            is a self-loop or a duplicate, or a hidden node is not a node of the model.
    """
    # QPU beta has been measured to be 5-8 (in inverse units of programmed J)
    # Considering the higher temperature within this range, to sample from a beta=1
    # Boltzmann distribution, a prefactor of 5 has to multiply the initial Hamiltonian.
    # To keep the energy scale of the initial Hamiltonian below the effective thermal
    # energy, we multiply the Hamiltonian weights by an even smaller prefactor so
    # that the prepared distribution is that of a paramagnet.
    _INIT_INVERSE_TEMP = 2.5

    def __init__(
        self,
        nodes: Iterable[Hashable],
        edges: Iterable[tuple[Hashable, Hashable]],
        hidden_nodes: Optional[Iterable[Hashable]] = None,
        linear: Optional[dict[Hashable, float]] = None,
        quadratic: Optional[dict[tuple[Hashable, Hashable], float]] = None,
    ) -> None:
        super().__init__()

        graph = GraphIndex.from_graph(nodes, edges)
        self._nodes = graph.nodes
        self._edges = graph.edges
        self._node_to_idx = graph.node_to_idx
        self._idx_to_node = dict(enumerate(self._nodes))
        self._edge_to_idx = {edge: k for k, edge in enumerate(self._edges)}
        self._n_nodes = graph.n_nodes
        self._n_edges = graph.n_edges

        self.register_buffer("_edge_idx_i", graph.edge_idx_i)
        self.register_buffer("_edge_idx_j", graph.edge_idx_j)
        self.register_buffer("_adjacency", graph.adjacency())

        quadratic_init = torch.zeros(self._n_nodes, self._n_nodes)
        if self._n_edges:
            degrees = graph.degrees().to(quadratic_init.dtype)
            quadratic_std = self._INIT_INVERSE_TEMP / (
                degrees[self._edge_idx_i] * degrees[self._edge_idx_j]
            )**0.25
            quadratic_init[self._edge_idx_i, self._edge_idx_j] = (
                torch.randn(self._n_edges) * quadratic_std
            )
        self._linear = torch.nn.Parameter(torch.zeros(self._n_nodes))
        self._quadratic = torch.nn.Parameter(quadratic_init)

        self._hidden_nodes = [] if hidden_nodes is None else list(hidden_nodes)
        hidden_set = set(self._hidden_nodes)
        if len(hidden_set) != len(self._hidden_nodes):
            raise ValueError("`hidden_nodes` contains duplicate entries.")
        unknown = hidden_set.difference(self._nodes)
        if unknown:
            raise ValueError(f"Hidden nodes {list(unknown)!r} are not nodes of the model.")
        is_hidden = torch.tensor([v in hidden_set for v in self._nodes], dtype=torch.bool)
        self.register_buffer("_visible_idx", torch.nonzero(~is_hidden).flatten())
        self.register_buffer("_hidden_idx", torch.nonzero(is_hidden).flatten())
        self._connected_hidden = bool(
            (is_hidden[self._edge_idx_i] & is_hidden[self._edge_idx_j]).any()
        )

        if linear is not None:
            self.set_linear(linear)
        if quadratic is not None:
            self.set_quadratic(quadratic)

    def extra_repr(self) -> str:
        return (
            f"n_nodes={self._n_nodes}, n_edges={self._n_edges}, "
            f"n_hidden={len(self._hidden_nodes)}"
        )

    # ------------------------------------------------------------------ parameters --------------

    def set_linear(self, linear: dict[Hashable, float]) -> None:
        """Set linear biases of the model.

        Args:
            linear (dict[Hashable, float]): A dictionary mapping from nodes of the model to its
                corresponding linear bias. Not all linear biases need to be set; nodes without a
                mapping keep their current values.

        Raises:
            ValueError: If a node is not in the model.
        """
        if not linear:
            return
        try:
            node_idx = [self._node_to_idx[node] for node in linear]
        except KeyError as err:
            raise ValueError(f"Node {err.args[0]!r} is not in the model.") from None
        device = self._linear.device
        values = torch.tensor(list(linear.values()), dtype=self._linear.dtype, device=device)
        with torch.no_grad():
            self._linear[torch.tensor(node_idx, device=device)] = values

    def set_quadratic(self, quadratic: dict[tuple[Hashable, Hashable], float]) -> None:
        """Set quadratic biases of the model.

        Args:
            quadratic (dict[tuple[Hashable, Hashable], float]): A dictionary mapping from edges of
                the model, in either orientation, to its corresponding quadratic bias. Not all
                quadratic biases need to be set; edges without a mapping keep their current values.

        Raises:
            ValueError: If an edge is not in the model.
        """
        if not quadratic:
            return
        edge_idx = []
        for u, v in quadratic:
            idx = self._edge_to_idx.get((u, v), self._edge_to_idx.get((v, u)))
            if idx is None:
                raise ValueError(f"Edge {(u, v)!r} is not in the model.")
            edge_idx.append(idx)
        device = self._quadratic.device
        edge_idx = torch.tensor(edge_idx, device=device)
        values = torch.tensor(
            list(quadratic.values()), dtype=self._quadratic.dtype, device=device
        )
        with torch.no_grad():
            self._quadratic[self._edge_idx_i[edge_idx], self._edge_idx_j[edge_idx]] = values

    @property
    def linear(self) -> torch.nn.Parameter:
        """The linear biases of the model, shape ``(n_nodes,)``."""
        return self._linear

    @property
    def quadratic(self) -> torch.nn.Parameter:
        """The quadratic biases of the model as a dense ``(n_nodes, n_nodes)`` tensor.

        The bias of the edge between the nodes with indices ``i < j`` is stored at ``[i, j]``.
        Entries outside :attr:`adjacency` are ignored by every computation. The per-edge biases,
        in the order of :attr:`edges`, are ``quadratic[edge_idx_i, edge_idx_j]``."""
        return self._quadratic

    @property
    def adjacency(self) -> torch.Tensor:
        """Strictly upper-triangular boolean tensor of shape ``(n_nodes, n_nodes)`` that is
        ``True`` exactly at the entries of :attr:`quadratic` that correspond to edges."""
        return self._adjacency

    def coupling(self) -> torch.Tensor:
        """The coupling matrix :math:`J` with off-graph entries forced to zero.

        Returns:
            torch.Tensor: A strictly upper-triangular ``(n_nodes, n_nodes)`` tensor.
        """
        return self._quadratic * self._adjacency

    def symmetric_coupling(self) -> torch.Tensor:
        """The symmetrized coupling matrix :math:`J + J^T`, whose row ``k`` holds the couplings
        of node ``k`` to every other node.

        Returns:
            torch.Tensor: A symmetric ``(n_nodes, n_nodes)`` tensor with zero diagonal.
        """
        coupling = self.coupling()
        return coupling + coupling.mT

    # ------------------------------------------------------------------ graph --------------------

    @property
    def nodes(self) -> list[Hashable]:
        """List of nodes in the model. This list includes both visible and hidden nodes."""
        return self._nodes

    @property
    def hidden_nodes(self) -> list[Hashable]:
        """List of hidden nodes in the model."""
        return self._hidden_nodes

    @property
    def visible_nodes(self) -> list[Hashable]:
        """List of visible nodes in the model, in the order of :attr:`visible_idx`."""
        return [self._nodes[idx] for idx in self._visible_idx.tolist()]

    @property
    def edges(self) -> list[tuple[Hashable, Hashable]]:
        """List of edges in the model, in the orientation given at construction."""
        return self._edges

    @property
    def node_to_idx(self) -> dict[Hashable, int]:
        """A dictionary mapping from node to index of model variables."""
        return self._node_to_idx

    @property
    def idx_to_node(self) -> dict[int, Hashable]:
        """A dictionary mapping from index of model variables to nodes."""
        return self._idx_to_node

    @property
    def n_nodes(self) -> int:
        """Total number of model variables or graph nodes (including hidden units)."""
        return self._n_nodes

    @property
    def n_edges(self) -> int:
        """Total number of edges in the model or graph."""
        return self._n_edges

    @property
    def n_visible(self) -> int:
        """Number of visible units."""
        return self._visible_idx.numel()

    @property
    def n_hidden(self) -> int:
        """Number of hidden units."""
        return self._hidden_idx.numel()

    @property
    def visible_idx(self) -> torch.Tensor:
        """A ``torch.Tensor`` of model variable indices corresponding to visible units."""
        return self._visible_idx

    @property
    def hidden_idx(self) -> torch.Tensor:
        """A ``torch.Tensor`` of model variable indices corresponding to hidden units."""
        return self._hidden_idx

    @property
    def connected_hidden(self) -> bool:
        """Whether any edge connects two hidden units."""
        return self._connected_hidden

    @property
    def edge_idx_i(self) -> torch.Tensor:
        """The smaller node index of each edge, in the order of :attr:`edges`."""
        return self._edge_idx_i

    @property
    def edge_idx_j(self) -> torch.Tensor:
        """The larger node index of each edge, in the order of :attr:`edges`."""
        return self._edge_idx_j

    # ------------------------------------------------------------------ energies -----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the Hamiltonian.

        Args:
            x (torch.Tensor): A tensor of shape (..., N) where N denotes the number of variables in
                the model.

        Returns:
            torch.Tensor: Hamiltonians of shape (...,).
        """
        coupling = self.coupling()
        return x @ self._linear + ((x @ coupling) * x).sum(-1)

    def effective_field(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Effective fields :math:`h_k + \sum_{l} J_{kl} s_l` acting on nodes.

        The effective field of node :math:`k` is the derivative of the energy with respect to
        :math:`s_k`; the conditional distribution of :math:`s_k` given all other spins is
        :math:`P(s_k = \pm 1) = \sigma(\mp 2 h^{\text{eff}}_k)`. Entries of ``x`` that are
        ``torch.nan`` denote unknown spins and contribute nothing to the fields, which makes this
        method suitable both for block-Gibbs sampling (all spins known) and for conditioning
        hidden units on visible units (hidden entries ``torch.nan``).

        Args:
            x (torch.Tensor): Spins of shape (..., N) where N denotes the number of variables in
                the model; ``torch.nan`` marks unknown spins.
            idx (torch.Tensor, optional): Indices of the nodes whose fields are returned. If
                ``None``, the fields of all nodes are returned. Defaults to ``None``.

        Returns:
            torch.Tensor: Effective fields of shape (..., N) or (..., ``len(idx)``).
        """
        spins = torch.nan_to_num(x, nan=0.0)
        coupling = self.symmetric_coupling()
        if idx is None:
            return self._linear + spins @ coupling
        return self._linear[idx] + spins @ coupling[:, idx]

    def moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-averaged first and (uncentred) second moments of spins.

        These are the dense sufficient statistics of the model: the average energy of the batch
        is ``mean @ linear + (coupling() * second_moment).sum()``.

        Args:
            x (torch.Tensor): Spins of shape (..., N) where N denotes the number of variables in
                the model. All leading dimensions are averaged over.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensors of shape (N,) and (N, N) holding the
            average spins and the average pairwise products.
        """
        x = self._flatten(x)
        return x.mean(0), (x.mT @ x) / x.shape[0]

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten all leading dimensions of a (..., N) tensor into one batch dimension."""
        if x.shape[-1] != self._n_nodes:
            raise ValueError(
                f"Expected spins with trailing dimension {self._n_nodes}, got {x.shape[-1]}."
            )
        return x.reshape(-1, self._n_nodes)

    # ------------------------------------------------------------------ learning -----------------

    def quasi_objective(
        self,
        s_observed: torch.Tensor,
        s_model: torch.Tensor,
        kind: Optional[Literal["sampling", "exact-disc"]] = None,
        *,
        sampler: Optional[TorchSampler] = None,
    ) -> torch.Tensor:
        """A quasi-objective function with gradients equivalent to the gradients of the
        negative log likelihood.

        The objective is the difference between the average energy of the observed spins and the
        average energy of the model spins. Its gradient with respect to the linear and quadratic
        biases is the difference of the average sufficient statistics of data and model, i.e. the
        gradient of the negative log likelihood.

        Args:
            s_observed (torch.Tensor): Tensor of observed spins (data) with shape (..., V) where V
                denotes the number of visible variables in the model. All leading dimensions are
                averaged over.
            s_model (torch.Tensor): Tensor of spins drawn from the model with shape (..., N) where
                N denotes the total number of variables in the model. All leading dimensions are
                averaged over.
            kind (Literal["sampling", "exact-disc"]): Method for computing, or approximating,
                marginal expectations of hidden units given the observations. Required if, and
                only if, the model has hidden units. The "sampling" method samples the hidden
                units conditionally for each observation with ``sampler``. The "exact-disc" method
                computes exact marginals, which is possible when hidden units are disconnected,
                i.e., no connections between hidden units.
            sampler (TorchSampler, optional): The sampler used to sample the hidden units
                conditioned on the observations; its :meth:`~TorchSampler.sample` method receives
                the observations padded with ``torch.nan`` at the hidden units. Only used, and
                required, when ``kind`` is "sampling". Defaults to None.

        Returns:
            torch.Tensor: Scalar difference of the average energy of data and model whose gradients
            are equivalent to the gradients of the negative log likelihood.
        """
        if self._hidden_nodes:
            if kind == "exact-disc":
                if sampler is not None:
                    warnings.warn(f"`sampler` is not used by kind {kind!r} ({sampler})")
                if self._connected_hidden:
                    raise ValueError(
                        'The "exact-disc" method requires hidden units to be disconnected from '
                        'each other.'
                    )
                obs = self._conditional_expectation(s_observed)
            elif kind == "sampling":
                if sampler is None:
                    raise ValueError('`sampler` is required when `kind` is "sampling".')
                obs = self._conditional_samples(s_observed, sampler)
            else:
                raise ValueError(
                    f'Invalid kind ({kind}). Should be one of "sampling" or "exact-disc"'
                )
        else:
            if kind is not None:
                raise ValueError(
                    f"`kind` {kind} should not be specified if the model is fully visible."
                )
            obs = s_observed

        obs = self._flatten(obs)
        s_model = self._flatten(s_model)
        mean_diff = obs.mean(0) - s_model.mean(0)
        # Second moments are accumulated from unscaled spins and combined in one fused operation.
        moment_diff = torch.addmm(
            obs.mT @ obs, s_model.mT, s_model,
            beta=1.0 / obs.shape[0], alpha=-1.0 / s_model.shape[0],
        )
        coupling = self.coupling()
        return mean_diff @ self._linear + torch.dot(coupling.reshape(-1), moment_diff.reshape(-1))

    def pad_visible(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds observed visible spins into the full node order, marking hidden units with
        ``torch.nan``.

        Args:
            x (torch.Tensor): Observed spins of shape (..., V) where V is the number of visible
                units, in the order of :attr:`visible_idx`.

        Raises:
            ValueError: If the trailing dimension of ``x`` is not the number of visible units.

        Returns:
            torch.Tensor: A (..., N) tensor with ``x`` at :attr:`visible_idx` and ``torch.nan`` at
            :attr:`hidden_idx`.
        """
        if x.shape[-1] != self.n_visible:
            raise ValueError(
                f"Expected observations with trailing dimension {self.n_visible} (number of "
                f"visible units), got {x.shape[-1]}."
            )
        dtype = x.dtype if x.is_floating_point() else self._linear.dtype
        padded = torch.full(
            (*x.shape[:-1], self._n_nodes), torch.nan, dtype=dtype, device=x.device
        )
        padded[..., self._visible_idx] = x
        return padded

    def _conditional_expectation(self, s_observed: torch.Tensor) -> torch.Tensor:
        """Exact conditional expectations of the hidden units given the observations, for models
        whose hidden units are disconnected from each other.

        Args:
            s_observed (torch.Tensor): Observed spins of shape (..., V).

        Returns:
            torch.Tensor: A (..., N) tensor holding the observed spins at the visible units and
            the conditional expectation ``-tanh(effective field)`` at the hidden units. The
            expectations are treated as constants by autograd.
        """
        if self._connected_hidden:
            raise ValueError(
                "Exact conditional expectations require hidden units to be disconnected from "
                "each other."
            )
        padded = self.pad_visible(s_observed)
        with torch.no_grad():
            field = self.effective_field(padded, self._hidden_idx)
        padded[..., self._hidden_idx] = -torch.tanh(field)
        return padded

    def _conditional_samples(self, s_observed: torch.Tensor, sampler: TorchSampler) -> torch.Tensor:
        """Samples of the hidden units given the observations, drawn with ``sampler``.

        Args:
            s_observed (torch.Tensor): Observed spins of shape (..., V).
            sampler (TorchSampler): Sampler supporting conditional sampling of ``torch.nan``
                entries.

        Returns:
            torch.Tensor: A (..., M, N) tensor of M samples per observation. The visible entries
            are the observed spins (and remain differentiable); the hidden entries are samples.
        """
        padded = self.pad_visible(s_observed)
        with torch.no_grad():
            samples = sampler.sample(padded)
        samples = samples.reshape(*s_observed.shape[:-1], -1, self._n_nodes).clone()
        samples[..., self._visible_idx] = s_observed.unsqueeze(-2).to(samples.dtype)
        return samples

    # ------------------------------------------------------------------ dimod interop ------------

    def to_ising(
        self,
        prefactor: float = 1.0,
        linear_range: Optional[tuple[float, float]] = None,
        quadratic_range: Optional[tuple[float, float]] = None,
    ) -> tuple[dict, dict]:
        """Convert the model to Ising format.

        Convert the model to Ising format with scaling (``prefactor``) followed by clipping (if
        ``linear_range`` and/or ``quadratic_range`` are supplied).

        Args:
            prefactor (float): A scaling term applied to the linear and quadratic biases prior to,
                if applicable, clipping. Defaults to 1.
            linear_range (tuple[float, float], optional): The minimum and maximum values to clip
                linear biases with.
            quadratic_range (tuple[float, float], optional): The minimum and maximum values to
                clip quadratic biases with.

        Returns:
            tuple[dict, dict]: The linear and quadratic biases in dictionary format compatible with
            `dimod.Sampler.sample_ising`; the quadratic biases are keyed by :attr:`edges`.
        """
        linear = prefactor * self._linear.detach()
        quadratic = prefactor * self._quadratic.detach()[self._edge_idx_i, self._edge_idx_j]
        if linear_range is not None:
            linear = linear.clip(*linear_range)
        if quadratic_range is not None:
            quadratic = quadratic.clip(*quadratic_range)
        h = dict(zip(self._nodes, linear.cpu().tolist()))
        J = dict(zip(self._edges, quadratic.cpu().tolist()))
        return h, J

    def estimate_beta(self, spins: torch.Tensor) -> float:
        """Estimate the maximum pseudolikelihood temperature using
        ``dwave.system.temperatures.maximum_pseudolikelihood_temperature``.

        Args:
            spins (torch.Tensor): A tensor of shape (b, N) where b is the sample size,
                and N denotes the number of variables in the model.

        Returns:
            float: The estimated inverse temperature of the model.
        """
        bqm = BinaryQuadraticModel.from_ising(*self.to_ising())
        return float(1 / mple(bqm, (spins.detach().cpu().numpy(), self._nodes))[0])
