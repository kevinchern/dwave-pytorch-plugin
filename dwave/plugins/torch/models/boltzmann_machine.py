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

from typing import Hashable, Iterable

import torch

from dwave.plugins.torch.utils import GraphIndex, estimate_beta

__all__ = ["GraphRestrictedBoltzmannMachine"]


class GraphRestrictedBoltzmannMachine(GraphIndex):
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
    hidden units have one column per *visible* node, in the order of :attr:`visible_idx`;
    :meth:`pad_visible` embeds them into the full node order with ``torch.nan`` marking the
    hidden units. The learning objective :meth:`quasi_objective` takes complete spin
    configurations, so the hidden units of the data are filled in first: exactly, with
    :meth:`conditional_expectation`, when no two hidden units are adjacent, or with conditional
    samples drawn by the ``complete`` method of a
    :class:`~dwave.plugins.torch.samplers.TorchSampler` otherwise.

    The graph attributes and buffers---``nodes``, ``edges``, ``node_to_idx``, ``edge_idx_i``,
    ``edge_idx_j`` and ``adjacency``---are those of :class:`~dwave.plugins.torch.utils.GraphIndex`.
    The parameters and buffers of the module are registered under the attribute names listed
    below and in the base class, which are therefore the keys of
    :meth:`~torch.nn.Module.state_dict`.

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

    Attributes:
        linear (torch.nn.Parameter): The linear biases, of shape ``(n_nodes,)``.
        quadratic (torch.nn.Parameter): The quadratic biases as a dense ``(n_nodes, n_nodes)``
            tensor. The bias of the edge between the nodes with indices ``i < j`` is stored at
            ``[i, j]``; entries outside :attr:`adjacency` are ignored by every computation. The
            per-edge biases, in the order of :attr:`edges`, are returned by :meth:`edge_biases`.
        visible_idx (torch.Tensor): Indices of the visible units, in the order of the columns
            of observations.
        hidden_idx (torch.Tensor): Indices of the hidden units.
        hidden_nodes (tuple[Hashable, ...]): The hidden nodes.
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
        hidden_nodes: Iterable[Hashable] | None = None,
        linear: dict[Hashable, float] | None = None,
        quadratic: dict[tuple[Hashable, Hashable], float] | None = None,
    ) -> None:
        super().__init__(nodes, edges)

        quadratic_init = torch.zeros(self.n_nodes, self.n_nodes)
        if self.n_edges:
            degrees = self.degrees().to(quadratic_init.dtype)
            quadratic_std = self._INIT_INVERSE_TEMP / (
                degrees[self.edge_idx_i] * degrees[self.edge_idx_j]
            )**0.25
            quadratic_init[self.edge_idx_i, self.edge_idx_j] = (
                torch.randn(self.n_edges) * quadratic_std
            )
        self.linear = torch.nn.Parameter(torch.zeros(self.n_nodes))
        self.quadratic = torch.nn.Parameter(quadratic_init)

        self.hidden_nodes = () if hidden_nodes is None else tuple(hidden_nodes)
        hidden_set = set(self.hidden_nodes)
        if len(hidden_set) != len(self.hidden_nodes):
            raise ValueError("`hidden_nodes` contains duplicate entries.")
        unknown = hidden_set.difference(self.nodes)
        if unknown:
            raise ValueError(f"Hidden nodes {list(unknown)!r} are not nodes of the model.")
        is_hidden = torch.tensor([v in hidden_set for v in self.nodes], dtype=torch.bool)
        self.register_buffer("visible_idx", torch.nonzero(~is_hidden).flatten())
        self.register_buffer("hidden_idx", torch.nonzero(is_hidden).flatten())

        if linear is not None:
            self.set_linear(linear)
        if quadratic is not None:
            self.set_quadratic(quadratic)

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, n_hidden={self.n_hidden}"

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
            node_idx = [self.node_to_idx[node] for node in linear]
        except KeyError as err:
            raise ValueError(f"Node {err.args[0]!r} is not in the model.") from None
        device = self.linear.device
        values = torch.tensor(list(linear.values()), dtype=self.linear.dtype, device=device)
        with torch.no_grad():
            self.linear[torch.tensor(node_idx, device=device)] = values

    def set_quadratic(self, quadratic: dict[tuple[Hashable, Hashable], float]) -> None:
        """Set quadratic biases of the model.

        Args:
            quadratic (dict[tuple[Hashable, Hashable], float]): A dictionary mapping from edges of
                the model, in either orientation, to its corresponding quadratic bias. Not all
                quadratic biases need to be set; edges without a mapping keep their current values.

        Raises:
            ValueError: If a key is not an edge of the model, e.g. it references an unknown
                node, is a self-loop or joins two nodes that are not adjacent.
        """
        if not quadratic:
            return
        # The bias of edge (u, v) lives at the canonical position [min(i, j), max(i, j)]
        rows, cols = [], []
        for u, v in quadratic:
            try:
                i, j = sorted((self.node_to_idx[u], self.node_to_idx[v]))
            except KeyError:
                raise ValueError(f"Edge {(u, v)!r} is not in the model.") from None
            rows.append(i)
            cols.append(j)
        device = self.quadratic.device
        rows = torch.tensor(rows, device=device)
        cols = torch.tensor(cols, device=device)
        is_edge = self.adjacency[rows, cols]
        if not is_edge.all():
            offending = next(edge for edge, ok in zip(quadratic, is_edge.tolist()) if not ok)
            raise ValueError(f"Edge {offending!r} is not in the model.")
        values = torch.tensor(
            list(quadratic.values()), dtype=self.quadratic.dtype, device=device
        )
        with torch.no_grad():
            self.quadratic[rows, cols] = values

    def coupling(self, quadratic: torch.Tensor | None = None) -> torch.Tensor:
        """The coupling matrix :math:`J` with off-graph entries forced to zero.

        Args:
            quadratic (torch.Tensor, optional): Dense quadratic biases of shape
                ``(..., n_nodes, n_nodes)``. Defaults to the model's :attr:`quadratic`.

        Returns:
            torch.Tensor: A strictly upper-triangular tensor of the shape of ``quadratic``.
        """
        return super().coupling(self.quadratic if quadratic is None else quadratic)

    def symmetric_coupling(self, quadratic: torch.Tensor | None = None) -> torch.Tensor:
        """The symmetrized coupling matrix :math:`J + J^T`, whose row ``k`` holds the couplings
        of node ``k`` to every other node.

        Args:
            quadratic (torch.Tensor, optional): Dense quadratic biases of shape
                ``(..., n_nodes, n_nodes)``. Defaults to the model's :attr:`quadratic`.

        Returns:
            torch.Tensor: A symmetric tensor of the shape of ``quadratic`` with zero diagonal.
        """
        return super().symmetric_coupling(self.quadratic if quadratic is None else quadratic)

    def edge_biases(self, quadratic: torch.Tensor | None = None) -> torch.Tensor:
        """Quadratic biases of the edges, of shape ``(..., n_edges)`` and in the order of
        :attr:`edges`.

        Args:
            quadratic (torch.Tensor, optional): Dense quadratic biases of shape
                ``(..., n_nodes, n_nodes)`` in canonical orientation. Defaults to the model's
                :attr:`quadratic`.

        Returns:
            torch.Tensor: The entries of ``quadratic`` at ``[edge_idx_i, edge_idx_j]``.
        """
        return super().edge_biases(self.quadratic if quadratic is None else quadratic)

    # ------------------------------------------------------------------ graph --------------------

    @property
    def visible_nodes(self) -> tuple[Hashable, ...]:
        """The visible nodes, in the order of :attr:`visible_idx`."""
        return tuple(self.nodes[idx] for idx in self.visible_idx.tolist())

    @property
    def n_visible(self) -> int:
        """Number of visible units."""
        return self.visible_idx.numel()

    @property
    def n_hidden(self) -> int:
        """Number of hidden units."""
        return self.hidden_idx.numel()

    @property
    def connected_hidden(self) -> bool:
        """Whether any edge connects two hidden units."""
        is_hidden = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.hidden_idx.device)
        is_hidden[self.hidden_idx] = True
        return bool((is_hidden[self.edge_idx_i] & is_hidden[self.edge_idx_j]).any())

    # ------------------------------------------------------------------ energies -----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the Hamiltonian.

        Args:
            x (torch.Tensor): A tensor of shape (..., N) where N denotes the number of variables in
                the model.

        Returns:
            torch.Tensor: Hamiltonians of shape (...,).
        """
        return self.energy(x, self.linear, self.quadratic)

    def effective_field(
        self,
        x: torch.Tensor,
        idx: torch.Tensor | None = None,
        coupling: torch.Tensor | None = None,
        *,
        linear: torch.Tensor | None = None,
        quadratic: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
            coupling (torch.Tensor, optional): The :meth:`symmetric_coupling` matrix, which a
                caller evaluating the fields of several blocks of nodes can pass to avoid
                recomputing it. Defaults to ``None``, i.e. it is computed.
            linear (torch.Tensor, optional): Linear biases to use instead of the model's
                :attr:`linear`, possibly a batch of them (see
                :meth:`~dwave.plugins.torch.utils.GraphIndex.effective_field`).
            quadratic (torch.Tensor, optional): Dense quadratic biases to use instead of the
                model's :attr:`quadratic`.

        Returns:
            torch.Tensor: Effective fields of shape (..., N) or (..., ``len(idx)``).
        """
        return super().effective_field(
            x,
            linear=self.linear if linear is None else linear,
            quadratic=self.quadratic if quadratic is None else quadratic,
            idx=idx,
            coupling=coupling,
        )

    def sufficient_statistics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-averaged sufficient statistics of spins, in the layout of the parameters.

        The sufficient statistics of the model are the spins and the products of spins along the
        edges. Their batch averages are the derivatives of the average energy with respect to
        :attr:`linear` and :attr:`quadratic`: the average energy of the batch is
        ``mean @ linear + (quadratic * second).sum()``, and the gradient of the negative log
        likelihood is the difference between the statistics of the model and those of the data.

        Args:
            x (torch.Tensor): Spins of shape (..., N) where N denotes the number of variables in
                the model. All leading dimensions are averaged over. Entries may be expectations of
                spins rather than spins (see :meth:`conditional_expectation`).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensors of shape (N,) and (N, N) holding the
            average spins and the average products of spins along the edges. The latter is in the
            canonical orientation of :attr:`quadratic`; entries that are not edges are zero.
        """
        x = self._flatten(x)
        return x.mean(0), (x.mT @ x) * self.adjacency / x.shape[0]

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten all leading dimensions of a (..., N) tensor into one batch dimension."""
        if x.shape[-1] != self.n_nodes:
            raise ValueError(
                f"Expected spins with trailing dimension {self.n_nodes}, got {x.shape[-1]}."
            )
        return x.reshape(-1, self.n_nodes)

    # ------------------------------------------------------------------ learning -----------------

    def quasi_objective(self, s_data: torch.Tensor, s_model: torch.Tensor) -> torch.Tensor:
        """A quasi-objective function whose gradients are the gradients of the negative log
        likelihood.

        The objective is the difference between the average energy of the data and the average
        energy of spins drawn from the model, ``self(s_data).mean() - self(s_model).mean()``.
        Its gradient with respect to the linear and quadratic biases is the difference of the
        :meth:`sufficient_statistics` of data and model, i.e. the gradient of the negative log
        likelihood. The objective is differentiable with respect to ``s_data`` as well, which
        lets gradients flow into an encoder that produces the data (see
        :func:`~dwave.plugins.torch.nn.functional.pseudo_kl_divergence_loss`).

        Both arguments are complete spin configurations with one column per node. For a model
        with hidden units, fill in the hidden units of the data first: exactly with
        :meth:`conditional_expectation` when no two hidden units are adjacent, otherwise with
        samples drawn conditioned on the data by the ``complete`` method of a
        :class:`~dwave.plugins.torch.samplers.TorchSampler`:

        .. code-block:: python

            s_data = model.conditional_expectation(model.pad_visible(x))  # exact
            s_data = sampler.complete(x)  # sampled
            model.quasi_objective(s_data, sampler.sample()).backward()

        Args:
            s_data (torch.Tensor): Data spins of shape (..., N) where N denotes the number of
                variables in the model. All leading dimensions are averaged over. Entries may be
                expectations of spins.
            s_model (torch.Tensor): Spins drawn from the model, of shape (..., N). All leading
                dimensions are averaged over.

        Returns:
            torch.Tensor: Scalar difference between the average energies of data and model.
        """
        return self(self._flatten(s_data)).mean() - self(self._flatten(s_model)).mean()

    # ------------------------------------------------------------------ hidden units -------------

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
        dtype = x.dtype if x.is_floating_point() else self.linear.dtype
        padded = torch.full(
            (*x.shape[:-1], self.n_nodes), torch.nan, dtype=dtype, device=x.device
        )
        padded[..., self.visible_idx] = x
        return padded

    def conditional_expectation(self, x: torch.Tensor) -> torch.Tensor:
        r"""Exact conditional expectations of unknown spins given the observed spins.

        Entries of ``x`` equal to ``torch.nan`` are unknown. Provided that no two unknown spins
        are adjacent, an unknown spin :math:`s_k` interacts with observed spins only, so its
        conditional distribution is determined by its effective field :math:`h^{\text{eff}}_k`
        (see :meth:`effective_field`) and its conditional expectation is
        :math:`-\tanh(h^{\text{eff}}_k)`. For a model with hidden units,
        ``model.conditional_expectation(model.pad_visible(observations))`` yields the
        expectations of the hidden units given the data, which is exact when the hidden units are
        disconnected from each other (see :attr:`connected_hidden`).

        The expectations are constants to autograd, i.e. they are not differentiated with respect
        to the parameters. This is what :meth:`quasi_objective` requires: the positive-phase
        statistics are expectations of the sufficient statistics under the conditional
        distribution, and the gradient of the negative log likelihood is obtained by holding these
        expectations fixed. The observed entries of ``x`` are returned as they are and remain
        differentiable.

        Args:
            x (torch.Tensor): Spins of shape (..., N) where N denotes the number of variables in
                the model; ``torch.nan`` marks unknown spins.

        Raises:
            ValueError: If two unknown spins are adjacent in some row of ``x``.

        Returns:
            torch.Tensor: A tensor of shape (..., N) holding the observed spins and the
            conditional expectations of the unknown spins.
        """
        unknown = torch.isnan(x)
        with torch.no_grad():
            neighbours = (self.adjacency | self.adjacency.mT).to(x.dtype)
            unknown_spins = unknown.to(x.dtype)
            if ((unknown_spins @ neighbours) * unknown_spins).any():
                raise ValueError(
                    "Exact conditional expectations require that no two unknown spins are "
                    "adjacent; with hidden units, the hidden units must be disconnected from "
                    "each other."
                )
            field = self.effective_field(x)
        return torch.where(unknown, -torch.tanh(field), x)

    # ------------------------------------------------------------------ temperature --------------

    def estimate_beta(self, spins: torch.Tensor) -> float:
        """Estimate the maximum pseudolikelihood temperature using
        ``dwave.system.temperatures.maximum_pseudolikelihood_temperature``.

        Args:
            spins (torch.Tensor): A tensor of shape (b, N) where b is the sample size,
                and N denotes the number of variables in the model.

        Returns:
            float: The estimated inverse temperature of the model.
        """
        return estimate_beta(self.nodes, self.edges, self.linear, self.edge_biases(), spins)
