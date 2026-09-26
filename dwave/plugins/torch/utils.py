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

from typing import Hashable, Iterable, Sequence

import numpy as np
import torch
from dimod import BinaryQuadraticModel, SampleSet
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

__all__ = ["GraphIndex", "estimate_beta", "randspin", "sampleset_to_tensor", "to_bqm", "to_ising"]


class GraphIndex(torch.nn.Module):
    """Integer indexing of the nodes and edges of a simple graph.

    Nodes are indexed by their position in ``nodes`` and edges by their position in ``edges``.
    Quadratic biases are given *per edge*, as tensors of shape ``(..., n_edges)`` in the order of
    :attr:`edges`. Where a dense matrix product is the efficient computation, as in :meth:`energy`
    and :meth:`effective_field`, the coupling matrix is built on the fly by :meth:`dense_quadratic`
    and discarded afterwards, so the per-edge biases are the only representation that is stored,
    optimized and exchanged. The index tensors are registered as buffers, so they move with
    :meth:`~torch.nn.Module.to` and are part of :meth:`~torch.nn.Module.state_dict`. Modules
    defined on a graph, such as
    :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` and
    :class:`~dwave.plugins.torch.nn.Ising`, subclass it.

    Args:
        nodes (Iterable[Hashable]): Nodes of the graph.
        edges (Iterable[tuple[Hashable, Hashable]]): Edges of the graph.

    Raises:
        ValueError: If ``nodes`` contains duplicates, an edge references an unknown node, an
            edge is a self-loop, or an edge is duplicated (in either orientation).

    The module also evaluates the Ising :meth:`energy` and :meth:`effective_field` of spins
    under biases that are given explicitly, for a single model or a batch of models on the graph.
    Subclasses that own parameters, such as the Boltzmann machine, wrap these methods with their
    own biases.

    Attributes:
        nodes (tuple[Hashable, ...]): The nodes, in index order.
        edges (tuple[tuple[Hashable, Hashable], ...]): The edges, in the orientation given.
        node_to_idx (dict[Hashable, int]): Mapping from node to index. It is derived from
            ``nodes`` and should not be modified.
        edge_to_idx (dict[tuple[Hashable, Hashable], int]): Mapping from an edge, in either
            orientation, to its index in :attr:`edges`. It is derived from ``edges`` and should
            not be modified.
        edge_idx_i (torch.Tensor): Smaller node index of each edge, shape ``(n_edges,)``.
        edge_idx_j (torch.Tensor): Larger node index of each edge, shape ``(n_edges,)``.
    """

    def __init__(
        self, nodes: Iterable[Hashable], edges: Iterable[tuple[Hashable, Hashable]]
    ) -> None:
        super().__init__()
        self.nodes = tuple(nodes)
        self.edges = tuple(tuple(edge) for edge in edges)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        if len(self.node_to_idx) != self.n_nodes:
            raise ValueError("`nodes` contains duplicate entries.")
        try:
            endpoints = torch.tensor(
                [[self.node_to_idx[u], self.node_to_idx[v]] for u, v in self.edges],
                dtype=torch.long,
            ).reshape(-1, 2)
        except KeyError as err:
            raise ValueError(f"Edge endpoint {err.args[0]!r} is not a node.") from None

        loops = endpoints[:, 0] == endpoints[:, 1]
        if loops.any():
            raise ValueError(
                f"Self-loops are not allowed. Edges with self-loops: "
                f"{[self.edges[k] for k in loops.nonzero().flatten().tolist()]}"
            )
        self.edge_to_idx = {
            edge: k for k, (u, v) in enumerate(self.edges) for edge in ((u, v), (v, u))
        }
        if len(self.edge_to_idx) != 2 * self.n_edges:
            raise ValueError("Duplicate edges are not allowed.")

        self.register_buffer("edge_idx_i", endpoints.min(1).values)
        self.register_buffer("edge_idx_j", endpoints.max(1).values)

    def extra_repr(self) -> str:
        return f"n_nodes={self.n_nodes}, n_edges={self.n_edges}"

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def degrees(self) -> torch.Tensor:
        """Degree of every node, shape ``(n_nodes,)``."""
        return torch.bincount(
            torch.cat([self.edge_idx_i, self.edge_idx_j]), minlength=self.n_nodes
        )

    # ------------------------------------------------------------------ coupling matrices --------

    def dense_quadratic(self, quadratic: torch.Tensor) -> torch.Tensor:
        """Build the dense coupling matrix :math:`J` of per-edge quadratic biases.

        This is the single place where a dense matrix is made from the per-edge biases; callers
        build it right before a matrix product and let it go afterwards.

        Args:
            quadratic (torch.Tensor): Quadratic biases of the edges, of shape ``(..., n_edges)``
                and in the order of :attr:`edges`.

        Raises:
            ValueError: If the trailing dimension of ``quadratic`` is not the number of edges.

        Returns:
            torch.Tensor: A strictly upper-triangular tensor of shape ``(..., n_nodes, n_nodes)``
            holding the bias of edge ``k`` at ``[edge_idx_i[k], edge_idx_j[k]]`` and zeros
            elsewhere.
        """
        if quadratic.shape[-1] != self.n_edges:
            raise ValueError(
                f"Expected {self.n_edges} edge biases, got {quadratic.shape[-1]}."
            )
        coupling = quadratic.new_zeros(*quadratic.shape[:-1], self.n_nodes, self.n_nodes)
        coupling[..., self.edge_idx_i, self.edge_idx_j] = quadratic
        return coupling

    def symmetric_coupling(self, quadratic: torch.Tensor) -> torch.Tensor:
        """The symmetrized coupling matrix :math:`J + J^T` of per-edge quadratic biases, whose
        row ``k`` holds the couplings of node ``k`` to every other node.

        Args:
            quadratic (torch.Tensor): Quadratic biases of the edges, of shape ``(..., n_edges)``.

        Returns:
            torch.Tensor: A symmetric tensor of shape ``(..., n_nodes, n_nodes)`` with zero
            diagonal.
        """
        coupling = self.dense_quadratic(quadratic)
        return coupling + coupling.mT

    # ------------------------------------------------------------------ Ising energies -----------

    def energy(
        self, x: torch.Tensor, linear: torch.Tensor, quadratic: torch.Tensor
    ) -> torch.Tensor:
        r"""Energies :math:`\sum_i h_i s_i + \sum_{(i, j)} J_{ij} s_i s_j` of spins under given
        biases.

        The biases define one Ising model or a batch of them. Unbatched biases, of shapes
        ``(n_nodes,)`` and ``(n_edges,)``, apply to spins of any shape ``(..., n_nodes)``. Batched
        biases, of shapes ``(*batch, n_nodes)`` and ``(*batch, n_edges)``, define one model per
        batch element and apply to spins of shape ``(*batch, M, n_nodes)``, i.e. ``M``
        configurations per model, or ``(*batch, n_nodes)``, one configuration per model. The
        coupling matrices are built on the fly with :meth:`dense_quadratic`.

        Args:
            x (torch.Tensor): Spins.
            linear (torch.Tensor): Linear biases.
            quadratic (torch.Tensor): Quadratic biases of the edges, in the order of
                :attr:`edges`.

        Returns:
            torch.Tensor: Energies of shape ``x.shape[:-1]``.
        """
        x, squeeze = self._align_spins(x, linear)
        coupling = self.dense_quadratic(quadratic)
        energy = (x @ linear.unsqueeze(-1)).squeeze(-1) + ((x @ coupling) * x).sum(-1)
        return energy.squeeze(-1) if squeeze else energy

    def effective_field(
        self,
        x: torch.Tensor,
        *,
        linear: torch.Tensor,
        quadratic: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
        coupling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Effective fields :math:`h_k + \sum_{l} J_{kl} s_l` acting on nodes under given biases.

        The effective field of node :math:`k` is the derivative of the energy with respect to
        :math:`s_k`; the conditional distribution of :math:`s_k` given all other spins is
        :math:`P(s_k = \pm 1) = \sigma(\mp 2 h^{\text{eff}}_k)`. Entries of ``x`` that are
        ``torch.nan`` denote unknown spins and contribute nothing to the fields. Biases are batched
        as in :meth:`energy`.

        Args:
            x (torch.Tensor): Spins of shape ``(..., n_nodes)``; ``torch.nan`` marks unknown spins.
            linear (torch.Tensor): Linear biases.
            quadratic (torch.Tensor, optional): Quadratic biases of the edges, in the order of
                :attr:`edges`. Required unless ``coupling`` is given.
            idx (torch.Tensor, optional): Indices of the nodes whose fields are returned. If
                ``None``, the fields of all nodes are returned. Defaults to ``None``.
            coupling (torch.Tensor, optional): The :meth:`symmetric_coupling` of ``quadratic``,
                which a caller evaluating the fields of several blocks of nodes can pass to avoid
                rebuilding it. Defaults to ``None``, i.e. it is built from ``quadratic``.

        Raises:
            ValueError: If neither ``quadratic`` nor ``coupling`` is given.

        Returns:
            torch.Tensor: Effective fields of shape ``(..., n_nodes)`` or ``(..., len(idx))``.
        """
        if coupling is None:
            if quadratic is None:
                raise ValueError("Either `quadratic` or `coupling` is required.")
            coupling = self.symmetric_coupling(quadratic)
        x, squeeze = self._align_spins(x, linear)
        spins = torch.nan_to_num(x, nan=0.0)
        fields = linear if linear.ndim == 1 else linear.unsqueeze(-2)
        if idx is not None:
            fields, coupling = fields[..., idx], coupling[..., :, idx]
        fields = fields + spins @ coupling
        return fields.squeeze(-2) if squeeze else fields

    @staticmethod
    def _align_spins(x: torch.Tensor, linear: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Insert a sample dimension into ``x`` when it holds one configuration per batched model.

        Returns:
            tuple[torch.Tensor, bool]: The spins with a sample dimension, and whether one was
            inserted (and should be squeezed out of the results again).
        """
        if linear.ndim > 1 and x.ndim == linear.ndim:
            return x.unsqueeze(-2), True
        return x, False


def randspin(size: Sequence[int], **kwargs) -> torch.Tensor:
    """Random spins, i.e. ``±1`` values drawn uniformly and independently.

    Args:
        size (Sequence[int]): Shape of the output tensor.
        **kwargs: Keyword arguments of :func:`torch.randint`, such as ``generator``, ``device``
            and ``dtype``.

    Returns:
        torch.Tensor: A tensor of ``±1`` values of the given shape, of ``torch.int64`` unless
        ``dtype`` is given.
    """
    return 2 * torch.randint(0, 2, size, **kwargs) - 1


def sampleset_to_tensor(
    ordered_vars: Sequence[Hashable], sample_set: SampleSet, device: torch.device | None = None
) -> torch.Tensor:
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor`` with one row per read.

    Aggregated samples, i.e. those with ``num_occurrences > 1``, are repeated accordingly, so
    that every row of the result has equal weight and statistics of the rows are statistics of
    the reads.

    Args:
        ordered_vars (Sequence[Hashable]): The desired order of the columns.
        sample_set (dimod.SampleSet): A sample set.
        device (torch.device, optional): The device of the constructed tensor. If ``None``, the
            tensor is constructed on the current device.

    Returns:
        torch.Tensor: The samples as a ``(num_reads, len(ordered_vars))`` tensor of
        ``torch.float32``.
    """
    var_to_sample_i = {v: i for i, v in enumerate(sample_set.variables)}
    permutation = [var_to_sample_i[v] for v in ordered_vars]
    record = sample_set.record
    sample = np.repeat(record.sample[:, permutation], record.num_occurrences, axis=0)
    return torch.tensor(sample, dtype=torch.float32, device=device)


def _scale_and_clip(
    biases: torch.Tensor, prefactor: float, bounds: tuple[float, float] | None
) -> torch.Tensor:
    """Scales ``biases`` by ``prefactor`` and then clips them to ``bounds``, if given."""
    biases = prefactor * biases
    return biases if bounds is None else biases.clip(*bounds)


def to_ising(
    nodes: Sequence[Hashable],
    edges: Sequence[tuple[Hashable, Hashable]],
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    prefactor: float = 1.0,
    linear_range: tuple[float, float] | None = None,
    quadratic_range: tuple[float, float] | None = None,
) -> tuple[dict[Hashable, float], dict[tuple[Hashable, Hashable], float]]:
    """Converts linear and quadratic biases to the Ising dictionaries used by dimod.

    The biases are scaled by ``prefactor`` and then clipped to ``linear_range`` and
    ``quadratic_range`` (if given), which is how a Hamiltonian is prepared for a sampler that
    operates at a fixed temperature and with bounded biases, e.g. a quantum annealer.

    Args:
        nodes (Sequence[Hashable]): Node labels, in the order of ``linear``.
        edges (Sequence[tuple[Hashable, Hashable]]): Edge labels, in the order of ``quadratic``.
        linear (torch.Tensor): Linear biases of shape ``(len(nodes),)``.
        quadratic (torch.Tensor): Quadratic biases of the edges, shape ``(len(edges),)``.
        prefactor (float): Scaling applied to all biases prior to clipping. Defaults to 1.
        linear_range (tuple[float, float], optional): Minimum and maximum of the linear biases.
        quadratic_range (tuple[float, float], optional): Minimum and maximum of the quadratic
            biases.

    Raises:
        ValueError: If the number of biases does not match the number of nodes or edges.

    Returns:
        tuple[dict, dict]: Linear biases keyed by node and quadratic biases keyed by edge, as
        accepted by :meth:`dimod.Sampler.sample_ising` and
        :meth:`dimod.BinaryQuadraticModel.from_ising`.
    """
    nodes = list(nodes)
    edges = [tuple(edge) for edge in edges]
    linear = torch.as_tensor(linear).detach()
    quadratic = torch.as_tensor(quadratic).detach()
    if tuple(linear.shape) != (len(nodes),):
        raise ValueError(
            f"Expected {len(nodes)} linear biases (one per node), got shape {tuple(linear.shape)}."
        )
    if tuple(quadratic.shape) != (len(edges),):
        raise ValueError(
            f"Expected {len(edges)} quadratic biases (one per edge), got shape "
            f"{tuple(quadratic.shape)}."
        )
    linear = _scale_and_clip(linear, prefactor, linear_range)
    quadratic = _scale_and_clip(quadratic, prefactor, quadratic_range)
    h = dict(zip(nodes, linear.cpu().tolist()))
    J = dict(zip(edges, quadratic.cpu().tolist()))
    return h, J


def to_bqm(
    nodes: Sequence[Hashable],
    edges: Sequence[tuple[Hashable, Hashable]],
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    prefactor: float = 1.0,
    linear_range: tuple[float, float] | None = None,
    quadratic_range: tuple[float, float] | None = None,
) -> BinaryQuadraticModel:
    """Converts linear and quadratic biases to a ``dimod.BinaryQuadraticModel``.

    The arguments are those of :func:`to_ising`, whose dictionaries are passed on to
    :meth:`dimod.BinaryQuadraticModel.from_ising`.

    Returns:
        dimod.BinaryQuadraticModel: The (scaled and clipped) model in the ``SPIN`` vartype.
    """
    return BinaryQuadraticModel.from_ising(
        *to_ising(nodes, edges, linear, quadratic, prefactor, linear_range, quadratic_range)
    )


def estimate_beta(
    nodes: Sequence[Hashable],
    edges: Sequence[tuple[Hashable, Hashable]],
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    spins: torch.Tensor,
) -> float:
    """Maximum pseudolikelihood estimate of the inverse temperature at which spins were sampled
    from an Ising model.

    Uses ``dwave.system.temperatures.maximum_pseudolikelihood_temperature`` on the binary
    quadratic model of the given biases; see :func:`to_bqm` for the arguments describing the
    model. See `Global Warming: Temperature Estimation in Annealers
    <https://doi.org/10.3389/fict.2016.00023>`_ for more on estimating beta.

    Args:
        nodes (Sequence[Hashable]): Node labels, in the order of ``linear``.
        edges (Sequence[tuple[Hashable, Hashable]]): Edge labels, in the order of ``quadratic``.
        linear (torch.Tensor): Linear biases of shape ``(len(nodes),)``.
        quadratic (torch.Tensor): Quadratic biases of the edges, shape ``(len(edges),)``.
        spins (torch.Tensor): Spins of shape ``(M, len(nodes))`` with one column per node, in the
            order of ``nodes``.

    Returns:
        float: The estimated inverse temperature.
    """
    bqm = to_bqm(nodes, edges, linear, quadratic)
    samples = torch.as_tensor(spins).detach().cpu().numpy()
    return float(1 / mple(bqm, (samples, list(nodes)))[0])
