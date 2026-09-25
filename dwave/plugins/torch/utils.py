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

from typing import Hashable, Iterable, Optional, Sequence

import numpy as np
import torch
from dimod import BinaryQuadraticModel, SampleSet

__all__ = ["GraphIndex", "sampleset_to_tensor", "to_bqm", "to_ising"]


class GraphIndex(torch.nn.Module):
    """Integer indexing of the nodes and edges of a simple graph.

    Nodes are indexed by their position in ``nodes``. Every edge is stored in *canonical
    orientation*, i.e. with the smaller node index first, so that a dense ``(n_nodes, n_nodes)``
    matrix indexed by ``edge_idx_i, edge_idx_j`` is strictly upper triangular. The index tensors
    are registered as buffers, so they move with :meth:`~torch.nn.Module.to` and are part of
    :meth:`~torch.nn.Module.state_dict`. Modules defined on a graph, such as
    :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` and
    :class:`~dwave.plugins.torch.nn.Ising`, subclass it.

    Args:
        nodes (Iterable[Hashable]): Nodes of the graph.
        edges (Iterable[tuple[Hashable, Hashable]]): Edges of the graph.

    Raises:
        ValueError: If ``nodes`` contains duplicates, an edge references an unknown node, an
            edge is a self-loop, or an edge is duplicated (in either orientation).

    Attributes:
        nodes (tuple[Hashable, ...]): The nodes, in index order.
        edges (tuple[tuple[Hashable, Hashable], ...]): The edges, in the orientation given.
        node_to_idx (dict[Hashable, int]): Mapping from node to index. It is derived from
            ``nodes`` and should not be modified.
        edge_idx_i (torch.Tensor): Smaller node index of each edge, shape ``(n_edges,)``.
        edge_idx_j (torch.Tensor): Larger node index of each edge, shape ``(n_edges,)``.
        adjacency (torch.Tensor): Strictly upper-triangular boolean tensor of shape
            ``(n_nodes, n_nodes)`` that is ``True`` at ``[edge_idx_i[k], edge_idx_j[k]]`` for
            every edge ``k``.
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
        edge_idx_i = endpoints.min(1).values
        edge_idx_j = endpoints.max(1).values
        if torch.unique(edge_idx_i * self.n_nodes + edge_idx_j).numel() != self.n_edges:
            raise ValueError("Duplicate edges are not allowed.")
        adjacency = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.bool)
        adjacency[edge_idx_i, edge_idx_j] = True

        self.register_buffer("edge_idx_i", edge_idx_i)
        self.register_buffer("edge_idx_j", edge_idx_j)
        self.register_buffer("adjacency", adjacency)

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

    def edge_biases(self, quadratic: torch.Tensor) -> torch.Tensor:
        """Extract the per-edge biases from dense quadratic biases.

        Args:
            quadratic (torch.Tensor): Dense quadratic biases of shape ``(..., n_nodes, n_nodes)``
                in canonical orientation.

        Returns:
            torch.Tensor: Per-edge biases of shape ``(..., n_edges)`` in the order of
            :attr:`edges`.
        """
        return quadratic[..., self.edge_idx_i, self.edge_idx_j]

    def dense_quadratic(self, edge_biases: torch.Tensor) -> torch.Tensor:
        """Build dense quadratic biases from per-edge biases.

        Args:
            edge_biases (torch.Tensor): Per-edge biases of shape ``(..., n_edges)`` in the order
                of :attr:`edges`.

        Raises:
            ValueError: If the trailing dimension of ``edge_biases`` is not the number of edges.

        Returns:
            torch.Tensor: Dense quadratic biases of shape ``(..., n_nodes, n_nodes)`` with
            ``edge_biases`` at the canonical edge positions and zeros elsewhere.
        """
        if edge_biases.shape[-1] != self.n_edges:
            raise ValueError(
                f"Expected {self.n_edges} edge biases, got {edge_biases.shape[-1]}."
            )
        quadratic = edge_biases.new_zeros(*edge_biases.shape[:-1], self.n_nodes, self.n_nodes)
        quadratic[..., self.edge_idx_i, self.edge_idx_j] = edge_biases
        return quadratic


def sampleset_to_tensor(
    ordered_vars: Sequence[Hashable], sample_set: SampleSet, device: Optional[torch.device] = None
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
    biases: torch.Tensor, prefactor: float, bounds: Optional[tuple[float, float]]
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
    linear_range: Optional[tuple[float, float]] = None,
    quadratic_range: Optional[tuple[float, float]] = None,
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
    linear_range: Optional[tuple[float, float]] = None,
    quadratic_range: Optional[tuple[float, float]] = None,
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
