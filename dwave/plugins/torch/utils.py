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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Iterable, Optional, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from dimod import SampleSet

__all__ = ["GraphIndex", "sampleset_to_tensor", "spread", "to_ising"]


@dataclass(frozen=True)
class GraphIndex:
    """Integer indexing of the nodes and edges of a simple graph.

    Nodes are indexed by their position in ``nodes``. Every edge is stored in *canonical
    orientation*, i.e. with the smaller node index first, so that a dense ``(n_nodes, n_nodes)``
    matrix indexed by ``edge_idx_i, edge_idx_j`` is strictly upper triangular.

    Args:
        nodes (list[Hashable]): The nodes.
        edges (list[tuple[Hashable, Hashable]]): The edges, in the orientation given by the user.
        node_to_idx (dict[Hashable, int]): Map from node to index.
        edge_idx_i (torch.Tensor): Smaller node index of each edge, shape ``(n_edges,)``.
        edge_idx_j (torch.Tensor): Larger node index of each edge, shape ``(n_edges,)``.
    """

    nodes: list
    edges: list
    node_to_idx: dict
    edge_idx_i: torch.Tensor
    edge_idx_j: torch.Tensor

    @classmethod
    def from_graph(
        cls, nodes: Iterable[Hashable], edges: Iterable[tuple[Hashable, Hashable]]
    ) -> GraphIndex:
        """Index a graph given as node and edge lists.

        Args:
            nodes (Iterable[Hashable]): Nodes of the graph.
            edges (Iterable[tuple[Hashable, Hashable]]): Edges of the graph.

        Raises:
            ValueError: If ``nodes`` contains duplicates, an edge references an unknown node, an
                edge is a self-loop, or an edge is duplicated (in either orientation).

        Returns:
            GraphIndex: The indexed graph.
        """
        nodes = list(nodes)
        edges = [tuple(edge) for edge in edges]
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        if len(node_to_idx) != len(nodes):
            raise ValueError("`nodes` contains duplicate entries.")
        try:
            endpoints = torch.tensor(
                [[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long
            ).reshape(-1, 2)
        except KeyError as err:
            raise ValueError(f"Edge endpoint {err.args[0]!r} is not a node.") from None

        loops = endpoints[:, 0] == endpoints[:, 1]
        if loops.any():
            raise ValueError(
                f"Self-loops are not allowed. Edges with self-loops: "
                f"{[edges[k] for k in loops.nonzero().flatten().tolist()]}"
            )
        edge_idx_i = endpoints.min(1).values
        edge_idx_j = endpoints.max(1).values
        if torch.unique(edge_idx_i * len(nodes) + edge_idx_j).numel() != len(edges):
            raise ValueError("Duplicate edges are not allowed.")
        return cls(nodes, edges, node_to_idx, edge_idx_i, edge_idx_j)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def adjacency(self) -> torch.Tensor:
        """Strictly upper-triangular boolean adjacency matrix of shape ``(n_nodes, n_nodes)``,
        ``True`` at ``[edge_idx_i[k], edge_idx_j[k]]`` for every edge ``k``."""
        adjacency = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.bool)
        adjacency[self.edge_idx_i, self.edge_idx_j] = True
        return adjacency

    def degrees(self) -> torch.Tensor:
        """Degree of every node, shape ``(n_nodes,)``."""
        return torch.bincount(
            torch.cat([self.edge_idx_i, self.edge_idx_j]), minlength=self.n_nodes
        )


def sampleset_to_tensor(
    ordered_vars: list, sample_set: SampleSet, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor``.

    Args:
        ordered_vars: list[Literal]: The desired order of sample set variables.
        sample_set (dimod.SampleSet): A sample set.
        device (torch.device, optional): The device of the constructed tensor.
            If ``None`` and data is a tensor then the device of data is used.
            If ``None`` and data is not a tensor then the result tensor is constructed
            on the current device.

    Returns:
        torch.Tensor: The sample set as a ``torch.Tensor``.
    """
    var_to_sample_i = {v: i for i, v in enumerate(sample_set.variables)}
    permutation = [var_to_sample_i[v] for v in ordered_vars]
    sample = sample_set.record.sample[:, permutation]
    return torch.tensor(sample, dtype=torch.float32, device=device)


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
    linear = prefactor * linear
    quadratic = prefactor * quadratic
    if linear_range is not None:
        linear = linear.clip(*linear_range)
    if quadratic_range is not None:
        quadratic = quadratic.clip(*quadratic_range)
    h = dict(zip(nodes, linear.cpu().tolist()))
    J = dict(zip(edges, quadratic.cpu().tolist()))
    return h, J


def spread(sample_set: SampleSet) -> SampleSet:
    """Expands aggregated samples so that every sample occurs exactly once.

    Samples with ``num_occurrences > 1`` are repeated accordingly; all other record fields are
    copied along. Sample sets whose occurrences are all one are returned unchanged.

    Args:
        sample_set (dimod.SampleSet): A (possibly aggregated) sample set.

    Returns:
        dimod.SampleSet: A sample set with ``num_occurrences`` equal to one for every sample.
    """
    from dimod import SampleSet

    record = sample_set.record
    if len(record) == 0 or (record.num_occurrences == 1).all():
        return sample_set
    expanded = record[np.repeat(np.arange(len(record)), record.num_occurrences)].copy()
    expanded.num_occurrences = 1
    return SampleSet(expanded, sample_set.variables, sample_set.info, sample_set.vartype)
