# Copyright 2026 D-Wave
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

import abc
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dwave.plugins.torch.utils import GraphIndex

__all__ = ["SpinStatistic", "IdentityStatistic", "IsingStatistic"]


def _index_tensor(indices: Iterable[int]) -> torch.Tensor:
    """Indices as a ``torch.long`` tensor, from a tensor or any iterable of integers."""
    if not isinstance(indices, torch.Tensor):
        indices = list(indices)
    return torch.as_tensor(indices, dtype=torch.long)


class SpinStatistic(torch.nn.Module, abc.ABC):
    """Untrainable spin statistics for Ising output statistics.

    A statistic is a :class:`torch.nn.Module` without parameters that maps spins of shape
    ``(B, M, N)`` to statistics of shape ``(B, M, dim_out)``. Subclasses implement
    :meth:`_transform`; calling the statistic validates the shapes of its input and output.

    .. caution:: These transformations should not contain any trainable parameters. While possible,
    current implementation does not accumulate gradients for parameters used in such functions.

    Args:
        dim_out: The output dimension of the statistic.

    Attributes:
        dim_out (int): Output dimension of the statistic.
    """

    def __init__(self, dim_out: int) -> None:
        super().__init__()
        self.dim_out = int(dim_out)

    @abc.abstractmethod
    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """The main function that will be invoked by ``forward``.

        Args:
            x (torch.Tensor): Input spins of shape (B, N, D) where B is batch size, N is sample size,
                and D is the number of spins.

        Returns:
            torch.Tensor: Output statistic applied to the third dimension of ``x``.
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the defined transformation.

        Args:
            x: Input spins of shape (B, N, D) where B is batch size, N is sample size, and D is the
                number of spins.

        Raises:
            ValueError: If input does not have 3 dimensions.
            ValueError: If output does not have 3 dimensions.
            ValueError: Output dimension does not match that of promised.

        Returns:
            torch.Tensor: Output statistic applied to the third dimension of ``x``.
        """
        if x.ndim != 3:
            raise ValueError("Input tensor should have `ndim == 3`.")

        output = self._transform(x)

        if output.ndim != 3:
            raise ValueError("Output tensor should have `ndim == 3`.")

        if output.shape[-1] != self.dim_out:
            raise ValueError(f"Output dimension ({output.shape[-1]}) does not match that "
                             f"promised ({self.dim_out}).")
        return output


class IdentityStatistic(SpinStatistic):
    """Identity function of statistics.

    Args:
        dim_out: Dimension of the inputs, which is also the dimension of the outputs.
    """

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Identity function."""
        return x


class IsingStatistic(SpinStatistic):
    """Sufficient statistics of Ising models; a concatenation of node values and pairwise products.

    Computes the concatenation of selected node spins and element-wise
    products of spin pairs, i.e. [x[..., indices], x[..., indices_j] * x[..., indices_i]].
    :meth:`from_graph` builds the statistic of all nodes and edges of a graph.

    Args:
        node_indices: Indices of nodes to include directly.
        endpoints_1: First indices for pairwise interaction terms.
        endpoints_2: Second indices for pairwise interaction terms.

    Attributes:
        node_indices (torch.Tensor): Buffer of the node indices.
        endpoints_1 (torch.Tensor): Buffer of the first indices of the interaction terms.
        endpoints_2 (torch.Tensor): Buffer of the second indices of the interaction terms.
    """

    def __init__(
        self,
        node_indices: Iterable[int],
        endpoints_1: Iterable[int],
        endpoints_2: Iterable[int]
    ) -> None:
        node_indices = _index_tensor(node_indices)
        endpoints_1 = _index_tensor(endpoints_1)
        endpoints_2 = _index_tensor(endpoints_2)
        if endpoints_1.numel() != endpoints_2.numel():
            raise ValueError(
                "Interaction indices should be of the same length, got "
                f"length {endpoints_1.numel()} for i and length {endpoints_2.numel()} for j"
            )
        super().__init__(node_indices.numel() + endpoints_1.numel())
        self.register_buffer("node_indices", node_indices, persistent=False)
        self.register_buffer("endpoints_1", endpoints_1, persistent=False)
        self.register_buffer("endpoints_2", endpoints_2, persistent=False)

    @classmethod
    def from_graph(cls, graph: GraphIndex) -> IsingStatistic:
        """The sufficient statistics of the Ising model on a graph: the spins of all nodes, in
        index order, followed by the products of spins along the edges, in the order of the
        graph's edges.

        Args:
            graph (GraphIndex): The graph, for example an :class:`~dwave.plugins.torch.nn.Ising`
                layer.

        Returns:
            IsingStatistic: A statistic with ``dim_out == graph.n_nodes + graph.n_edges``.
        """
        return cls(range(graph.n_nodes), graph.edge_idx_i, graph.edge_idx_j)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` and pairwise products defined by the given indices."""
        return torch.cat(
            [x[..., self.node_indices], x[..., self.endpoints_2] * x[..., self.endpoints_1]], -1
        )
