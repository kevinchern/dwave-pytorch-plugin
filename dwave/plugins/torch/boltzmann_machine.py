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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from dimod import BinaryQuadraticModel
from dwave.system.temperatures import maximum_pseudolikelihood_temperature

from dwave.plugins.torch.utils import sample_to_tensor, spread

if TYPE_CHECKING:
    from dimod import Sampler

__all__ = [
    "GraphRestrictedBoltzmannMachine",
]


class AbstractBoltzmannMachine(ABC, torch.nn.Module):
    """Abstract class for Boltzmann machines.

    Args:
        h_range (tuple[float, float], optional): Range of linear weights.
            If ``None``, uses an infinite range.
        j_range (tuple[float, float], optional): Range of quadratic weights.
            If ``None``, uses an infinite range.
        hidx (torch.Tensor, optional): Indices of hidden units.
            If ``None``, the model is fully visible.
    """

    def __init__(
        self,
        h_range: tuple[float, float] = None,
        j_range: tuple[float, float] = None,
        hidx: torch.Tensor = None,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "h_range",
            torch.tensor(h_range if h_range is not None else [-torch.inf, torch.inf]),
        )
        self.register_buffer(
            "j_range",
            torch.tensor(j_range if j_range is not None else [-torch.inf, torch.inf]),
        )
        self.fully_visible = hidx is None
        if hidx is not None:
            self.register_buffer("hidx", hidx)

        if (h_range and not j_range) or (j_range and not h_range):
            raise NotImplementedError(
                "Both or neither weight range should be specified."
            )

        self.register_forward_pre_hook(lambda *args: self.clip_parameters())

    @abstractmethod
    def sufficient_statistics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the sufficient statistics of a Boltzmann machine, i.e., average spin
        and average interaction values (per edge) of ``x``.

        Args:
            x (torch.Tensor): A tensor of shape (..., N) where N denotes the number of
                variables in the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The sufficient statistics of ``x``.
        """

    def estimate_beta(self, spins: torch.Tensor) -> float:
        """Estimate the maximum pseudolikelihood temperature using
        ``dwave.system.temperature``.

        Args:
            spins (torch.Tensor): A tensor of shape (b, N) where b is the sample size,
            and N denotes the number of variables in the model.

        Returns:
            float: the estimated effective inverse temperature of the model."""
        h, J = self.ising
        bqm = BinaryQuadraticModel.from_ising(h, J)
        beta = 1.0 / maximum_pseudolikelihood_temperature(bqm, spins.numpy())[0]
        return beta

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.fully_visible:
            raise ValueError("Fully-visible models should not `_pad` data.")
        bs, n_vis = x.shape
        n_hid = self.hidx.shape[0]
        n = n_vis + n_hid
        padded = torch.ones((bs, n)) * torch.nan
        padded[:, self.vidx] = x
        return padded

    @abstractmethod
    def _compute_effective_field(self) -> torch.Tensor:
        """Compute the effective field of disconnected hidden units.

        Returns:
            torch.Tensor: effective field of hidden units
        """

    def compute_expectation_disconnected(
        self, spins: torch.Tensor, beta: float
    ) -> torch.Tensor:
        """Assuming the missing spins are disconnected from each other, compute the
        conditional expectation of missing spins (as indicated by `torch.nan`s).

        Args:
            corrupted_spins (torch.Tensor): a tensor of spins with shape (b, N) where b
            is the samlpe size and N is the number of variables in the model. Entries
            with values `torch.nan` are assumed to be disconnected from each other and
            the resulting tensor will have these values populated with expectations.

            beta (float): the effective inverse temperature of the model and sampler.
            This quantity is, in the typical context of using a D-Wave QPU, estimated
            with samples from the QPU and `AbstractBoltzmannMachine.estimate_beta`.

        Returns:
            torch.Tensor: a tensor of spins, as in `corrupted_spins`, but with
            `torch.nan`s replaced by the expected values.
        """
        padded = self._pad(spins)
        h_eff = self._compute_effective_field(padded)
        padded[:, self.hidx] = -torch.tanh(h_eff * beta)
        return padded

    def clip_parameters(self) -> None:
        """Clips linear and quadratic bias weights in-place."""
        self.get_parameter("h").data.clamp_(*self.h_range)
        self.get_parameter("J").data.clamp_(*self.j_range)

    @property
    def ising(self) -> tuple[dict, dict]:
        """Converts the model to Ising format."""
        self.clip_parameters()
        return self._ising

    @property
    @abstractmethod
    def _ising(self) -> tuple[dict, dict]:
        """Convert the model to Ising format."""

    def objective(
        self, s_observed: torch.Tensor, s_model: torch.Tensor
    ) -> torch.Tensor:
        """An objective function with gradients equivalent to the gradients of the
        negative log likelihood.

        Args:
            s_observed (torch.Tensor): Tensor of observed spins (data) with shape
                (b1, N) where b1 denotes the batch size and N denotes the number of
                variables in the model.
            s_model (torch.Tensor): Tensor of spins drawn from the model with shape
                (b2, N) where b2 denotes the batch size and N denotse the number of
                variables in the model.

        Returns:
            torch.Tensor: Scalar difference of the average energy of data and model.
        """
        self.clip_parameters()
        return self(s_observed).mean() - self(s_model).mean()

    def sample(
        self, sampler: Sampler, device: torch.device = None, **sample_params: dict
    ) -> torch.Tensor:
        """Sample from the Boltzmann machine.

        This method samples and converts a sample of spins to tensors and ensures they
        are not aggregated---provided the aggregation information is retained in the
        sample set.

        Args:
            sampler (Sampler): The sampler used to sample from the model.
            sampler_params (dict): Parameters of the `sampler.sample` method.
            device (torch.device, optional): The device of the constructed tensor.
                If ``None`` and data is a tensor then the device of data is used.
                If ``None`` and data is not a tensor then the result tensor is
                constructed on the current device.

        Returns:
            torch.Tensor: Spins sampled from the model
                (shape prescribed by ``sampler`` and ``sample_params``).
        """
        h, J = self.ising
        ss = spread(sampler.sample_ising(h, J, **sample_params))
        spins = sample_to_tensor(ss, device=device)
        return spins


class GraphRestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Creates a graph-restricted Boltzmann machine.

    Args:
        num_nodes (int): Number of variables in the model.
        edge_idx_i (torch.Tensor): List of endpoints i of a list of edges.
        edge_idx_j (torch.Tensor): List of endpoints j of a list of edges.
        h_range (tuple[float, float], optional): Range of linear weights.
            If ``None``, uses an infinite range.
        j_range (tuple[float, float], optional): Range of quadratic weights.
            If ``None``, uses an infinite range.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_idx_i: torch.Tensor,
        edge_idx_j: torch.Tensor,
        *,
        h_range: tuple = None,
        j_range: tuple = None,
        hidx: torch.Tensor | None = None,
    ):
        super().__init__(h_range=h_range, j_range=j_range, hidx=hidx)

        num_edges = len(edge_idx_i)
        if edge_idx_i.size(0) != edge_idx_j.size(0):
            raise ValueError("Endpoints 'edge_idx_i' and 'edge_idx_j' are mismatched")

        if torch.unique(torch.cat([edge_idx_i, edge_idx_j])).size(0) > num_nodes:
            raise ValueError(
                "Vertices are required to be contiguous nonnegative integers starting from 0 (inclusive). The input edge set implies otherwise."
            )
        self.register_buffer("edge_idx_i", edge_idx_i)
        self.register_buffer("edge_idx_j", edge_idx_j)

        if not self.fully_visible:
            vidx = torch.tensor([i for i in range(num_nodes) if i not in self.hidx])
            self.register_buffer("vidx", vidx)
            adj = list()
            bidx = list()
            bin_pointer = -1
            J_indices = torch.arange(num_edges)
            jidx = list()
            for idx in self.hidx.tolist():
                mask_i = self.edge_idx_i == idx
                mask_j = self.edge_idx_j == idx
                edges = torch.cat([self.edge_idx_j[mask_i], self.edge_idx_i[mask_j]])
                jidx.extend(J_indices[mask_i + mask_j].tolist())
                bin_pointer += edges.shape[0]
                bidx.append(bin_pointer)
                adj.extend(edges.tolist())
            self.register_buffer("flat_adj", torch.tensor(adj))
            self.register_buffer("jidx", torch.tensor(jidx))
            self.register_buffer("bidx", torch.tensor(bidx))

        self.h = torch.nn.Parameter(0.01 * (2 * torch.randint(0, 2, (num_nodes,)) - 1))
        self.J = torch.nn.Parameter(1.0 * (2 * torch.randint(0, 2, (num_edges,)) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the Hamiltonian.

        Args:
            x (torch.tensor): A tensor of shape (B, N) where B denotes batch size and
                N denotes the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonians of shape (B,).
        """
        return x @ self.h + self.interactions(x) @ self.J

    def interactions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute interactions prescribed by the model's edges.

        Args:
            x (torch.tensor): Tensor of shape (..., N) where N denotes the number of
                variables in the model.

        Returns:
            torch.tensor: Tensor of interaction terms of shape (..., M) where M denotes
                the number of edges in the model.
        """
        return x[..., self.edge_idx_i] * x[..., self.edge_idx_j]

    def _compute_effective_field(self, padded: torch.Tensor) -> torch.Tensor:
        bs = padded.shape[0]

        self.clip_parameters()
        contribution = padded[:, self.flat_adj] * self.J[self.jidx]
        cc = contribution.cumsum(1)
        h_eff = self.h[self.hidx] + cc[:, self.bidx].diff(
            dim=1, prepend=torch.zeros(bs).unsqueeze(1)
        )
        return h_eff

    def sufficient_statistics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the sufficient statistics of a Boltzmann machine, i.e., average spin
        and average interaction values (per edge) of ``x``.

        Args:
            x (torch.Tensor): A tensor of shape (..., N) where N denotes the number of
                variables in the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The sufficient statistics of ``x``.
        """
        interactions = self.interactions(x)
        return x.mean(dim=0), interactions.mean(dim=0)

    @property
    def _ising(self) -> tuple[dict, dict]:
        """Convert the model to Ising format"""
        linear_biases = self.h.detach().cpu().tolist()
        edge_idx_i = self.edge_idx_i.detach().cpu().tolist()
        edge_idx_j = self.edge_idx_j.detach().cpu().tolist()
        quadratic_bias_list = self.J.detach().cpu().tolist()
        quadratic_biases = {
            (a, b): w for a, b, w in zip(edge_idx_i, edge_idx_j, quadratic_bias_list)
        }

        return linear_biases, quadratic_biases
