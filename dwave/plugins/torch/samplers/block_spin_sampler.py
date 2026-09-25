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
from typing import Literal, Optional

import networkx as nx
import torch

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.samplers.base import TorchSampler
from dwave.plugins.torch.tensor import randspin
from dwave.plugins.torch.utils import GraphIndex

__all__ = ["BlockSampler"]


class BlockSampler(TorchSampler):
    r"""A block-spin update sampler for Ising models on a graph.

    The nodes of the graph are partitioned into blocks (colour classes) such that no edge
    connects two nodes of the same block. Given all other spins, the spins of a block are
    conditionally independent, so a whole block is updated at once from its effective fields
    :math:`h^{\text{eff}} = h + s J_{\text{sym}}`, a single dense matrix product per block.

    Bound to a :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine`, the sampler
    keeps ``num_chains`` persistent Markov chains in :attr:`state`; every call to :meth:`sample`
    without arguments advances all chains through the inverse-temperature ``schedule`` and
    returns them. Conditional sampling (:meth:`sample` with partially observed spins) runs on a
    temporary state and leaves the persistent chains untouched. :meth:`sample_biases` samples a
    batch of Ising models given by their biases on the same graph with one matrix product per
    block for the whole batch, and is available whatever the sampler is bound to, for example an
    :class:`~dwave.plugins.torch.nn.Ising` layer.

    Block-Gibbs and Block-Metropolis obey detailed balance and are ergodic methods at finite
    nonzero temperature which, at fixed parameters, converge upon Boltzmann distributions.
    Block-Metropolis allows higher acceptance rates for proposals (faster single-step mixing), but
    is non-ergodic in the limit of zero or infinite temperature. Decorrelation from an initial
    condition can be slower. Block-Gibbs represents best practice for independent sampling.

    Args:
        model (GraphIndex): The model to sample from, or the graph module (for example an Ising
            layer) on whose graph explicit biases are sampled.
        colouring (Callable[[Hashable], Hashable], optional): A function mapping every node of
            ``model`` to its colour; nodes of one colour form a block and adjacent nodes must
            have different colours. If ``None``, a greedy colouring is computed. Defaults to
            ``None``.
        num_chains (int): Number of persistent Markov chains to run in parallel. Defaults to 1.
        schedule (Iterable[float]): The inverse temperatures of the successive sweeps performed
            by each :meth:`sample` or :meth:`sample_biases` call. Defaults to ``(1.0,)``, a
            single sweep at unit inverse temperature.
        proposal_acceptance_criteria (Literal["Gibbs", "Metropolis"]): The proposal acceptance
            criterion used to accept or reject states in the Markov chain. Defaults to "Gibbs".
        initial_states (torch.Tensor, optional): A tensor of ``±1`` values of shape
            ``(num_chains, model.n_nodes)`` holding the initial states of the Markov chains. If
            ``None``, initial states are drawn uniformly at random. Defaults to ``None``.
        seed (int, optional): Seed of the sampler's private random number generator. If ``None``,
            the global PyTorch generator is used. Defaults to ``None``.

    Raises:
        ValueError: If ``num_chains`` is not positive, the acceptance criterion is unknown, the
            schedule is empty, ``colouring`` is not a proper colouring, or ``initial_states`` has
            the wrong shape or non-spin values.

    Attributes:
        state (torch.Tensor): Buffer holding the current states of the persistent Markov chains,
            of shape ``(num_chains, n_nodes)``.
        schedule (tuple[float, ...]): The inverse temperatures of the successive sweeps of one
            sampling call.
        proposal_acceptance_criteria (str): The proposal acceptance criterion, ``"Gibbs"`` or
            ``"Metropolis"``.
    """

    def __init__(
        self,
        model: GraphIndex,
        colouring: Optional[Callable[[Hashable], Hashable]] = None,
        num_chains: int = 1,
        schedule: Iterable[float] = (1.0,),
        proposal_acceptance_criteria: Literal["Gibbs", "Metropolis"] = "Gibbs",
        initial_states: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(model)

        if num_chains < 1:
            raise ValueError("Number of chains should be a positive integer.")
        criterion = str(proposal_acceptance_criteria).title()
        if criterion not in ("Gibbs", "Metropolis"):
            raise ValueError(
                'Proposal acceptance criterion should be one of "Gibbs" or "Metropolis".'
            )
        self.proposal_acceptance_criteria = criterion
        self.schedule = tuple(float(beta) for beta in schedule)
        if not self.schedule:
            raise ValueError("`schedule` should contain at least one inverse temperature.")
        self._seed = None if seed is None else int(seed)
        self._generator: Optional[torch.Generator] = None

        # The blocks are consecutive slices of the node indices sorted by colour. The indices
        # are a (non-persistent) buffer so that the blocks live on the device of the model.
        block_idx, self._block_ptr = self._partition_nodes(colouring)
        self.register_buffer("_block_idx", block_idx, persistent=False)

        initial_states = self._prepare_initial_states(num_chains, initial_states)
        if isinstance(model, GraphRestrictedBoltzmannMachine):
            initial_states = initial_states.to(model.linear)
        else:
            initial_states = initial_states.to(model.adjacency.device)
        self.register_buffer("state", initial_states)

    # ------------------------------------------------------------------ setup --------------------

    @staticmethod
    def _monochromatic_edges(
        model: GraphIndex, colour_idx: torch.Tensor
    ) -> list[tuple[Hashable, Hashable]]:
        """The edges of ``model`` whose endpoints have the same colour.

        Args:
            model (GraphIndex): The graph module.
            colour_idx (torch.Tensor): The colour of every node, shape ``(n_nodes,)``, on the CPU.

        Returns:
            list[tuple[Hashable, Hashable]]: The offending edges, in the order of ``model.edges``.
        """
        same = colour_idx[model.edge_idx_i.cpu()] == colour_idx[model.edge_idx_j.cpu()]
        return [model.edges[k] for k in same.nonzero().flatten().tolist()]

    def _partition_nodes(
        self, colouring: Optional[Callable[[Hashable], Hashable]]
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Partition the node indices into blocks of equal colour, ordered by colour.

        Args:
            colouring (Callable[[Hashable], Hashable], optional): See the class docstring.

        Raises:
            ValueError: If two adjacent nodes have the same colour.

        Returns:
            tuple[torch.Tensor, tuple[int, ...]]: The node indices sorted by colour, and the
            offsets at which the blocks start and end (one more than the number of blocks).
        """
        model = self.model
        if colouring is None:
            graph = nx.Graph()
            graph.add_nodes_from(model.nodes)
            graph.add_edges_from(model.edges)
            colour_of = nx.greedy_color(graph, strategy="largest_first")
            colours = [colour_of[node] for node in model.nodes]
        else:
            colours = [colouring(node) for node in model.nodes]

        keys = list(dict.fromkeys(colours))
        try:
            keys = sorted(keys)
        except TypeError:
            keys = sorted(keys, key=repr)
        colour_index = {colour: k for k, colour in enumerate(keys)}
        colour_idx = torch.tensor([colour_index[c] for c in colours], dtype=torch.long)

        offending = self._monochromatic_edges(model, colour_idx)
        if offending:
            raise ValueError(
                "`colouring` is not a valid colouring of the model: the endpoints of the edges "
                f"{offending[:5]}{' ...' if len(offending) > 5 else ''} have the same colour."
            )
        block_idx = torch.argsort(colour_idx, stable=True)
        counts = torch.bincount(colour_idx, minlength=len(keys))
        block_ptr = (0, *torch.cumsum(counts, 0).tolist())
        return block_idx, block_ptr

    def _prepare_initial_states(
        self, num_chains: int, initial_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Validate the initial states or draw them uniformly at random.

        Args:
            num_chains (int): Number of chains.
            initial_states (torch.Tensor, optional): A tensor of ``±1`` values of shape
                ``(num_chains, model.n_nodes)``. If ``None``, random spins are drawn.

        Raises:
            ValueError: If ``initial_states`` has the wrong shape or contains non-spin values.

        Returns:
            torch.Tensor: The initial states as a floating-point tensor.
        """
        n_nodes = self.model.n_nodes
        if initial_states is None:
            generator = None if self._seed is None else torch.Generator().manual_seed(self._seed)
            return randspin((num_chains, n_nodes), generator=generator).float()

        initial_states = torch.as_tensor(initial_states)
        if tuple(initial_states.shape) != (num_chains, n_nodes):
            raise ValueError(
                "Initial states should be of shape (num_chains, n_nodes) = "
                f"{(num_chains, n_nodes)}, but got {tuple(initial_states.shape)} instead."
            )
        if not torch.all(initial_states.abs() == 1):
            raise ValueError("Initial states contain nonspin values.")
        return initial_states.float()

    # ------------------------------------------------------------------ properties ---------------

    @property
    def partition(self) -> list[torch.Tensor]:
        """The blocks of node indices, one tensor per colour (in sorted colour order)."""
        ptr = self._block_ptr
        return [self._block_idx[start:stop] for start, stop in zip(ptr[:-1], ptr[1:])]

    @property
    def num_chains(self) -> int:
        """Number of persistent Markov chains."""
        return self.state.shape[0]

    @property
    def seed(self) -> Optional[int]:
        """Seed of the sampler's random number generator, or ``None``."""
        return self._seed

    def _rng(self, device: Optional[torch.device] = None) -> Optional[torch.Generator]:
        """The sampler's random number generator on ``device`` (by default the device of its
        state), or ``None`` when the global generator is used. The generator is re-seeded when
        the device changes."""
        if self._seed is None:
            return None
        device = self.state.device if device is None else torch.device(device)
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self._seed)
        return self._generator

    # ------------------------------------------------------------------ updates ------------------

    @torch.no_grad()
    def _gibbs_update(
        self,
        beta: float | torch.Tensor,
        block: torch.Tensor,
        effective_field: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
        """Performs a Gibbs update of ``block`` in-place.

        Args:
            beta (float | torch.Tensor): The (scalar) inverse temperature to sample at.
            block (torch.Tensor): Indices of the nodes of one block.
            effective_field (torch.Tensor): Effective fields of the block, shape
                ``(..., len(block))``.
            x (torch.Tensor): Spins to update, shape ``(..., n_nodes)``.
        """
        prob = torch.sigmoid(-2.0 * beta * effective_field)
        x[..., block] = 2.0 * torch.bernoulli(prob, generator=self._rng(x.device)) - 1.0

    @torch.no_grad()
    def _metropolis_update(
        self,
        beta: float | torch.Tensor,
        block: torch.Tensor,
        effective_field: torch.Tensor,
        x: torch.Tensor,
    ) -> None:
        """Performs a Metropolis update of ``block`` in-place.

        Every spin of the block is proposed to flip; a flip that lowers the energy is always
        accepted, otherwise it is accepted with probability ``exp(-beta * delta_energy)``.

        Args:
            beta (float | torch.Tensor): The (scalar) inverse temperature to sample at.
            block (torch.Tensor): Indices of the nodes of one block.
            effective_field (torch.Tensor): Effective fields of the block, shape
                ``(..., len(block))``.
            x (torch.Tensor): Spins to update, shape ``(..., n_nodes)``.
        """
        current = x[..., block]
        delta_energy = -2.0 * current * effective_field
        prob = torch.exp(-beta * delta_energy).clamp_(max=1.0)
        flip = torch.bernoulli(prob, generator=self._rng(x.device))
        x[..., block] = current * (1.0 - 2.0 * flip)

    @torch.no_grad()
    def _step(
        self,
        beta: float | torch.Tensor,
        x: torch.Tensor,
        linear: torch.Tensor,
        coupling: torch.Tensor,
        clamp_mask: Optional[torch.Tensor] = None,
        clamped_values: Optional[torch.Tensor] = None,
    ) -> None:
        """Performs one sweep, i.e. a block-spin update of every block, in-place.

        Args:
            beta (float | torch.Tensor): Inverse temperature to sample at.
            x (torch.Tensor): Spins to update, shape ``(..., n_nodes)``; for batched biases,
                ``(*batch, M, n_nodes)``.
            linear (torch.Tensor): Linear biases, shape ``(n_nodes,)`` or ``(*batch, n_nodes)``.
            coupling (torch.Tensor): The symmetric coupling matrix of the quadratic biases, shape
                ``(n_nodes, n_nodes)`` or ``(*batch, n_nodes, n_nodes)``.
            clamp_mask (torch.Tensor, optional): Boolean tensor of the shape of ``x`` that is
                ``True`` where spins are clamped to ``clamped_values``.
            clamped_values (torch.Tensor, optional): Values of the clamped spins; only read where
                ``clamp_mask`` is ``True``.
        """
        gibbs = self.proposal_acceptance_criteria == "Gibbs"
        for block in self.partition:
            effective_field = self.model.effective_field(
                x, linear=linear, coupling=coupling, idx=block
            )
            if gibbs:
                self._gibbs_update(beta, block, effective_field, x)
            else:
                self._metropolis_update(beta, block, effective_field, x)
            if clamp_mask is not None:
                x[..., block] = torch.where(
                    clamp_mask[..., block], clamped_values[..., block], x[..., block]
                )

    @torch.no_grad()
    def sample(self, x: Optional[torch.Tensor] = None, num_samples: int = 1) -> torch.Tensor:
        """Performs block updates on the model's parameters.

        Without ``x``, every persistent chain is advanced by one sweep per inverse temperature in
        :attr:`schedule` and the chains are returned. With ``x``, the ``torch.nan`` entries of
        ``x`` are sampled conditioned on its ``±1`` entries: ``num_samples`` copies of every row
        of ``x`` are initialized with random spins at the unobserved entries, swept through the
        schedule with the observed spins held fixed, and returned. The persistent chains are not
        modified. If the unobserved spins of a row all belong to a single block, one Gibbs sweep
        yields an exact conditional sample.

        Args:
            x (torch.Tensor, optional): Partially observed spins of shape ``(..., n_nodes)`` with
                ``torch.nan`` marking the spins to sample. Defaults to ``None``.
            num_samples (int): Number of conditional samples per row of ``x``. Defaults to 1.

        Raises:
            TypeError: If the sampler is not bound to a
                :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine`.

        Returns:
            torch.Tensor: Spins of shape ``(num_chains, n_nodes)`` if ``x`` is ``None``, otherwise
            of shape ``(..., num_samples, n_nodes)``.
        """
        model = self._require_model()
        linear, coupling = model.linear, model.symmetric_coupling()
        if x is None:
            for beta in self.schedule:
                self._step(beta, self.state, linear, coupling)
            return self.state.clone()

        if num_samples < 1:
            raise ValueError("`num_samples` should be a positive integer.")
        x, clamp_mask, batch_shape = self._validate_conditional_input(x)
        x = x.repeat_interleave(num_samples, 0)
        clamp_mask = clamp_mask.repeat_interleave(num_samples, 0)

        random_spins = randspin(x.shape, generator=self._rng(x.device), device=x.device).to(x.dtype)
        state = torch.where(clamp_mask, x, random_spins)
        for beta in self.schedule:
            self._step(beta, state, linear, coupling, clamp_mask, x)
        return state.reshape(*batch_shape, num_samples, model.n_nodes)

    @torch.no_grad()
    def sample_biases(
        self, linear: torch.Tensor, quadratic: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        """Performs block updates on a batch of Ising models given by their biases.

        For every model, ``num_samples`` chains are initialized with random spins and swept once
        per inverse temperature in :attr:`schedule`; every block is updated for all models and
        chains with a single batched matrix product. The persistent chains are not modified.

        Args:
            linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``, one model per
                batch element; ``(n_nodes,)`` for a single model.
            quadratic (torch.Tensor): Dense quadratic biases of shape
                ``(*batch, n_nodes, n_nodes)`` in canonical orientation.
            num_samples (int): Number of chains, i.e. samples, per model. Defaults to 1.

        Raises:
            ValueError: If the biases have inconsistent shapes or ``num_samples`` is not positive.

        Returns:
            torch.Tensor: Spins of shape ``(*batch, num_samples, n_nodes)``.
        """
        if num_samples < 1:
            raise ValueError("`num_samples` should be a positive integer.")
        batch_shape = self._validate_biases(linear, quadratic)
        linear = linear.detach()
        coupling = self.model.symmetric_coupling(quadratic.detach())
        state = randspin(
            (*batch_shape, num_samples, self.model.n_nodes),
            generator=self._rng(linear.device),
            device=linear.device,
        ).to(linear.dtype)
        for beta in self.schedule:
            self._step(beta, state, linear, coupling)
        return state
