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

from collections.abc import Iterable
from typing import Optional

import torch

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.samplers.block_spin_sampler import BlockSampler

__all__ = ["BipartiteGibbsSampler"]


class BipartiteGibbsSampler(BlockSampler):
    """A block-Gibbs sampler specialized for bipartite graph-restricted Boltzmann machines.

    The model's nodes must be partitioned into visible and hidden units with every edge
    connecting a visible unit to a hidden unit (a restricted Boltzmann machine). The two layers
    are the two blocks of a :class:`BlockSampler`: each Gibbs sweep samples all visible units
    given the hidden units and then all hidden units given the visible units.

    Args:
        model (GraphRestrictedBoltzmannMachine): The bipartite model to sample from.
        num_chains (int): Number of Markov chains to run in parallel. Defaults to 1.
        schedule (Iterable[float]): The inverse temperatures of the successive sweeps performed
            by each :meth:`sample` call. Defaults to ``(1.0,)``.
        initial_states (torch.Tensor, optional): A tensor of ``±1`` values of shape
            ``(num_chains, model.n_nodes)`` holding the initial states of the Markov chains. If
            ``None``, initial states are drawn uniformly at random. Defaults to ``None``.
        seed (int, optional): Seed of the sampler's private random number generator. Defaults
            to ``None``.

    Raises:
        ValueError: If an edge connects two visible or two hidden units.
    """

    def __init__(
        self,
        model: GraphRestrictedBoltzmannMachine,
        num_chains: int = 1,
        schedule: Iterable[float] = (1.0,),
        initial_states: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> None:
        hidden = set(model.hidden_nodes)
        offending = [(u, v) for u, v in model.edges if (u in hidden) == (v in hidden)]
        if offending:
            raise ValueError(
                "BipartiteGibbsSampler requires a bipartite model in which every edge connects "
                f"a visible unit and a hidden unit; offending edges: {offending[:5]}"
                f"{' ...' if len(offending) > 5 else ''}."
            )
        super().__init__(
            model,
            colouring=lambda node: node in hidden,
            num_chains=num_chains,
            schedule=schedule,
            proposal_acceptance_criteria="Gibbs",
            initial_states=initial_states,
            seed=seed,
        )
