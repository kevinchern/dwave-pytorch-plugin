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

import abc

import torch

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine

__all__ = ["TorchSampler"]


class TorchSampler(torch.nn.Module, abc.ABC):
    """Base class for all PyTorch plugin samplers.

    A sampler is a :class:`torch.nn.Module` that holds the
    :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` it samples from as the
    submodule :attr:`model`. Consequently :meth:`~torch.nn.Module.to`, :meth:`~torch.nn.Module.cuda`,
    :meth:`~torch.nn.Module.state_dict` and friends act on the model and on the sampler's own
    state (e.g. persistent Markov chains) together.

    Subclasses implement :meth:`sample`. Calling the sampler (``sampler(x)``) is equivalent to
    :meth:`sample`.

    Args:
        model (GraphRestrictedBoltzmannMachine): The model to sample from.
    """

    def __init__(self, model: GraphRestrictedBoltzmannMachine) -> None:
        super().__init__()
        if not isinstance(model, GraphRestrictedBoltzmannMachine):
            raise TypeError(
                "`model` should be a GraphRestrictedBoltzmannMachine, "
                f"got {type(model).__name__}."
            )
        self.model = model

    @abc.abstractmethod
    def sample(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Draw samples from the model.

        Args:
            x (torch.Tensor, optional): If ``None``, samples are drawn from the joint distribution
                of all model variables and returned with shape ``(num_samples, n_nodes)``.
                Otherwise ``x`` is a tensor of shape ``(..., n_nodes)`` of partially observed
                spins: entries equal to ``torch.nan`` are sampled conditioned on the ``±1``
                entries, which are kept fixed. The result has shape
                ``(..., num_samples, n_nodes)``.

        Returns:
            torch.Tensor: Spin samples with entries in ``{-1, +1}``.
        """

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Alias of :meth:`sample`."""
        return self.sample(x)

    def _validate_conditional_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate partially observed spins and move them to the model's device and dtype.

        Args:
            x (torch.Tensor): Tensor of shape ``(..., n_nodes)`` with ``±1`` (observed) and
                ``torch.nan`` (to be sampled) entries.

        Raises:
            ValueError: If ``x`` has the wrong trailing dimension or contains values other than
                ``±1`` and ``torch.nan``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The validated spins and a boolean mask of the same
            shape that is ``True`` where spins are observed (clamped).
        """
        n_nodes = self.model.n_nodes
        x = torch.as_tensor(x)
        if x.ndim < 1 or x.shape[-1] != n_nodes:
            raise ValueError(
                f"x must have shape (..., {n_nodes}), got {tuple(x.shape)}."
            )
        x = x.to(device=self.model.linear.device, dtype=self.model.linear.dtype)
        clamp_mask = ~torch.isnan(x)
        if not torch.all(x[clamp_mask].abs() == 1):
            raise ValueError("x must contain only ±1 or NaN values.")
        return x, clamp_mask
