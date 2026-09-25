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
from dwave.plugins.torch.utils import GraphIndex

__all__ = ["TorchSampler"]


class TorchSampler(torch.nn.Module, abc.ABC):
    """Base class for all PyTorch plugin samplers.

    A sampler is a :class:`torch.nn.Module` bound to the graph of a
    :class:`~dwave.plugins.torch.utils.GraphIndex`, which it holds as the submodule
    :attr:`model`. Consequently :meth:`~torch.nn.Module.to`, :meth:`~torch.nn.Module.cuda`,
    :meth:`~torch.nn.Module.state_dict` and friends act on the graph module and on the sampler's
    own state (e.g. persistent Markov chains) together.

    There are two ways to sample. :meth:`sample` draws from the parameters of the model, which
    therefore has to be a :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine`;
    it supports conditional sampling of partially observed spins and, through :meth:`complete`,
    fills in the hidden units of observed data, which is how the positive phase of
    :meth:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine.quasi_objective` is
    formed for models with hidden units that cannot be marginalized exactly.
    :meth:`sample_biases` draws from Ising models on the same graph whose biases are given
    explicitly, one model per batch element, which is how the inputs of an
    :class:`~dwave.plugins.torch.nn.Ising` layer are sampled. Calling the sampler
    (``sampler(x)``) is equivalent to :meth:`sample`.

    Args:
        model (GraphIndex): The graph module the sampler is bound to: a
            :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` to sample from
            its parameters, or any :class:`~dwave.plugins.torch.utils.GraphIndex`, such as an
            :class:`~dwave.plugins.torch.nn.Ising` layer, to sample explicit biases on its graph.
    """

    def __init__(self, model: GraphIndex) -> None:
        super().__init__()
        if not isinstance(model, GraphIndex):
            raise TypeError(
                "`model` should be a GraphIndex, such as a GraphRestrictedBoltzmannMachine or an "
                f"Ising layer, got {type(model).__name__}."
            )
        self.model = model

    @abc.abstractmethod
    def sample(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Draw samples from the model's parameters.

        Args:
            x (torch.Tensor, optional): If ``None``, samples are drawn from the joint distribution
                of all model variables and returned with shape ``(num_samples, n_nodes)``.
                Otherwise ``x`` is a tensor of shape ``(..., n_nodes)`` of partially observed
                spins: entries equal to ``torch.nan`` are sampled conditioned on the ``±1``
                entries, which are kept fixed. The result has shape
                ``(..., num_samples, n_nodes)``.

        Raises:
            TypeError: If :attr:`model` is not a
                :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine`.

        Returns:
            torch.Tensor: Spin samples with entries in ``{-1, +1}``.
        """

    @abc.abstractmethod
    def sample_biases(self, linear: torch.Tensor, quadratic: torch.Tensor) -> torch.Tensor:
        """Draw samples from Ising models on the graph of :attr:`model` given by their biases.

        Args:
            linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``, one model per
                batch element; ``(n_nodes,)`` for a single model.
            quadratic (torch.Tensor): Dense quadratic biases of shape
                ``(*batch, n_nodes, n_nodes)`` in canonical orientation; entries outside the
                adjacency of the graph are ignored.

        Returns:
            torch.Tensor: Spins with entries in ``{-1, +1}`` of shape
            ``(*batch, num_samples, n_nodes)``, where the number of samples per model is
            determined by the sampler.
        """

    def forward(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Alias of :meth:`sample`."""
        return self.sample(x)

    def complete(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Complete observed visible spins with samples of the hidden units.

        For every row of ``x``, the hidden units of the model are sampled conditioned on the
        observed spins with :meth:`sample`. The result is suitable as the data argument of
        :meth:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine.quasi_objective`: its
        visible entries are the observations themselves, so gradients propagate to ``x``, and its
        hidden entries are samples, which are constants.

        Args:
            x (torch.Tensor): Observed spins of shape ``(..., n_visible)`` with one column per
                visible node, in the order of
                :attr:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine.visible_idx`.
            **kwargs: Keyword arguments passed on to :meth:`sample`, for example ``num_samples``
                of :class:`~dwave.plugins.torch.samplers.BlockSampler`.

        Returns:
            torch.Tensor: Spins of shape ``(..., num_samples, n_nodes)``.
        """
        model = self._require_model()
        padded = model.pad_visible(x)
        with torch.no_grad():
            samples = self.sample(padded, **kwargs)
        samples = samples.reshape(*x.shape[:-1], -1, model.n_nodes).clone()
        samples[..., model.visible_idx] = x.unsqueeze(-2).to(samples)
        return samples

    def _require_model(self) -> GraphRestrictedBoltzmannMachine:
        """The bound model, which must own parameters to be sampled from.

        Raises:
            TypeError: If :attr:`model` is not a
                :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine`.
        """
        if not isinstance(self.model, GraphRestrictedBoltzmannMachine):
            raise TypeError(
                "Sampling from the model's own biases requires a GraphRestrictedBoltzmannMachine; "
                f"this sampler is bound to a {type(self.model).__name__}. Use `sample_biases` to "
                "sample from explicit biases."
            )
        return self.model

    def _validate_conditional_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        """Validate partially observed spins, move them to the model's device and dtype, and
        flatten their batch dimensions.

        Args:
            x (torch.Tensor): Tensor of shape ``(..., n_nodes)`` with ``±1`` (observed) and
                ``torch.nan`` (to be sampled) entries.

        Raises:
            ValueError: If ``x`` has the wrong trailing dimension or contains values other than
                ``±1`` and ``torch.nan``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Size]: The validated spins flattened to shape
            ``(batch, n_nodes)``, a boolean mask of the same shape that is ``True`` where spins
            are observed (clamped), and the batch shape ``(...)`` of ``x`` with which to restore
            the leading dimensions of the result.
        """
        model = self._require_model()
        n_nodes = model.n_nodes
        x = torch.as_tensor(x)
        if x.ndim < 1 or x.shape[-1] != n_nodes:
            raise ValueError(
                f"x must have shape (..., {n_nodes}), got {tuple(x.shape)}."
            )
        batch_shape = x.shape[:-1]
        x = x.to(device=model.linear.device, dtype=model.linear.dtype)
        x = x.reshape(-1, n_nodes)
        clamp_mask = ~torch.isnan(x)
        if not torch.all(x[clamp_mask].abs() == 1):
            raise ValueError("x must contain only ±1 or NaN values.")
        return x, clamp_mask, batch_shape

    def _validate_biases(self, linear: torch.Tensor, quadratic: torch.Tensor) -> torch.Size:
        """Validate the shapes of explicit biases.

        Args:
            linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``.
            quadratic (torch.Tensor): Dense quadratic biases of shape
                ``(*batch, n_nodes, n_nodes)``.

        Raises:
            ValueError: If the shapes do not match each other or the graph.

        Returns:
            torch.Size: The batch shape ``(*batch)``.
        """
        n_nodes = self.model.n_nodes
        if linear.ndim < 1 or linear.shape[-1] != n_nodes:
            raise ValueError(
                f"linear must have shape (..., {n_nodes}), got {tuple(linear.shape)}."
            )
        batch_shape = linear.shape[:-1]
        if tuple(quadratic.shape) != (*batch_shape, n_nodes, n_nodes):
            raise ValueError(
                f"quadratic must have shape (..., {n_nodes}, {n_nodes}) = "
                f"{(*batch_shape, n_nodes, n_nodes)}, got {tuple(quadratic.shape)}."
            )
        return batch_shape
