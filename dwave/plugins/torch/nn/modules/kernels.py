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
"""Kernel functions."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn

__all__ = ["Kernel", "GaussianKernel"]


class Kernel(nn.Module, abc.ABC):
    """Base class for kernels.

    `Kernels <https://en.wikipedia.org/wiki/Kernel_method>`_ are functions that compute a similarity
    measure between data points. A kernel is called with two samples, ``x`` of shape
    ``(n_x, f1, f2, ..., fk)`` and ``y`` of shape ``(n_y, f1, f2, ..., fk)``, and returns the
    ``(n_x, n_y)`` matrix of kernel values of every item of ``x`` with every item of ``y``.
    Subclasses implement :meth:`_kernel`; calling the kernel checks that the feature shapes of
    ``x`` and ``y`` agree.
    """

    @abc.abstractmethod
    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluates the kernel between every item of ``x`` and every item of ``y``.

        Args:
            x (torch.Tensor): A (n_x, f1, f2, ..., fk) tensor.
            y (torch.Tensor): A (n_y, f1, f2, ..., fk) tensor of the same feature shape.

        Returns:
            torch.Tensor: A (n_x, n_y) tensor of kernel values.
        """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluates the kernel between every item of ``x`` and every item of ``y``.

        Args:
            x (torch.Tensor): A (n_x, f1, f2, ..., fk) tensor.
            y (torch.Tensor): A (n_y, f1, f2, ..., fk) tensor.

        Raises:
            ValueError: If the feature shapes of ``x`` and ``y`` differ.

        Returns:
            torch.Tensor: A (n_x, n_y) tensor of kernel values.
        """
        if x.shape[1:] != y.shape[1:]:
            raise ValueError(
                f"Feature shapes of x and y must match, got {tuple(x.shape)} and {tuple(y.shape)}."
            )
        return self._kernel(x, y)


class GaussianKernel(Kernel):
    r"""A sum of Gaussian kernels with geometrically spaced bandwidths.

    The kernel between two data points :math:`x` and :math:`y` is

    .. math::
        k(x, y) = \sum_{i=0}^{n_{\text{kernels}} - 1} \exp(-\|x - y\|^2 / \sigma_i),
        \qquad \sigma_i = \sigma \cdot \text{factor}^{\,i - \lfloor n_{\text{kernels}} / 2 \rfloor},

    where :math:`\|x - y\|` is the Euclidean distance between the flattened features and
    :math:`\sigma` is the base bandwidth. If ``bandwidth`` is ``None``, :math:`\sigma` is set,
    without gradients, to the mean squared distance between distinct items of the sample the
    kernel matrix is evaluated on: the sum of the squared distances divided by :math:`n(n - 1)`
    for an :math:`n \times n` matrix. This is the heuristic of
    `Sutherland et al. <https://arxiv.org/abs/1707.07269>`_, meant for evaluating the kernel on
    a sample against itself, as
    :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss` does with the
    pooled sample.

    Args:
        n_kernels (int): Number of bandwidths, i.e. of Gaussian kernels summed.
        factor (float): Ratio between successive bandwidths. Defaults to 2.
        bandwidth (float, optional): The base bandwidth :math:`\sigma`. If ``None``, it is
            estimated from the data as described above. Defaults to ``None``.

    Attributes:
        factors (torch.Tensor): Buffer of the multipliers
            :math:`\text{factor}^{\,i - \lfloor n_{\text{kernels}} / 2 \rfloor}` of the base
            bandwidth, shape ``(n_kernels,)``.
        bandwidth (float | None): The base bandwidth, or ``None`` for the data-dependent one.
    """

    def __init__(
        self, n_kernels: int, factor: float = 2.0, bandwidth: float | None = None
    ) -> None:
        super().__init__()
        exponents = torch.arange(n_kernels, dtype=torch.get_default_dtype()) - n_kernels // 2
        self.register_buffer("factors", float(factor) ** exponents)
        self.bandwidth = bandwidth

    @torch.no_grad()
    def _get_bandwidth(self, distance_matrix: torch.Tensor) -> torch.Tensor | float:
        """The base bandwidth: :attr:`bandwidth` if given, otherwise the mean of the off-diagonal
        entries of ``distance_matrix``.

        Args:
            distance_matrix (torch.Tensor): The (n, n) pairwise squared distances of a sample
                against itself, whose diagonal is zero.

        Raises:
            ValueError: If the bandwidth is to be estimated from fewer than two items.

        Returns:
            torch.Tensor | float: The base bandwidth.
        """
        if self.bandwidth is not None:
            return self.bandwidth
        num_samples = distance_matrix.shape[0]
        if num_samples < 2:
            raise ValueError("Estimating the bandwidth requires at least two items.")
        return distance_matrix.sum() / (num_samples * (num_samples - 1))

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distance_matrix = torch.cdist(x.flatten(1), y.flatten(1), p=2)**2
        bandwidth = self._get_bandwidth(distance_matrix) * self.factors
        return torch.exp(-distance_matrix.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)).sum(dim=0)
