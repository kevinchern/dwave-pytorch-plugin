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
    ``(..., n_x, F)`` and ``y`` of shape ``(..., n_y, F)``, whose items are ``F``-dimensional
    vectors, and returns the ``(..., n_x, n_y)`` matrices of kernel values of every item of ``x``
    with every item of ``y``; leading batch dimensions broadcast. Items with several feature
    dimensions are flattened by the caller, as
    :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss` does. Subclasses
    implement :meth:`_kernel`; calling the kernel checks that the feature dimensions of ``x`` and
    ``y`` agree.
    """

    @abc.abstractmethod
    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluates the kernel between every item of ``x`` and every item of ``y``.

        Args:
            x (torch.Tensor): A (..., n_x, F) tensor.
            y (torch.Tensor): A (..., n_y, F) tensor.

        Returns:
            torch.Tensor: A (..., n_x, n_y) tensor of kernel values.
        """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluates the kernel between every item of ``x`` and every item of ``y``.

        Args:
            x (torch.Tensor): A (..., n_x, F) tensor.
            y (torch.Tensor): A (..., n_y, F) tensor.

        Raises:
            ValueError: If the feature dimensions of ``x`` and ``y`` differ.

        Returns:
            torch.Tensor: A (..., n_x, n_y) tensor of kernel values.
        """
        if x.ndim < 2 or y.ndim < 2 or x.shape[-1] != y.shape[-1]:
            raise ValueError(
                "x and y must be (..., n, F) tensors with the same feature dimension F, got "
                f"shapes {tuple(x.shape)} and {tuple(y.shape)}."
            )
        return self._kernel(x, y)


class GaussianKernel(Kernel):
    r"""A sum of Gaussian kernels with geometrically spaced bandwidths.

    The kernel between two data points :math:`x` and :math:`y` is

    .. math::
        k(x, y) = \sum_{i=0}^{n_{\text{kernels}} - 1} \exp(-\|x - y\|^2 / \sigma_i),
        \qquad \sigma_i = \sigma \cdot \text{factor}^{\,i - \lfloor n_{\text{kernels}} / 2 \rfloor},

    where :math:`\|x - y\|` is the Euclidean distance between the items and :math:`\sigma` is
    the base bandwidth. If ``bandwidth`` is ``None``, :math:`\sigma` is set, without gradients and
    separately for every batch element, to the mean squared distance between distinct items of the
    sample the kernel matrix is evaluated on: the sum of the squared distances divided by
    :math:`n(n - 1)` for an :math:`n \times n` matrix (the plain mean for a matrix that is not
    square). This is the heuristic of `Sutherland et al. <https://arxiv.org/abs/1707.07269>`_,
    meant for evaluating the kernel on a sample against itself, as
    :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss` and
    :class:`~dwave.plugins.torch.nn.SquaredMMD` do with the pooled sample.

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
        entries of ``distance_matrix`` for every batch element.

        Args:
            distance_matrix (torch.Tensor): The (..., n, n) pairwise squared distances of a
                sample against itself, whose diagonal is zero.

        Raises:
            ValueError: If the bandwidth is to be estimated from fewer than two items.

        Returns:
            torch.Tensor | float: The base bandwidth, of shape ``(...)`` when estimated.
        """
        if self.bandwidth is not None:
            return self.bandwidth
        n_x, n_y = distance_matrix.shape[-2:]
        if n_x < 2 or n_y < 2:
            raise ValueError("Estimating the bandwidth requires at least two items.")
        diagonal = n_x if n_x == n_y else 0
        return distance_matrix.sum((-2, -1)) / (n_x * n_y - diagonal)

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distance_matrix = torch.cdist(x, y, p=2)**2
        bandwidth = torch.as_tensor(
            self._get_bandwidth(distance_matrix), dtype=distance_matrix.dtype, device=x.device
        )
        bandwidths = bandwidth.unsqueeze(-1) * self.factors                       # (..., n_kernels)
        return torch.exp(
            -distance_matrix.unsqueeze(-1) / bandwidths.unsqueeze(-2).unsqueeze(-2)
        ).sum(-1)
