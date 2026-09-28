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
"""Statistics of a set of samples, for the :class:`~dwave.plugins.torch.nn.Ising` layer.

A statistic is a :class:`torch.nn.Module` that maps a set of samples of shape ``(..., M, N)``, and
any further inputs, to values of shape ``(..., D)``, and is symmetric in the ``M`` samples. It may
provide ``leave_one_out(spins, *inputs)``, the values of the statistic on every set of all samples
but one, of shape ``(..., M, D)``; :func:`~dwave.plugins.torch.nn.functional.expectation` uses it
to attach an unbiased gradient of the statistic's expectation to its value, and evaluates the
statistic once per sample when it is absent.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from dwave.plugins.torch.nn.modules.kernels import Kernel

__all__ = ["Mean", "SquaredMMD"]


class Mean(torch.nn.Module):
    r"""The sample mean of a per-sample transform,
    :math:`F(s_1, \dots, s_M) = \frac1M \sum_m g(s_m)`.

    With this statistic the :class:`~dwave.plugins.torch.nn.Ising` layer estimates the expectation
    :math:`\mathbb E[g(S)]` of the transform; with the identity transform, the default, it
    estimates the mean spins, and ``Mean(graph.statistics)`` estimates the sufficient statistics of
    the Ising model on ``graph``. The leave-one-out values are closed form, so the gradient of the
    layer costs nothing beyond the transform; the resulting weights are the centred transforms
    divided by ``M - 1``, i.e. the gradient is the unbiased sample covariance of the transform
    with the sufficient statistics.

    Args:
        transform (Callable, optional): Maps spins of shape ``(..., M, N)``, and any further inputs
            of the statistic, to values of shape ``(..., M, D)`` sample by sample. A
            :class:`torch.nn.Module` is registered as a submodule and its parameters receive
            gradients through the layer. If ``None``, the identity. Defaults to ``None``.

    Attributes:
        transform (Callable | None): The transform.
    """

    def __init__(self, transform: Callable[..., torch.Tensor] | None = None) -> None:
        super().__init__()
        self.transform = transform

    def per_sample(self, spins: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        """The transform of every sample, shape ``(..., M, D)``."""
        return spins if self.transform is None else self.transform(spins, *inputs)

    def forward(self, spins: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        """The mean over the samples of the transform, shape ``(..., D)``."""
        return self.per_sample(spins, *inputs).mean(-2)

    def leave_one_out(self, spins: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        """The mean over all samples but one, for every sample, shape ``(..., M, D)``."""
        values = self.per_sample(spins, *inputs)
        return (values.sum(-2, keepdim=True) - values) / (values.shape[-2] - 1)


class SquaredMMD(torch.nn.Module):
    r"""The unbiased estimate of the squared maximum mean discrepancy between the samples and a
    reference sample.

    For samples :math:`s_1, \dots, s_M` and a reference sample :math:`z_1, \dots, z_R`,

    .. math::

        F(s; z) = \frac{1}{M(M-1)} \sum_{m \ne m'} k(s_m, s_{m'})
                - \frac{2}{MR} \sum_{m, r} k(s_m, z_r)
                + \frac{1}{R(R-1)} \sum_{r \ne r'} k(z_r, z_{r'}),

    the estimator of :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss`,
    here for batches of sample sets. As a function of the samples it is a U-statistic of degree
    two, so the leave-one-out gradient of the :class:`~dwave.plugins.torch.nn.Ising` layer is
    unbiased for the gradient of its expectation,
    :math:`-2\,\mathrm{Cov}(w(S), T(S))` with the witness
    :math:`w(s) = \mathbb E_{s'}[k(s, s')] - \frac1R \sum_r k(s, z_r)` and the sufficient
    statistics :math:`T`. The reference sample receives its gradient through the kernel, so an
    encoder producing ``z`` and a Boltzmann machine producing the samples are trained by one loss.

    The leave-one-out values are closed form in the row sums of the kernel matrices and cost
    nothing beyond the value. The kernel is evaluated once, on the pooled samples and reference
    sample; a kernel whose bandwidth is estimated from the data (see
    :class:`~dwave.plugins.torch.nn.GaussianKernel`) therefore uses one bandwidth for all terms,
    which the leave-one-out values hold fixed.

    Args:
        kernel (Kernel): The kernel.

    Attributes:
        kernel (Kernel): The kernel, a submodule.
    """

    def __init__(self, kernel: Kernel) -> None:
        super().__init__()
        self.kernel = kernel

    def _row_sums(
        self, spins: torch.Tensor, reference: torch.Tensor, min_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row sums of the kernel matrices.

        Args:
            spins (torch.Tensor): Samples of shape ``(..., M, N)``.
            reference (torch.Tensor): Reference sample of shape ``(..., R, N)`` or ``(R, N)``.
            min_samples (int): The least number of samples ``M`` that the caller requires.

        Raises:
            ValueError: If there are fewer than ``min_samples`` samples or fewer than two reference
                items.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The sums over the other samples of the
            kernel with every sample, shape ``(..., M)``; the means over the reference of the
            kernel with every sample, shape ``(..., M)``; and the mean kernel between distinct
            reference items, shape ``(...)``.
        """
        n_samples, n_reference = spins.shape[-2], reference.shape[-2]
        if n_samples < min_samples:
            raise ValueError(f"At least {min_samples} samples are required, got {n_samples}.")
        if n_reference < 2:
            raise ValueError(f"At least two reference items are required, got {n_reference}.")
        reference = reference.expand(*spins.shape[:-2], *reference.shape[-2:])
        pooled = torch.cat([spins, reference], -2)
        gram = self.kernel(pooled, pooled)
        samples = gram[..., :n_samples, :n_samples]
        cross = gram[..., :n_samples, n_samples:]
        others = gram[..., n_samples:, n_samples:]
        between_samples = samples.sum(-1) - samples.diagonal(dim1=-2, dim2=-1)
        to_reference = cross.mean(-1)
        within_reference = (
            others.sum((-2, -1)) - others.diagonal(dim1=-2, dim2=-1).sum(-1)
        ) / (n_reference * (n_reference - 1))
        return between_samples, to_reference, within_reference

    def forward(self, spins: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """The squared maximum mean discrepancy estimate, shape ``(..., 1)``.

        Args:
            spins (torch.Tensor): Samples of shape ``(..., M, N)``, at least two per set.
            reference (torch.Tensor): Reference sample of shape ``(..., R, N)`` or ``(R, N)``, at
                least two items.
        """
        between, to_reference, within = self._row_sums(spins, reference, min_samples=2)
        n_samples = spins.shape[-2]
        value = between.sum(-1) / (n_samples * (n_samples - 1)) - 2 * to_reference.mean(-1) + within
        return value.unsqueeze(-1)

    def leave_one_out(self, spins: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """The estimate on every set of all samples but one, shape ``(..., M, 1)``; requires at
        least three samples per set."""
        between, to_reference, within = self._row_sums(spins, reference, min_samples=3)
        n_samples = spins.shape[-2]
        between_others = (between.sum(-1, keepdim=True) - 2 * between) / (
            (n_samples - 1) * (n_samples - 2)
        )
        to_reference_others = (
            n_samples * to_reference.mean(-1, keepdim=True) - to_reference
        ) / (n_samples - 1)
        return (between_others - 2 * to_reference_others + within.unsqueeze(-1)).unsqueeze(-1)
