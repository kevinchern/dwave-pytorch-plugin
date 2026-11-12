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
"""Functional interface."""
from __future__ import annotations

import torch

from dwave.plugins.torch.kernels import (
    _check_soft_spins,
    _off_diagonal_distances,
    _validate_alpha,
    kernel_means,
)

__all__ = ["maximum_mean_discrepancy_loss", "bit2spin_soft", "spin2bit_soft"]


def maximum_mean_discrepancy_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor | None = None,
    block_size: int | None = None,
) -> torch.Tensor:
    r"""Unbiased estimates of the squared maximum mean discrepancy (MMD) between two samples of
    spins under every kernel of the battery, differentiable in both samples.

    The `squared MMD <https://jmlr.org/papers/v13/gretton12a.html>`_ between the distributions
    :math:`p` and :math:`q` of the samples under a kernel :math:`k` is

    .. math::
        \mathrm{MMD}^2(p, q) = \mathbb E_{x, x' \sim p}[k(x, x')]
            + \mathbb E_{y, y' \sim q}[k(y, y')] - 2\,\mathbb E_{x \sim p, y \sim q}[k(x, y)],

    which is zero if :math:`p = q` and positive otherwise for a characteristic kernel. Its
    unbiased estimate from :math:`n_x` samples :math:`x_i` and :math:`n_y` samples :math:`y_j`
    is the U-statistic

    .. math::
        \frac{1}{n_x (n_x - 1)} \sum_{i \ne i'} k(x_i, x_{i'})
        + \frac{1}{n_y (n_y - 1)} \sum_{j \ne j'} k(y_j, y_{j'})
        - \frac{2}{n_x n_y} \sum_{i, j} k(x_i, y_j).

    The kernels are the Hamming-isotropic battery of :mod:`dwave.plugins.torch.kernels`, and
    the result holds one estimate per profile: entry 0 the headline kernel, the Gaussian kernel
    :math:`e^{-d_H / \sqrt{dim}}` or, when kernel weights ``alpha`` are given, the alpha row
    :math:`\sum_k \alpha_k K_k / \binom{dim}{k}`, and entry :math:`k = 1..dim` the order-:math:`k`
    kernel, whose estimate is the mean squared error per order-:math:`k` parity coefficient (the
    order-:math:`k` Fourier weight of the mismatch). The kernels are evaluated at the real
    distance :math:`d = (dim - \langle x, y \rangle) / 2`: on hard spins this is the Hamming
    distance and every entry is exact, and on soft spins in ``[-1, 1]`` (a ``tanh`` output, or
    the straight-through spins of :func:`gumbel_spins`) the loss is a smooth surrogate whose
    minimizers at the corners of the cube are the hard-spin ones. Between the corners the
    headline's Gaussian surrogate :math:`e^{-(dim - \langle x, y \rangle) / (2 \sqrt{dim})}` is
    still a positive-definite kernel, so entry 0 is a squared MMD of soft points too; the
    order-``k`` surrogates for ``k >= 2`` are polynomials in the inner product that are not
    positive definite off the corners, so those entries are smooth extensions of the hard-spin
    estimates rather than squared MMDs of the soft points. Both samples receive their gradient
    through the inner products; the order kernels are summed by
    :class:`~dwave.plugins.torch.kernels.KravchukSums` in memory that does not grow with
    ``dim``.

    With ``block_size`` given, both samples must hold the same number of rows ``m`` and are cut
    into ``m / block_size`` blocks scored block against block and averaged, the paired
    construction of :meth:`~dwave.plugins.torch.evaluation.Evaluator.score`, with Gram work and
    memory falling from ``O(m^2)`` to ``O(m block_size)``. Nothing is cast: ``x``, ``y`` and
    ``alpha`` must share one dtype and device.

    Args:
        x (torch.Tensor): A ``(n_x, dim)`` tensor of samples from distribution ``p``, hard or
            soft spins in ``[-1, 1]`` of a floating dtype, ``n_x >= 2``.
        y (torch.Tensor): A ``(n_y, dim)`` tensor of samples from distribution ``q``, likewise.
        alpha (torch.Tensor, optional): Kernel weights over orders ``1..dim``, a 1-D tensor of
            ``dim`` finite nonnegative entries with at least one positive, for example the
            weights an :class:`~dwave.plugins.torch.evaluation.Evaluator` calibrated. Defaults
            to ``None``, the Gaussian headline.
        block_size (int, optional): Rows per block, ``>= 2`` and a divisor of the common row
            count. Defaults to ``None``, one block of all rows.

    Raises:
        ValueError: If ``x`` or ``y`` is not a 2-D floating tensor of finite values in
            ``[-1, 1]`` with at least two rows, their feature dimensions differ, ``alpha`` is
            invalid, or ``block_size`` is given and is below 2, does not divide the rows, or
            the row counts differ.

    Returns:
        torch.Tensor: A ``(dim + 1,)`` tensor in the samples' dtype: per profile, the unbiased
        estimate of the squared MMD, differentiable in ``x`` and ``y``.
    """
    _check_soft_spins(x, "x")
    _check_soft_spins(y, "y")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"x and y must be 2-D (n, dim) samples, got shapes {tuple(x.shape)} and "
            f"{tuple(y.shape)}."
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Feature shapes of x and y must match, got {tuple(x.shape)} and {tuple(y.shape)}."
        )
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError(
            "x and y must each hold at least two samples, got shapes "
            f"{tuple(x.shape)} and {tuple(y.shape)}."
        )
    dim = x.shape[1]
    if alpha is not None:
        _validate_alpha(alpha, dim)
    if block_size is None:
        blocks = [(x, y)]
    else:
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "x and y must hold the same number of rows to be scored block against block, "
                f"got {x.shape[0]} and {y.shape[0]}."
            )
        if block_size < 2 or x.shape[0] % block_size:
            raise ValueError(
                f"block_size must be >= 2 and divide the {x.shape[0]} rows, got {block_size!r}."
            )
        blocks = list(zip(x.split(int(block_size)), y.split(int(block_size))))
    per_block = []
    for x_block, y_block in blocks:
        d_xx = _off_diagonal_distances(x_block, dim)
        d_yy = _off_diagonal_distances(y_block, dim)
        d_xy = (dim - x_block @ y_block.mT) / 2
        per_block.append(
            kernel_means(d_xx.reshape(-1), dim, alpha)
            + kernel_means(d_yy.reshape(-1), dim, alpha)
            - 2 * kernel_means(d_xy.reshape(-1), dim, alpha)
        )
    return torch.stack(per_block).mean(0)


def bit2spin_soft(b: torch.Tensor) -> torch.Tensor:
    """Maps input :math:`b` to :math:`2b-1`.

    The mapping does not require :math:`b` to be binary, only that it is in the interval :math:`[0, 1]`.

    Args:
        b (torch.Tensor): Input tensor of values in :math:`[0, 1]`.

    Raises:
        ValueError: If not all ``b`` values are in :math:`[0, 1]`.

    Returns:
        torch.Tensor: A tensor with values :math:`2b-1`.
    """
    if not ((b >= 0) & (b <= 1)).all():
        raise ValueError(f"Not all inputs are in [0, 1]: {b}")
    return b * 2 - 1


def spin2bit_soft(s: torch.Tensor) -> torch.Tensor:
    """Maps input :math:`s` to :math:`(s+1)/2`.

    The mapping does not require :math:`s` to be spin-valued, only that it is in the interval :math:`[-1, 1]`.

    Args:
        s (torch.Tensor): Input tensor of values in :math:`[-1, 1]`.

    Raises:
        ValueError: If not all ``s`` values are in `[-1, 1]`.

    Returns:
        torch.Tensor: A tensor with values :math:`(s+1)/2`.
    """
    if (s.abs() > 1).any():
        raise ValueError(f"Not all inputs are in [-1, 1]: {s}")
    return (s + 1) / 2
