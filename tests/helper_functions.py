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

import unittest.mock

from inspect import signature

import numpy as np
import torch
from dimod import BinaryQuadraticModel

from dwave.plugins.torch.graph import randspin
from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.utils import to_bqm


def set_weights(bm: GRBM, linear, quadratic) -> None:
    """Set the linear biases and the per-edge quadratic biases (in edge order) of a model."""
    with torch.no_grad():
        bm.linear.copy_(torch.as_tensor(linear, dtype=bm.linear.dtype))
        bm.quadratic.copy_(torch.as_tensor(quadratic, dtype=bm.quadratic.dtype))


def model_to_bqm(bm: GRBM) -> BinaryQuadraticModel:
    """The model as a dimod binary quadratic model."""
    return to_bqm(bm, bm.linear, bm.quadratic)


def randspins(*shape, seed: int = 0) -> torch.Tensor:
    """Random spins of the given shape, drawn from a generator seeded with ``seed``."""
    return randspin(shape, generator=torch.Generator().manual_seed(seed)).float()


def enumerate_walsh(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every state of ``{-1, +1}^n`` and every parity character, by direct enumeration.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``states`` of shape ``(2^n, n)`` (state ``a``
        has spin ``1 - 2 * bit_i(a)`` at coordinate ``i``), ``orders`` of shape ``(2^n,)`` (the
        size of subset ``S``), and ``chi`` of shape ``(2^n, 2^n)`` with ``chi[S, a]`` the parity
        ``prod_{i in S} states[a, i]``.
    """
    m = 1 << n
    states = np.array([[1 - 2 * ((z >> i) & 1) for i in range(n)] for z in range(m)], dtype=np.int8)
    orders = np.array([bin(s).count("1") for s in range(m)])
    chi = np.ones((m, m))
    for subset in range(m):
        for i in range(n):
            if (subset >> i) & 1:
                chi[subset] *= states[:, i]
    return states, orders, chi


# ---------------------------------------------------------------- exact tests of samplers -------

class RecordedBernoulli:
    """Replaces ``torch.bernoulli`` by a deterministic threshold rule and records the probabilities
    it is called with, so that sampling code is tested exactly rather than statistically.

    A draw is 1 where the probability exceeds ``threshold`` and 0 elsewhere.

    Args:
        threshold: The probability above which a draw is 1. Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.probabilities: list[torch.Tensor] = []
        self._patch = unittest.mock.patch("torch.bernoulli", side_effect=self._draw)

    def _draw(self, probabilities: torch.Tensor, *, generator=None) -> torch.Tensor:
        self.probabilities.append(probabilities.clone())
        return (probabilities > self.threshold).to(probabilities.dtype)

    def __enter__(self) -> RecordedBernoulli:
        self._patch.start()
        return self

    def __exit__(self, *exc) -> None:
        self._patch.stop()


def constant_randspin(value: float = 1.0):
    """Patches the block sampler's random spin initialization to return spins equal to ``value``."""
    return unittest.mock.patch(
        "dwave.plugins.torch.samplers.block_spin_sampler.randspin",
        side_effect=lambda size, **kwargs: torch.full(tuple(size), value),
    )


def exact_update_probabilities(graph, x, linear, quadratic, block, beta, criterion) -> torch.Tensor:
    """The probabilities a block update must draw with, computed from exact energies.

    For ``"Gibbs"`` these are the conditional probabilities of the spins of ``block`` being ``+1``
    given all other spins; for ``"Metropolis"`` the acceptance probabilities of flipping them.
    Biases follow the batching convention of ``GraphIndex.energy``.

    Returns:
        torch.Tensor: Probabilities of shape ``(*x.shape[:-1], len(block))``.
    """
    probabilities = []
    for node in block.tolist():
        plus, minus = x.clone(), x.clone()
        plus[..., node], minus[..., node] = 1.0, -1.0
        e_plus = graph.energy(plus, linear, quadratic)
        e_minus = graph.energy(minus, linear, quadratic)
        if criterion == "Gibbs":
            probabilities.append(torch.sigmoid(-beta * (e_plus - e_minus)))
        else:
            current = graph.energy(x, linear, quadratic)
            flipped = torch.where(x[..., node] > 0, e_minus, e_plus)
            probabilities.append(torch.exp(-beta * (flipped - current)).clamp(max=1.0))
    return torch.stack(probabilities, -1)


def replay_sweep(graph, x, linear, quadratic, partition, beta, criterion, recorded,
                 clamp_mask=None, clamped_values=None, threshold: float = 0.5) -> torch.Tensor:
    """Replays one block sweep from the probabilities recorded by :class:`RecordedBernoulli`.

    Asserts that every recorded probability is the exact one for the state at that point of the
    sweep, applies the threshold draws as the sampler does, restores clamped spins, and returns
    the resulting state.
    """
    x = x.clone()
    for block, probabilities in zip(partition, recorded, strict=True):
        expected = exact_update_probabilities(graph, x, linear, quadratic, block, beta, criterion)
        torch.testing.assert_close(probabilities, expected)
        draw = (probabilities > threshold).to(x.dtype)
        if criterion == "Gibbs":
            x[..., block] = 2 * draw - 1
        else:
            x[..., block] = x[..., block] * (1 - 2 * draw)
        if clamp_mask is not None:
            x[..., block] = torch.where(
                clamp_mask[..., block], clamped_values[..., block], x[..., block]
            )
    return x


def model_probably_good(
        model: torch.nn.Module, shape_in: tuple[int, ...], shape_out: tuple[int, ...]
) -> bool:
    """Checks whether the model output has expected shape, is probably unconstrained, and the model
    has its configs stored.

    This function generates dummy data with a padded batch dimension on top of the
    input dimension (so ``shape_in`` should exclude a batch dimension). The data is passed through
    the ``model``. Subsequent tests are described in ``shapes_match``, ``probably_unconstrained``,
    and ``has_correct_config``.

    Args:
        model (torch.nn.Module): The module to be tested.
        shape_in (tuple[int, ...]): Input data shape excluding the batch dimension.
        shape_out (tuple[int, ...]): Output data shape excluding the batch dimension.

    Returns:
        bool: Indicator for whether the model meets the three conditions above.
    """
    bs = 100
    x = torch.randn((bs, ) + shape_in)
    y = model(x)
    padded_out = (bs,)+shape_out
    return (shapes_match(y, padded_out)
            and probably_unconstrained(y)
            and has_correct_config(model))


def has_correct_config(model: torch.nn.Module) -> bool:
    """Checks whether the model has its initialization arguments stored in a ``config`` field.

    Args:
        model (torch.nn.Module): The module to be tested.

    Returns:
        bool: Indicator for whether the model has its initialization arguments stored.
    """
    if not hasattr(model, "config"):
        return False
    sig = signature(model.__init__)
    return set(model.config.keys()) == set(sig.parameters.keys()) | {"module_name"}


def shapes_match(x: torch.Tensor, y: tuple[int, ...]) -> bool:
    """Checks whether `x.shape` is equal to `y`.

    Args:
        x (torch.Tensor): A tensor.
        y (tuple[int, ...]): The expected shape.

    Returns:
        bool: Indicator for whether the shape is as expected.
    """
    return tuple(x.shape) == y


def are_all_spins(x: torch.Tensor) -> bool:
    """Checks all entries of `x` are one in absolute value.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: indicator for whether all entries of `x` are in ``{-1, 1}``.
    """
    return (x.float().abs() == 1).all()


def has_mixed_signs(x: torch.Tensor) -> bool:
    """Checks whether `x` has both positive and negative values.

    Args:
        x (torch.Tensor): A tensor to be cast to type float.

    Returns:
        bool: Indicator for whether `x` consists of both positive and negative values.
    """
    return bool(x.max() > 0 and x.min() < 0)


def has_zeros(x: torch.Tensor) -> bool:
    """Checks whether `x` has exact zeros.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: Indicator for whether `x` has any zero-valued entries.
    """
    return (x == 0).float().any()


def bounded_in_plus_minus_one(x: torch.Tensor) -> bool:
    """Checks whether all entries of `x` are in ``[-1, 1]``.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: Indicator for whether all values of `x` are in ``[-1, 1]``.
    """
    return bool((x.abs() <= 1).all())


def probably_unconstrained(x: torch.Tensor):
    """Checks whether `x` has any activation-like constraints.
    Checks `x` has no exact zeros, not bounded in ``[-1, 1]``, and has both positive and
    negative-valued entries.

    Args:
        x (torch.Tensor): A tensor.

    Returns:
        bool: Indicator for whether `x` passes the "constraints".
    """
    return not has_zeros(x) and not bounded_in_plus_minus_one(x) and has_mixed_signs(x)
