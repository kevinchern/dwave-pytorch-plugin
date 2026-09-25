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

from typing import Any, Optional

import dimod
import torch

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.samplers.base import TorchSampler
from dwave.plugins.torch.utils import _scale_and_clip, sampleset_to_tensor, to_ising

__all__ = ["DimodSampler"]


class DimodSampler(TorchSampler):
    """PyTorch plugin wrapper for a dimod sampler.

    Unconditional sampling submits the model, scaled by ``prefactor`` and clipped to the given
    ranges (see :meth:`to_ising`), to :meth:`dimod.Sampler.sample_ising`. Conditional sampling
    builds, for every row of partially observed spins, the binary quadratic model of the
    unobserved variables: their effective fields (linear biases plus couplings to the observed
    spins, scaled by ``prefactor`` and clipped to ``linear_range``) and the couplings among them.

    Args:
        model (GraphRestrictedBoltzmannMachine): The model to sample from.
        sampler (dimod.Sampler): Dimod sampler.
        prefactor (float): The prefactor for which the Hamiltonian is scaled by. This quantity
            is typically the temperature at which the sampler operates at. Standard CPU-based
            samplers such as Metropolis- or Gibbs-based samplers will often default to sampling
            at an unit temperature, thus a unit prefactor should be used. In the case of a quantum
            annealer, a reasonable choice of a prefactor is 1/beta where beta is the effective
            inverse temperature and can be estimated using
            :meth:`GraphRestrictedBoltzmannMachine.estimate_beta`. Defaults to 1.
        linear_range (tuple[float, float], optional): Linear biases are clipped to
            ``linear_range`` prior to sampling. This clipping occurs after the ``prefactor``
            scaling has been applied. When None, no clipping is applied. Defaults to None.
        quadratic_range (tuple[float, float], optional): Quadratic biases are clipped to
            ``quadratic_range`` prior to sampling. This clipping occurs after the ``prefactor``
            scaling has been applied. When None, no clipping is applied. Defaults to None.
        sample_kwargs (dict[str, Any], optional): Keyword arguments for the dimod sampler.

    Attributes:
        sampler (dimod.Sampler): The wrapped dimod sampler.
        prefactor (float): The scaling applied to the Hamiltonian prior to sampling.
        linear_range (tuple[float, float] | None): The range linear biases are clipped to.
        quadratic_range (tuple[float, float] | None): The range quadratic biases are clipped to.
        sample_kwargs (dict[str, Any]): Keyword arguments passed to the dimod sampler.
    """

    def __init__(
        self,
        model: GraphRestrictedBoltzmannMachine,
        sampler: dimod.Sampler,
        prefactor: float = 1.0,
        linear_range: Optional[tuple[float, float]] = None,
        quadratic_range: Optional[tuple[float, float]] = None,
        sample_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(model)
        self.sampler = sampler
        self.prefactor = float(prefactor)
        self.linear_range = None if linear_range is None else tuple(linear_range)
        self.quadratic_range = None if quadratic_range is None else tuple(quadratic_range)
        self.sample_kwargs = dict(sample_kwargs or {})
        self._sample_set: Optional[dimod.SampleSet] = None

    @property
    def sample_set(self) -> dimod.SampleSet:
        """The sample set returned by the dimod sampler in the latest :meth:`sample` call (for
        conditional sampling, the one of the last row).

        Raises:
            RuntimeError: If :meth:`sample` has not been called yet.
        """
        if self._sample_set is None:
            # NOTE: an AttributeError would be swallowed by ``torch.nn.Module.__getattr__``
            raise RuntimeError("no samples found; call 'sample()' first")
        return self._sample_set

    def to_ising(self) -> tuple[dict, dict]:
        """The model in Ising format as it is submitted to the dimod sampler: biases scaled by
        :attr:`prefactor` and clipped to :attr:`linear_range` and :attr:`quadratic_range`.

        Returns:
            tuple[dict, dict]: Linear biases keyed by node and quadratic biases keyed by the edges
            of the model; see :func:`~dwave.plugins.torch.utils.to_ising`.
        """
        model = self.model
        return to_ising(
            model.nodes, model.edges, model.linear, model.edge_biases(),
            self.prefactor, self.linear_range, self.quadratic_range,
        )

    def sample(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from the dimod sampler and return the corresponding tensor.

        Args:
            x (torch.Tensor, optional): Partially observed spins of shape ``(..., n_nodes)``;
                entries equal to ``torch.nan`` are sampled, ``±1`` entries are kept fixed. If
                ``None``, samples are drawn from the joint distribution.

        Raises:
            ValueError: If ``x`` has an invalid shape, contains values other than ``±1`` or
                ``torch.nan``, or the sampler returns a different number of samples for
                different rows.

        Returns:
            torch.Tensor: Spins with entries in ``{-1, +1}`` of shape ``(num_reads, n_nodes)`` if
            ``x`` is ``None`` and ``(..., num_reads, n_nodes)`` otherwise.
        """
        model = self.model
        device = model.linear.device

        if x is None:
            h, J = self.to_ising()
            self._sample_set = self.sampler.sample_ising(h, J, **self.sample_kwargs)
            return sampleset_to_tensor(model.nodes, self._sample_set, device)

        x, clamp_mask, batch_shape = self._validate_conditional_input(x)
        n_nodes = model.n_nodes

        # Linear biases of the free variables conditioned on the observed spins, for all rows at
        # once (observed entries only contribute; NaN entries contribute nothing).
        with torch.no_grad():
            fields = _scale_and_clip(model.effective_field(x), self.prefactor, self.linear_range)
        _, J = self.to_ising()

        x_cpu, fields_cpu, free_cpu = x.cpu(), fields.cpu(), (~clamp_mask).cpu()
        nodes = model.nodes
        reduced_models: dict[bytes, tuple[torch.Tensor, list, dict]] = {}
        results = []
        for row in range(x_cpu.shape[0]):
            free = free_cpu[row]
            key = free.numpy().tobytes()
            if key not in reduced_models:
                free_idx = torch.nonzero(free).flatten()
                free_nodes = [nodes[idx] for idx in free_idx.tolist()]
                free_set = set(free_nodes)
                J_free = {e: b for e, b in J.items() if e[0] in free_set and e[1] in free_set}
                reduced_models[key] = (free_idx, free_nodes, J_free)
            free_idx, free_nodes, J_free = reduced_models[key]

            if not free_nodes:
                num_reads = int(self.sample_kwargs.get("num_reads", 1))
                results.append(x_cpu[row].expand(num_reads, n_nodes).clone())
                continue

            h_free = dict(zip(free_nodes, fields_cpu[row, free_idx].tolist()))
            bqm = dimod.BinaryQuadraticModel.from_ising(h_free, J_free)
            self._sample_set = self.sampler.sample(bqm, **self.sample_kwargs)
            free_samples = sampleset_to_tensor(free_nodes, self._sample_set)
            full = x_cpu[row].expand(free_samples.shape[0], n_nodes).clone()
            full[:, free_idx] = free_samples.to(full.dtype)
            results.append(full)

        num_reads = {result.shape[0] for result in results}
        if len(num_reads) != 1:
            raise ValueError(
                f"Expected all samples to have shape ({results[0].shape[0]}, {n_nodes}), got "
                f"sample sizes {sorted(num_reads)}."
            )
        return torch.stack(results).reshape(*batch_shape, -1, n_nodes).to(device)
