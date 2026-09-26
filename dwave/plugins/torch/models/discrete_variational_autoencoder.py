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

# The use of the discrete autoencoder implementations below (including the
# DiscreteVariationalAutoencoder) with a quantum computing system is
# protected by the intellectual property rights of D-Wave Quantum Inc.
# and its affiliates.
#
# The use of the discrete autoencoder implementations below (including the
# DiscreteVariationalAutoencoder) with D-Wave's quantum computing
# system will require access to D-Wave’s LeapTM quantum cloud service and
# will be governed by the Leap Cloud Subscription Agreement available at:
# https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/
#

from __future__ import annotations

from collections.abc import Callable

import torch

from dwave.plugins.torch.nn.functional import gumbel_spins

__all__ = ["DiscreteVariationalAutoencoder"]


class DiscreteVariationalAutoencoder(torch.nn.Module):
    """DiscreteAutoEncoder architecture amenable for training discrete models as priors.
    See https://iopscience.iop.org/article/10.1088/2632-2153/aba220

    Such discrete models include spin-variable models amenable for the QPU. This
    architecture is a modification of the standard autoencoder architecture, where
    the encoder outputs a latent representation of the data, and the decoder
    reconstructs the data from the latent representation. In our case, there is an
    additional step where the latent representation is mapped to a discrete
    representation, which is then passed to the decoder.

    Args:
        encoder (torch.nn.Module): The encoder must output latents that are later on
            passed to ``latent_to_discrete``. An encoder has signature (x) -> l. x has
            shape (batch_size, f1, f2, ...) and l has shape (batch_size, l1, l2, ...).
        decoder (torch.nn.Module): Decodes discrete tensors into data tensors. A decoder
            has signature (d) -> x'. d has shape (batch_size, n, d1, d2, ...) and x' has
            shape (batch_size, f'1, f'2, ...); if x' is the reconstructed data then
            fi=f'i, but x' might be another representation of the data (e.g. in a
            text-to-image model, x is a sequence of tokens, and x' is an image). Note
            that the decoder input is of shape (batch_size, n, d1, d2, ...), where n is
            a number of discrete representations to be created from a single latent
            representation of a single initial data point.
        latent_to_discrete (Callable[[torch.Tensor, int], torch.Tensor] | None): A
            stochastic and differentiable function that maps the output of the encoder
            to a discrete representation (a function is deterministic by definition;
            here "stochastic" means the function implicitly takes an additional noise
            variables as input). Importantly, since the function is stochastic, it
            allows for the creation of multiple discrete representations from the latent
            representation of a single data point. Thus, the signature of this function
            is (l, n) -> d, where l is the output of the encoder and has shape
            (batch_size, l1, l2, ...), n is the number of discrete representations per
            data point, and d has shape (batch_size, n, d1, d2, ...), which will be the
            input to the decoder. If None,
            :func:`~dwave.plugins.torch.nn.functional.gumbel_spins` is used, which
            interprets the latents as the logits of spins. Defaults to None.

    Attributes:
        encoder (torch.nn.Module): The encoder, a submodule.
        decoder (torch.nn.Module): The decoder, a submodule.
        latent_to_discrete (Callable[[torch.Tensor, int], torch.Tensor]): The map from latents
            to discrete representations.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_to_discrete: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_to_discrete = gumbel_spins if latent_to_discrete is None else latent_to_discrete

    def forward(
        self, x: torch.Tensor, n_samples: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ingests data into the :class:`DiscreteVariationalAutoencoder`.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, ...).
            n_samples (int, optional): Since the ``latent_to_discrete`` map is, in
                general, stochastic (see :class:`DiscreteVariationalAutoencoder` for more on this),
                several different discrete samples can be obtained by applying this map
                to the same encoded data point. This argument specifies how many such
                samples are obtained. Defaults to 1.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The latents, i.e. the encoded data,
            of shape (batch_size, ...); the discrete representation(s) of the encoded data, of
            shape (batch_size, n_samples, ...); and the reconstructed data, of shape
            (batch_size, n_samples, ...).
        """
        latents = self.encoder(x)
        discretes = self.latent_to_discrete(latents, n_samples)
        xhat = self.decoder(discretes)
        return latents, discretes, xhat
