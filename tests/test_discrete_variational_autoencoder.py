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
"""Unit tests of the discrete variational autoencoder: its wiring, its default discretization and
the objectives it is trained with. The losses, the Boltzmann machine and the samplers have their
own tests, so no training loops are needed here."""

import unittest

import torch
from dimod import ExactSolver
from parameterized import parameterized

from dwave.plugins.torch.models import DiscreteVariationalAutoencoder as DVAE
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.nn.functional import gumbel_spins
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.nn.functional import pseudo_kl_divergence_loss
from dwave.plugins.torch.nn.modules.kernels import GaussianKernel
from dwave.plugins.torch.samplers import DimodSampler

# Data in the corners of the unit square and the spin strings they encode to
CORNERS = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
SPINS = 2 * CORNERS - 1


class BitEncoder(torch.nn.Module):
    """A parameter-free encoder mapping bits to logits of ±20, at which the Gumbel noise flips a
    spin with probability sigmoid(-20), i.e. never in practice. ``n_latent_dims - 1`` dummy
    dimensions are inserted before the feature dimension."""

    def __init__(self, n_latent_dims: int = 1) -> None:
        super().__init__()
        self.n_latent_dims = n_latent_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = x * 40 - 20
        for _ in range(self.n_latent_dims - 1):
            logits = logits.unsqueeze(-2)
        return logits


class LinearDecoder(torch.nn.Module):
    """A linear decoder of (batch_size, n_samples, ...) discrete representations."""

    def __init__(self, n_latent: int, n_features: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_latent, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.reshape(*x.shape[:2], -1))


def sign_map(logits: torch.Tensor, n_samples: int) -> torch.Tensor:
    """A deterministic straight-through discretization: the signs of the logits, repeated."""
    spins = torch.sign(logits) - logits.detach() + logits
    return spins.unsqueeze(1).expand(-1, n_samples, *logits.shape[1:])


class TestDiscreteVariationalAutoencoder(unittest.TestCase):

    def test_submodules_and_default_map(self):
        encoder, decoder = BitEncoder(), LinearDecoder(2, 2)
        dvae = DVAE(encoder, decoder)
        self.assertIs(encoder, dvae.encoder)
        self.assertIs(decoder, dvae.decoder)
        self.assertDictEqual({"encoder": encoder, "decoder": decoder}, dict(dvae.named_children()))
        self.assertTrue(all(key.startswith("decoder.") for key in dvae.state_dict()))
        self.assertIs(gumbel_spins, dvae.latent_to_discrete)

        with self.subTest("A custom map is stored as given"):
            self.assertIs(sign_map, DVAE(encoder, decoder, sign_map).latent_to_discrete)

    @parameterized.expand([(1, 1), (1, 5), (2, 1), (2, 3)])
    def test_forward(self, n_latent_dims, n_samples):
        # With a deterministic map the outputs are the exact composition encoder -> map -> decoder
        encoder, decoder = BitEncoder(n_latent_dims), LinearDecoder(2, 2)
        dvae = DVAE(encoder, decoder, sign_map)
        latents, discretes, xhat = dvae(CORNERS, n_samples=n_samples)
        torch.testing.assert_close(latents, encoder(CORNERS))
        torch.testing.assert_close(discretes, sign_map(encoder(CORNERS), n_samples))
        torch.testing.assert_close(xhat, decoder(discretes))
        self.assertEqual((4, n_samples, *([1] * (n_latent_dims - 1)), 2), tuple(discretes.shape))
        self.assertEqual((4, n_samples, 2), tuple(xhat.shape))

    @parameterized.expand([1, 5])
    def test_default_map_encodes_bits_as_spins(self, n_samples):
        # Strong logits make the Gumbel-softmax discretization deterministic: bits become spins
        _, discretes, _ = DVAE(BitEncoder(), LinearDecoder(2, 2))(CORNERS, n_samples=n_samples)
        torch.testing.assert_close(discretes, SPINS.unsqueeze(1).expand(-1, n_samples, -1))

    def test_objectives_reach_every_parameter(self):
        # One evaluation of the training objectives: gradients flow through the straight-through
        # discretization to the encoder, to the decoder and to the prior
        encoder, decoder = torch.nn.Linear(2, 2), LinearDecoder(2, 2)
        dvae = DVAE(encoder, decoder, sign_map)
        prior = GRBM((0, 1), [(0, 1)], linear={0: 0.1, 1: -0.2}, quadratic={(0, 1): -1.2})
        prior_samples = SPINS

        latents, discretes, xhat = dvae(CORNERS, n_samples=3)
        reconstruction = torch.nn.functional.mse_loss(xhat, CORNERS.unsqueeze(1).expand_as(xhat))
        kl = pseudo_kl_divergence_loss(discretes, latents, prior_samples, prior)
        (reconstruction + 0.1 * kl).backward()
        for name, parameter in [*dvae.named_parameters(), *prior.named_parameters()]:
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.any(parameter.grad != 0))

        with self.subTest("The maximum mean discrepancy also reaches the encoder"):
            encoder.zero_grad()
            _, discretes, _ = dvae(CORNERS)
            mmd_loss(discretes.flatten(1), prior_samples, GaussianKernel(3)).backward()
            self.assertTrue(torch.any(encoder.weight.grad != 0))

    def test_uniform_prior_is_a_fixed_point_of_uniform_data(self):
        # The corners encode to the four spin strings, which the zero-parameter prior generates
        # uniformly. ExactSolver enumerates every state once, i.e. it samples that prior exactly,
        # so the gradient of the pseudo-KL divergence with respect to the prior vanishes: training
        # the prior on this data leaves it uniform.
        prior = GRBM((0, 1), [(0, 1)], linear={0: 0.0, 1: 0.0}, quadratic={(0, 1): 0.0})
        latents, discretes, _ = DVAE(BitEncoder(), LinearDecoder(2, 2))(CORNERS)
        samples = DimodSampler(prior, ExactSolver()).sample()
        self.assertEqual((4, 2), tuple(samples.shape))
        loss = pseudo_kl_divergence_loss(discretes, latents, samples, prior)
        for grad in torch.autograd.grad(loss, list(prior.parameters())):
            torch.testing.assert_close(grad, torch.zeros_like(grad))


if __name__ == "__main__":
    unittest.main()
