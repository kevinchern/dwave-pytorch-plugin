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
"""A discrete variational autoencoder with a Boltzmann machine prior.

The autoencoder is an encoder that outputs one logit per latent spin, the straight-through
Gumbel-softmax discretisation ``gumbel_spins`` and a decoder that reads the spins. Its prior over
the latent spins is a graph-restricted Boltzmann machine sampled by a block-Gibbs sampler with
persistent chains. The objective is the reconstruction error plus the pseudo Kullback-Leibler
divergence between the encoder's distribution over the spins and the prior, whose gradient trains
the encoder and the prior together (see ``pseudo_kl_divergence_loss``).
"""
import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.nn.functional import gumbel_spins, pseudo_kl_divergence_loss
from dwave.plugins.torch.samplers import BlockSampler


class DiscreteVariationalAutoencoder(nn.Module):
    """Encoder -> spins -> decoder, with ``n_samples`` spin configurations per data point.

    Args:
        encoder (nn.Module): Maps data of shape (batch_size, ...) to logits of shape
            (batch_size, n_latent), one logit per latent spin.
        decoder (nn.Module): Maps spins of shape (batch_size, n_samples, n_latent) to
            reconstructions of shape (batch_size, n_samples, ...).
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor, n_samples: int = 1):
        logits = self.encoder(x)
        # Exact ±1 spins in the forward pass, gradients of the softmax relaxation in the backward pass
        spins = gumbel_spins(logits, n_samples)
        return logits, spins, self.decoder(spins)


def run(device: str = "cpu", n_iterations: int = 300, batch_size: int = 64, kl_weight: float = 0.1):
    """Fit a discrete variational autoencoder with a Boltzmann machine prior to noisy binary
    prototypes.

    Args:
        device (str): Device on which to train, e.g. "cpu" or "cuda".
        n_iterations (int): Number of training iterations.
        batch_size (int): Number of data points per iteration, and of persistent Markov chains.
        kl_weight (float): Weight of the pseudo Kullback-Leibler divergence in the objective.
    """
    generator = torch.Generator().manual_seed(0)
    n_features = 12
    # Data: four binary prototypes, each bit flipped independently with probability 0.1
    prototypes = torch.randint(0, 2, (4, n_features), generator=generator).float().to(device)

    def sample_data() -> torch.Tensor:
        which = torch.randint(0, 4, (batch_size,), generator=generator)
        flips = (torch.rand(batch_size, n_features, generator=generator) < 0.1).float().to(device)
        return (prototypes[which] + flips) % 2

    # Prior: a Boltzmann machine on a 4x4 grid, one latent spin per node. The encoder's k-th logit
    # is the spin of prior.nodes[k]. Persistent block-Gibbs chains provide the negative phase.
    grid = nx.grid_2d_graph(4, 4)
    prior = GRBM(grid.nodes, grid.edges)
    sampler = BlockSampler(prior, num_chains=batch_size, seed=0).to(device)
    n_latent = prior.n_nodes

    dvae = DiscreteVariationalAutoencoder(
        encoder=nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, n_latent)),
        decoder=nn.Sequential(nn.Linear(n_latent, 32), nn.ReLU(), nn.Linear(32, n_features)),
    ).to(device)
    optimizer = torch.optim.Adam([*dvae.parameters(), *prior.parameters()], lr=1e-2)

    for iteration in range(1, n_iterations + 1):
        x = sample_data()
        logits, spins, reconstruction = dvae(x, n_samples=1)
        # Reconstruction error of the decoded spins ...
        target = x.unsqueeze(1).expand_as(reconstruction)
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstruction, target)
        # ... plus the pseudo-KL divergence to the prior: its positive phase is the encoder's spins,
        # its negative phase one sweep of the prior's persistent chains
        kl = pseudo_kl_divergence_loss(spins, logits, sampler.sample(), prior)
        loss = reconstruction_loss + kl_weight * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % 50 == 0:
            print(f"Iteration {iteration:4d} | reconstruction {reconstruction_loss.item():.3f} | "
                  f"pseudo-KL (up to a constant) {kl.item():8.3f}")

    # The prior has learned the latent configurations the encoder uses, so decoding its samples
    # yields data-like patterns: compare them with the nearest prototype
    with torch.no_grad():
        decoded = torch.sigmoid(dvae.decoder(sampler.sample().unsqueeze(1))).squeeze(1).round()
        mismatch = (decoded.unsqueeze(1) != prototypes).float().mean(-1).min(-1).values
    print(f"\nDecoded prior samples differ from their nearest prototype in "
          f"{100 * mismatch.mean().item():.1f}% of the bits")


if __name__ == "__main__":
    torch.manual_seed(0)
    run()
