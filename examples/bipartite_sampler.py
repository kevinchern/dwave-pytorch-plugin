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
import torch
from torch.optim import SGD

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers import BipartiteGibbsSampler


def run(device: str = "cpu"):
    """Run an example of fitting a restricted Boltzmann machine (a bipartite graph-restricted
    Boltzmann machine) with a BipartiteGibbsSampler to synthetic data generated uniformly at random.

    Args:
        device (str): Device on which to train, e.g. "cpu" or "cuda".
    """
    n_visible, n_hidden = 50, 20
    visible_nodes = [f"v{i}" for i in range(n_visible)]
    hidden_nodes = [f"h{j}" for j in range(n_hidden)]
    edges = [(v, h) for v in visible_nodes for h in hidden_nodes]
    grbm = GRBM(visible_nodes + hidden_nodes, edges, hidden_nodes=hidden_nodes)

    # The sampler holds the model, so moving the sampler moves the model as well. Its persistent
    # Markov chains provide the negative phase (persistent contrastive divergence).
    num_chains = batch_size = 100
    sampler = BipartiteGibbsSampler(grbm, num_chains=num_chains, schedule=[1.0], seed=123).to(device)

    n_iterations = 3
    X = 1 - 2.0 * torch.randint(0, 2, (n_iterations, batch_size, n_visible), device=device)

    optimizer = SGD(grbm.parameters(), lr=0.1)

    for iteration, x in enumerate(X):
        # Negative phase: advance the persistent chains
        s_model = sampler.sample()

        optimizer.zero_grad()

        # Positive phase: hidden units of a restricted Boltzmann machine are conditionally
        # independent given the visible units, so their expectations are exact ("exact-disc").
        # Alternatively, sample them: grbm.quasi_objective(x, s_model, "sampling", sampler=sampler)
        loss = grbm.quasi_objective(x, s_model, kind="exact-disc")

        loss.backward()
        optimizer.step()

        avg_grad = (grbm.linear.grad.abs().mean()
                    + grbm.quadratic.grad[grbm.adjacency].abs().mean()) / 2
        print(f"Iteration {iteration:3d} | Average |gradient|: {avg_grad.item():.2f}")
    print("\nTraining finished.")


if __name__ == "__main__":
    torch.manual_seed(123)
    run()
