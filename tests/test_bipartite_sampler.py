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

import unittest

import torch

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers.bipartite_sampler import BipartiteGibbsSampler
from dwave.plugins.torch.samplers.block_spin_sampler import BlockSampler


def set_weights(bm: GRBM, linear, quadratic) -> None:
    with torch.no_grad():
        bm.linear.copy_(torch.as_tensor(linear, dtype=bm.linear.dtype))
        bm.quadratic[bm.edge_idx_i, bm.edge_idx_j] = torch.as_tensor(
            quadratic, dtype=bm.quadratic.dtype
        )


def rbm() -> GRBM:
    nodes = ["v1", "v2", "h1", "h2"]
    edges = [["v1", "h1"], ["v1", "h2"], ["v2", "h1"], ["v2", "h2"]]
    return GRBM(nodes, edges, hidden_nodes=["h1", "h2"])


class TestBipartiteGibbsSampler(unittest.TestCase):

    def test_is_block_sampler(self):
        grbm = rbm()
        sampler = BipartiteGibbsSampler(grbm, num_chains=3, schedule=[1.0])
        self.assertIsInstance(sampler, BlockSampler)
        self.assertEqual("Gibbs", sampler.proposal_acceptance_criteria)
        self.assertEqual(2, len(sampler.partition))
        self.assertListEqual(sampler.partition[0].tolist(), grbm.visible_idx.tolist())
        self.assertListEqual(sampler.partition[1].tolist(), grbm.hidden_idx.tolist())

    def test_device(self):
        grbm = GRBM(["v1", "h1"], [["v1", "h1"]], hidden_nodes=["h1"])
        sampler = BipartiteGibbsSampler(grbm, num_chains=10, schedule=[1.0], seed=2).to("meta")
        self.assertEqual("meta", sampler.model.linear.device.type)
        self.assertEqual("meta", sampler.model.quadratic.device.type)
        self.assertEqual("meta", sampler.state.device.type)

    def test_visible_visible_connection(self):
        grbm = GRBM(["v1", "v2", "h1"], [["v1", "h1"], ["v1", "v2"]], hidden_nodes=["h1"])
        with self.assertRaisesRegex(ValueError, r"requires a bipartite model.*\('v1', 'v2'\)"):
            BipartiteGibbsSampler(grbm, num_chains=2, schedule=[1.0])

    def test_hidden_hidden_connection(self):
        grbm = GRBM(["v1", "h1", "h2"], [["v1", "h1"], ["h1", "h2"]], hidden_nodes=["h1", "h2"])
        with self.assertRaisesRegex(ValueError, r"requires a bipartite model.*\('h1', 'h2'\)"):
            BipartiteGibbsSampler(grbm, num_chains=2, schedule=[1.0])

    def test_prepare_initial_states(self):
        sampler = BipartiteGibbsSampler(rbm(), num_chains=2, schedule=[1.0])
        initial_states = torch.tensor([[-1, 1, 1, -1], [1, -1, -1, 1]])
        out = sampler._prepare_initial_states(num_chains=2, initial_states=initial_states)
        self.assertEqual(out.shape, (2, 4))
        torch.testing.assert_close(out, initial_states.float())

        with self.subTest("Non-spin initial states."):
            self.assertRaisesRegex(ValueError, "contain nonspin values", sampler._prepare_initial_states,
                                   initial_states=torch.tensor([[0, 1, -1, 1]]), num_chains=1)
        with self.subTest("Testing initial states with incorrect shape."):
            self.assertRaisesRegex(ValueError, "Initial states should be of shape", sampler._prepare_initial_states,
                                   num_chains=2, initial_states=torch.tensor([[-1, 1, 1, 1, -1]]))

    def test_compute_effective_field_bipartite(self):
        grbm = rbm()
        set_weights(grbm, [0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.7, 0.6])
        sampler = BipartiteGibbsSampler(grbm, num_chains=1, schedule=[1.0])
        sampler.state[:] = torch.tensor([[1., -1., 1., -1.]])

        expected_visible_field = torch.tensor([[0.1 + 0.5*1 + 0.2*(-1),  # v1
                                                -0.2 + (-0.7)*1 + 0.6*(-1)]])  # v2
        expected_hidden_field = torch.tensor([[0.3 + 0.5*1 + (-0.7)*(-1),  # h1
                                               -0.4 + 0.2*1 + 0.6*(-1)]])   # h2
        torch.testing.assert_close(expected_visible_field,
                                   sampler._compute_effective_field(grbm.visible_idx))
        torch.testing.assert_close(expected_hidden_field,
                                   sampler._compute_effective_field(grbm.hidden_idx))

    def test_gibbs_update(self):
        grbm = rbm()
        sample_size = 1_000_000
        sampler = BipartiteGibbsSampler(grbm, num_chains=sample_size, schedule=[1.0], seed=42)
        ones = torch.ones((sample_size, 1))
        visible_block, hidden_block = grbm.visible_idx, grbm.hidden_idx

        # At beta = 0 and zero effective field the updated spins are uniform, the others stay +1
        with self.subTest("visible block Gibbs update"):
            sampler.state[:] = 1.0
            sampler._gibbs_update(0.0, visible_block, ones * 0.0)
            torch.testing.assert_close(torch.tensor(0.5), sampler.state.mean(), atol=1e-3, rtol=1e-3)

        with self.subTest("hidden block Gibbs update"):
            sampler.state[:] = 1.0
            sampler._gibbs_update(0.0, hidden_block, ones * 0.0)
            torch.testing.assert_close(torch.tensor(0.5), sampler.state.mean(), atol=1e-3, rtol=1e-3)

        with self.subTest("Gibbs update with nonzero effective field"):
            effective_field = torch.tensor(1.2)
            sampler.state[:] = 1.0
            sampler._gibbs_update(1.0, visible_block, effective_field * ones)
            sampler._gibbs_update(1.0, hidden_block, effective_field * ones)
            torch.testing.assert_close(torch.tanh(-effective_field), sampler.state.mean(),
                                       atol=1e-3, rtol=1e-3)

    def test_sample(self):
        grbm = rbm()
        set_weights(grbm, [0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.7, 0.6])
        sampler1 = BipartiteGibbsSampler(grbm, num_chains=5, schedule=[1.0, 2.0], seed=42)
        sampler2 = BipartiteGibbsSampler(grbm, num_chains=5, schedule=[1.0, 2.0], seed=42)

        samples = sampler1.sample()
        for beta in sampler2.schedule:
            sampler2._step(beta)
        self.assertListEqual(samples.tolist(), sampler2.state.tolist())

    def test_sample_conditional(self):
        with self.subTest("clamp visible -> hidden becomes deterministic"):
            grbm = rbm()
            set_weights(grbm, [1e10] * 4, [0.0] * 4)
            sampler = BipartiteGibbsSampler(grbm, num_chains=3, schedule=[1.0, 2.0], seed=123)
            chains = sampler.state.clone()

            x = grbm.pad_visible(torch.tensor([[1., -1.], [1., 1.], [-1., -1.]]))
            result = sampler.sample(x)
            self.assertEqual((3, 1, 4), tuple(result.shape))
            torch.testing.assert_close(result[:, 0, grbm.visible_idx], x[:, grbm.visible_idx])
            torch.testing.assert_close(result[:, 0, grbm.hidden_idx], -torch.ones(3, 2))
            self.assertTrue(torch.equal(chains, sampler.state))

        with self.subTest("clamp hidden -> visible becomes deterministic"):
            grbm = rbm()
            set_weights(grbm, [1e6] * 4, [0.0] * 4)
            sampler = BipartiteGibbsSampler(grbm, num_chains=3, schedule=[1.0], seed=123)

            x = torch.full((3, 4), float("nan"))
            x[:, grbm.hidden_idx] = torch.tensor([[1., -1.], [-1., 1.], [1., 1.]])
            result = sampler.sample(x, num_samples=2)
            self.assertEqual((3, 2, 4), tuple(result.shape))
            torch.testing.assert_close(result[:, :, grbm.hidden_idx],
                                       x[:, None, grbm.hidden_idx].expand(3, 2, 2))
            torch.testing.assert_close(result[:, :, grbm.visible_idx], -torch.ones(3, 2, 2))

    def test_sample_conditional_matches_exact_conditional(self):
        grbm = rbm()
        set_weights(grbm, [0.1, -0.2, 0.3, -0.4], [0.5, 0.2, -0.7, 0.6])
        sampler = BipartiteGibbsSampler(grbm, num_chains=1, schedule=[1.0], seed=0)
        x = grbm.pad_visible(torch.tensor([[1.0, -1.0]]))
        samples = sampler.sample(x, num_samples=200_000)
        expected = -torch.tanh(grbm.effective_field(x, grbm.hidden_idx))
        torch.testing.assert_close(samples[0, :, grbm.hidden_idx].mean(0, keepdim=True), expected,
                                   atol=5e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
