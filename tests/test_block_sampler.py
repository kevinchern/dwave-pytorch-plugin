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

import networkx as nx
import numpy as np
import torch
from dimod import ExactSolver
from dwave.graphs import zephyr_four_color, zephyr_graph
from parameterized import parameterized

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers.block_spin_sampler import BlockSampler
from dwave.plugins.torch.utils import sampleset_to_tensor
from tests.helper_functions import model_to_bqm, set_weights


def total_variation(model: GRBM, samples: torch.Tensor) -> float:
    """Total variation distance between the empirical distribution of ``samples`` and the exact
    Boltzmann distribution of ``model`` at unit inverse temperature."""
    exact = ExactSolver().sample(model_to_bqm(model))
    states = sampleset_to_tensor(model.nodes, exact)
    energies = torch.tensor(np.ascontiguousarray(exact.record.energy), dtype=torch.float32)
    p_exact = torch.softmax(-energies, 0)
    weights = 2 ** torch.arange(model.n_nodes, dtype=torch.float32)
    keys = ((samples + 1) / 2 @ weights).long()
    exact_keys = ((states + 1) / 2 @ weights).long()
    counts = torch.bincount(keys, minlength=2**model.n_nodes).float() / samples.shape[0]
    return 0.5 * (counts[exact_keys] - p_exact).abs().sum().item()


def five_cycle_with_chord() -> GRBM:
    nodes = list("abcde")
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "a"), ("a", "c")]
    return GRBM(
        nodes, edges,
        linear=dict(zip(nodes, [0.3, -0.2, 0.5, 0.1, -0.4])),
        quadratic=dict(zip(edges, [0.8, -0.6, 0.4, -0.9, 0.5, -0.3])),
    )


class TestBlockSampler(unittest.TestCase):
    ZEPHYR = zephyr_graph(1, coordinates=True)
    GRBM_ZEPHYR = GRBM(ZEPHYR.nodes, ZEPHYR.edges)
    CRAYON_ZEPHYR = zephyr_four_color

    BIPARTITE = nx.complete_bipartite_graph(5, 3)
    GRBM_BIPARTITE = GRBM(BIPARTITE.nodes, BIPARTITE.edges)
    def CRAYON_BIPARTITE(b): return b < 5

    GRBM_SINGLE = GRBM([0], [])
    def CRAYON_SINGLE(s): return 0

    GRBM_CRAYON_TEST_CASES = [(GRBM_ZEPHYR, CRAYON_ZEPHYR),
                              (GRBM_BIPARTITE, CRAYON_BIPARTITE),
                              (GRBM_SINGLE, CRAYON_SINGLE)]

    def setUp(self) -> None:
        self.crayon_veqa = lambda v: v == "a"

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_sample(self, grbm, crayon):
        for pac in "Metropolis", "Gibbs":
            schedule = [0.0, 1.0, 2.0]
            bss1 = BlockSampler(grbm, crayon, 10, schedule, pac, seed=1)
            samples = bss1.sample()
            self.assertEqual((10, grbm.n_nodes), tuple(samples.shape))
            self.assertTrue(torch.all(samples.abs() == 1))

            bss2 = BlockSampler(grbm, crayon, 10, [1.0], pac, seed=1)
            for beta in schedule:
                bss2._step(beta, bss2.state)

            self.assertListEqual(bss1.state.tolist(), bss2.state.tolist())
            self.assertListEqual(samples.tolist(), bss1.state.tolist())

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_partition(self, grbm: GRBM, crayon):
        bss = BlockSampler(grbm, crayon, 10, [1.0], seed=5)
        # Check every block is indeed coloured correctly
        for block in bss.partition:
            self.assertEqual(1, len({crayon(grbm.nodes[bidx]) for bidx in block.tolist()}))
        # Check every node has been included exactly once
        indices = [idx for block in bss.partition for idx in block.tolist()]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertSetEqual(set(indices), set(range(grbm.n_nodes)))
        # Blocks are ordered by colour
        colours = [crayon(grbm.nodes[block[0].item()]) for block in bss.partition]
        self.assertListEqual(colours, sorted(colours))

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_automatic_colouring(self, grbm: GRBM, crayon):
        bss = BlockSampler(grbm, None, 3, [1.0])
        colour = torch.empty(grbm.n_nodes, dtype=torch.long)
        for k, block in enumerate(bss.partition):
            colour[block] = k
        self.assertTrue(torch.all(colour[grbm.edge_idx_i] != colour[grbm.edge_idx_j]))
        self.assertEqual((3, grbm.n_nodes), tuple(bss.sample().shape))

    def test_invalid_crayon(self):
        grbm = GRBM([0, 1], [(0, 1)])
        def crayon(n): return 1
        self.assertRaisesRegex(ValueError, "not a valid colouring", BlockSampler, grbm, crayon, 10, [1.0])

    def test_invalid_proposal(self):
        grbm = GRBM([0, 1], [(0, 1)])
        def crayon(n): return n
        self.assertRaisesRegex(ValueError, "Proposal acceptance criterion should be one of",
                               BlockSampler, grbm, crayon, 10, [1.0], "abc")

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_invalid_num_chains(self, grbm, crayon):
        self.assertRaisesRegex(ValueError, "should be a positive integer", BlockSampler, grbm, crayon, 0, [1.0])

    def test_invalid_schedule(self):
        grbm = GRBM([0, 1], [(0, 1)])
        self.assertRaisesRegex(ValueError, "at least one inverse temperature", BlockSampler, grbm, None, 1, [])

    def test_requires_model(self):
        self.assertRaisesRegex(TypeError, "GraphRestrictedBoltzmannMachine", BlockSampler, torch.nn.Linear(2, 2))

    def test_prepare_initial_states(self):
        grbm = GRBM([0, 1, 2], [(0, 1)])
        def crayon(n): return n
        bss = BlockSampler(grbm, crayon, 1, [1.0],)

        with self.subTest("Nonspin initial states."):
            self.assertRaisesRegex(ValueError, "contain nonspin values", bss._prepare_initial_states,
                                   initial_states=torch.tensor([[0, 1, -1]]), num_chains=1)

        with self.subTest("Testing initial states with incorrect shape."):
            self.assertRaisesRegex(ValueError, "Initial states should be of shape", bss._prepare_initial_states,
                                   num_chains=10, initial_states=torch.tensor([[-1, 1, 1, 1, -1]]))

    def test_initial_states_respected(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        initial_states = torch.tensor([[-1, 1], [1, 1], [-1, -1], [1, 1], [-1, 1], [-1, 1], [1, 1]])
        bss = BlockSampler(grbm, self.crayon_veqa, len(initial_states), [1.0], "Metropolis",
                           initial_states, 2)
        self.assertListEqual(bss.state.tolist(), initial_states.tolist())
        self.assertEqual(torch.float32, bss.state.dtype)
        self.assertEqual(7, bss.num_chains)

    def test_properties(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        bss = BlockSampler(grbm, self.crayon_veqa, 3, [0.5, 1], "metropolis", seed=7)
        self.assertEqual((0.5, 1.0), bss.schedule)
        self.assertEqual("Metropolis", bss.proposal_acceptance_criteria)
        self.assertEqual(7, bss.seed)
        self.assertIs(grbm, bss.model)
        self.assertIs(grbm, next(bss.children()))

    def test_gibbs_update(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        sample_size = 1_000_000
        bss = BlockSampler(grbm, self.crayon_veqa, sample_size, [1.0], "Gibbs", seed=2)
        bss.state[:] = 1
        zero = torch.tensor(0.0)
        ones = torch.ones((sample_size, 1))
        bss._gibbs_update(0.0, bss.partition[0], ones*zero, bss.state)
        torch.testing.assert_close(torch.tensor(0.5), bss.state.mean(), atol=1e-3, rtol=1e-3)
        bss._gibbs_update(0.0, bss.partition[1], ones*zero, bss.state)
        torch.testing.assert_close(torch.tensor(0.0), bss.state.mean(), atol=1e-3, rtol=1e-3)

        effective_field = torch.tensor(1.2)
        bss._gibbs_update(1.0, bss.partition[0], effective_field*ones, bss.state)
        bss._gibbs_update(1.0, bss.partition[1], effective_field*ones, bss.state)
        torch.testing.assert_close(
            torch.tanh(-effective_field),
            bss.state.mean(),
            atol=1e-3, rtol=1e-3)

    def test_metropolis_update_average(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        sample_size = 1_000_000
        bss = BlockSampler(grbm, self.crayon_veqa, sample_size, [1.0], "Metropolis", seed=2)
        bss.state[:] = 1
        ones = torch.ones((sample_size, 1))
        effective_field = torch.tensor(1.2)
        for i in range(10):
            bss._metropolis_update(1.0, bss.partition[0], effective_field*ones, bss.state)
            bss._metropolis_update(1.0, bss.partition[1], effective_field*ones, bss.state)
        torch.testing.assert_close(
            torch.tanh(-effective_field),
            bss.state.mean(),
            atol=1e-3, rtol=1e-3)

    def test_metropolis_update_oscillates(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        sample_size = 100
        bss = BlockSampler(grbm, self.crayon_veqa, sample_size, [1.0], "Metropolis", seed=2)
        bss.state[:] = 1
        zero_effective_field = torch.zeros((sample_size, 1))
        bss._metropolis_update(0.0, bss.partition[0], zero_effective_field, bss.state)
        self.assertTrue((bss.state[:, 1] == -1).all())
        bss._metropolis_update(0.0, bss.partition[1], zero_effective_field, bss.state)
        self.assertTrue((bss.state == -1).all())

    @parameterized.expand(["Gibbs", "Metropolis"])
    def test_stationary_distribution(self, pac):
        model = five_cycle_with_chord()
        sampler = BlockSampler(model, None, 100_000, [1.0] * 6, pac, seed=1)
        self.assertEqual(3, len(sampler.partition))
        self.assertLess(total_variation(model, sampler.sample()), 0.02)

    def test_seeds(self):
        model = five_cycle_with_chord()
        with self.subTest("Same seed, same samples"):
            s1 = BlockSampler(model, None, 50, [1.0], seed=3).sample()
            s2 = BlockSampler(model, None, 50, [1.0], seed=3).sample()
            self.assertTrue(torch.equal(s1, s2))
        with self.subTest("Different seeds, different samples"):
            s3 = BlockSampler(model, None, 50, [1.0], seed=4).sample()
            self.assertFalse(torch.equal(s1, s3))
        with self.subTest("No seed uses the global generator"):
            torch.manual_seed(0)
            s4 = BlockSampler(model, None, 50, [1.0]).sample()
            torch.manual_seed(0)
            s5 = BlockSampler(model, None, 50, [1.0]).sample()
            s6 = BlockSampler(model, None, 50, [1.0]).sample()
            self.assertTrue(torch.equal(s4, s5))
            self.assertFalse(torch.equal(s5, s6))

    def test_sample_conditional_three_blocks(self):
        # Triangle graph
        grbm = GRBM(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        set_weights(grbm, [1e10] * 3, [0.0] * 3)

        def crayon(n):
            return {"a": 0, "b": 1, "c": 2}[n]

        sampler = BlockSampler(grbm, crayon, 2, [1.0], "Gibbs", seed=123)
        chains = sampler.state.clone()

        # Row 0 unclamps block 0, row 1 unclamps block 2
        x = torch.tensor([
            [float("nan"), 1.0, 1.0],
            [1.0, 1.0, float("nan")]
        ])
        result = sampler.sample(x)
        self.assertEqual((2, 1, 3), tuple(result.shape))
        # Ensure clamped spins remain unchanged and unclamped spins will be -1
        expected = torch.tensor([[[-1.0, 1.0, 1.0]], [[1.0, 1.0, -1.0]]])
        torch.testing.assert_close(result, expected)

        with self.subTest("Persistent chains are untouched"):
            self.assertTrue(torch.equal(chains, sampler.state))

        with self.subTest("Several samples per row and several free blocks per row"):
            x = torch.tensor([[float("nan"), float("nan"), 1.0]])
            result = sampler.sample(x, num_samples=5)
            self.assertEqual((1, 5, 3), tuple(result.shape))
            torch.testing.assert_close(result, torch.tensor([[[-1.0, -1.0, 1.0]] * 5]))

        with self.subTest("Batch dimensions are preserved; batch size need not match chains"):
            x = torch.full((4, 3, 3), float("nan"))
            x[..., 1] = 1.0
            result = sampler.sample(x, num_samples=2)
            self.assertEqual((4, 3, 2, 3), tuple(result.shape))
            self.assertTrue(torch.all(result[..., 1] == 1.0))
            self.assertTrue(torch.all(result[..., [0, 2]] == -1.0))

        with self.subTest("Fully clamped rows are returned unchanged"):
            x = torch.tensor([[1.0, -1.0, 1.0]])
            torch.testing.assert_close(sampler.sample(x), x.unsqueeze(1))

        with self.subTest("Invalid inputs"):
            with self.assertRaisesRegex(ValueError, "x must have shape"):
                sampler.sample(torch.tensor([[float("nan"), 1.0]]))
            with self.assertRaisesRegex(ValueError, "only ±1 or NaN"):
                sampler.sample(torch.tensor([[0.0, 1.0, float("nan")]]))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                sampler.sample(x, num_samples=0)

    def test_sample_conditional_is_exact_for_single_block(self):
        model = five_cycle_with_chord()
        sampler = BlockSampler(model, None, 1, [1.0], seed=9)
        # Clamp everything but block 0 and compare with the exact conditional means
        block = sampler.partition[0]
        x = torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0]])
        x[:, block] = torch.nan
        samples = sampler.sample(x, num_samples=200_000)
        expected = -torch.tanh(model.effective_field(x, block))
        torch.testing.assert_close(samples[0, :, block].mean(0, keepdim=True), expected,
                                   atol=5e-3, rtol=0)

    def test_device(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        bss = BlockSampler(grbm, self.crayon_veqa, 10, [1.0], "Gibbs", seed=2)
        result = bss.to("meta")
        self.assertIs(bss, result)
        self.assertEqual("meta", bss.model.linear.device.type)
        self.assertEqual("meta", bss.model.quadratic.device.type)
        self.assertEqual("meta", bss.state.device.type)
        for block in bss.partition:
            self.assertEqual("meta", block.device.type)

    def test_state_dict(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        bss = BlockSampler(grbm, self.crayon_veqa, 4, [1.0], seed=2)
        state_dict = bss.state_dict()
        self.assertIn("state", state_dict)
        self.assertIn("model.linear", state_dict)
        self.assertIn("model.quadratic", state_dict)
        self.assertNotIn("_block_idx", state_dict, "blocks are derived from the colouring")
        self.assertEqual(0, len(list(bss.parameters())) - len(list(grbm.parameters())))

        with self.subTest("The chains are restored from the state dict"):
            other = BlockSampler(grbm, self.crayon_veqa, 4, [1.0], seed=3)
            other.load_state_dict(state_dict)
            self.assertTrue(torch.equal(bss.state, other.state))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda(self):
        model = five_cycle_with_chord()
        sampler = BlockSampler(model, None, 20_000, [1.0] * 6, seed=1).cuda()
        self.assertTrue(sampler.model.linear.is_cuda and sampler.state.is_cuda)
        samples = sampler.sample()
        self.assertTrue(samples.is_cuda)
        self.assertLess(total_variation(model.cpu(), samples.cpu()), 0.05)
        sampler = sampler.cuda()
        x = torch.tensor([[float("nan"), 1.0, -1.0, 1.0, float("nan")]])
        conditional = sampler.sample(x, num_samples=3)
        self.assertTrue(conditional.is_cuda)
        self.assertEqual((1, 3, 5), tuple(conditional.shape))
        self.assertTrue(torch.all(conditional[..., 1:4].cpu() == x[:, 1:4]))


if __name__ == "__main__":
    unittest.main()
