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
from dwave.plugins.torch.utils import GraphIndex, sampleset_to_tensor
from tests.helper_functions import (RecordedBernoulli, constant_randspin, exact_update_probabilities,
                                    model_to_bqm, replay_sweep, set_weights)


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


# Four spin configurations of the five-node model above, used as sweep inputs
SPINS = torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0],
                      [-1.0, -1.0, 1.0, -1.0, 1.0],
                      [1.0, 1.0, -1.0, -1.0, -1.0],
                      [-1.0, 1.0, 1.0, 1.0, 1.0]])


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

    # ------------------------------------------------------------------ construction ------------

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_partition(self, grbm: GRBM, crayon):
        bss = BlockSampler(grbm, crayon, 10, [1.0])
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
        bss = BlockSampler(grbm, crayon, 1, [1.0])

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

    # ------------------------------------------------------------------ updates -----------------
    # The updates are tested exactly: the Bernoulli draws are replaced by a threshold rule and the
    # probabilities the updates draw with are compared with the ones computed from the energies.

    def test_gibbs_update(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        bss = BlockSampler(grbm, self.crayon_veqa, 4, [1.0], "Gibbs")
        block = bss.partition[1]  # node a
        x = torch.ones(4, 2)
        field = torch.tensor([[1.2], [-0.5], [0.0], [3.0]])
        with RecordedBernoulli() as bernoulli:
            bss._gibbs_update(0.5, block, field, x)

        with self.subTest("A spin is +1 with probability sigmoid(-2 beta h)"):
            self.assertEqual(1, len(bernoulli.probabilities))
            torch.testing.assert_close(bernoulli.probabilities[0], torch.sigmoid(-2 * 0.5 * field))

        with self.subTest("Draws become spins and only the block changes"):
            torch.testing.assert_close(x[:, block], torch.tensor([[-1.0], [1.0], [-1.0], [-1.0]]))
            self.assertTrue(torch.all(x[:, bss.partition[0]] == 1))

    def test_metropolis_update(self):
        grbm = GRBM(list("ab"), [["a", "b"]])
        bss = BlockSampler(grbm, self.crayon_veqa, 4, [1.0], "Metropolis")
        block = bss.partition[1]  # node a
        x = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
        field = torch.tensor([[1.2], [1.2], [-0.5], [-0.5]])
        with RecordedBernoulli() as bernoulli:
            bss._metropolis_update(0.5, block, field, x)

        with self.subTest("A flip is accepted with probability min(1, exp(-beta * delta energy))"):
            delta_energy = -2 * torch.tensor([[1.0], [-1.0], [1.0], [-1.0]]) * field
            torch.testing.assert_close(
                bernoulli.probabilities[0], torch.exp(-0.5 * delta_energy).clamp(max=1.0)
            )

        with self.subTest("Accepted proposals flip the spins and only the block changes"):
            # Rows: energy lowered (flip), raised with p = exp(-1.2) (kept), raised with
            # p = exp(-0.5) (flip), lowered (flip)
            torch.testing.assert_close(x[:, block], torch.tensor([[-1.0], [-1.0], [-1.0], [1.0]]))
            self.assertTrue(torch.all(x[:, bss.partition[0]] == 1))

    def test_metropolis_update_oscillates(self):
        # At zero inverse temperature every proposal is accepted
        grbm = GRBM(list("ab"), [["a", "b"]])
        sample_size = 100
        bss = BlockSampler(grbm, self.crayon_veqa, sample_size, [1.0], "Metropolis")
        bss.state[:] = 1
        zero_effective_field = torch.zeros((sample_size, 1))
        bss._metropolis_update(0.0, bss.partition[0], zero_effective_field, bss.state)
        self.assertTrue((bss.state[:, 1] == -1).all())
        bss._metropolis_update(0.0, bss.partition[1], zero_effective_field, bss.state)
        self.assertTrue((bss.state == -1).all())

    @parameterized.expand(["Gibbs", "Metropolis"])
    def test_step_uses_exact_conditional_probabilities(self, criterion):
        # Every block update draws with the exact conditional (Gibbs) or acceptance (Metropolis)
        # probabilities of the state at that point of the sweep, here computed from the energies
        model = five_cycle_with_chord()
        sampler = BlockSampler(model, None, 1, [1.0], criterion)
        linear, quadratic = model.linear.detach(), model.quadratic.detach()
        coupling = model.symmetric_coupling().detach()

        x = SPINS.clone()
        with RecordedBernoulli() as bernoulli:
            sampler._step(0.7, x, linear, coupling)
        self.assertEqual(len(sampler.partition), len(bernoulli.probabilities))
        expected = replay_sweep(model, SPINS, linear, quadratic, sampler.partition, 0.7, criterion,
                                bernoulli.probabilities)
        torch.testing.assert_close(x, expected)

        with self.subTest("Clamped spins are restored after every block"):
            clamp_mask = torch.tensor([True, False, True, False, False]).expand(4, 5)
            clamped_values = -SPINS
            start = torch.where(clamp_mask, clamped_values, SPINS)
            x = start.clone()
            with RecordedBernoulli() as bernoulli:
                sampler._step(0.7, x, linear, coupling, clamp_mask, clamped_values)
            expected = replay_sweep(model, start, linear, quadratic, sampler.partition, 0.7, criterion,
                                    bernoulli.probabilities, clamp_mask, clamped_values)
            torch.testing.assert_close(x, expected)
            torch.testing.assert_close(x[clamp_mask], clamped_values[clamp_mask])

    # ------------------------------------------------------------------ sampling ----------------

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_sample(self, grbm, crayon):
        # Sampling is one sweep per inverse temperature of the schedule on the persistent chains
        for pac in "Metropolis", "Gibbs":
            schedule = [0.0, 1.0, 2.0]
            bss1 = BlockSampler(grbm, crayon, 10, schedule, pac, seed=1)
            samples = bss1.sample()
            self.assertEqual((10, grbm.n_nodes), tuple(samples.shape))
            self.assertTrue(torch.all(samples.abs() == 1))

            bss2 = BlockSampler(grbm, crayon, 10, [1.0], pac, seed=1)
            for beta in schedule:
                bss2._step(beta, bss2.state, grbm.linear, grbm.symmetric_coupling())

            self.assertListEqual(bss1.state.tolist(), bss2.state.tolist())
            self.assertListEqual(samples.tolist(), bss1.state.tolist())

    def test_sample_conditional_three_blocks(self):
        # Triangle graph
        grbm = GRBM(["a", "b", "c"], [["a", "b"], ["b", "c"], ["a", "c"]])
        set_weights(grbm, [1e10] * 3, [0.0] * 3)

        def crayon(n):
            return {"a": 0, "b": 1, "c": 2}[n]

        sampler = BlockSampler(grbm, crayon, 2, [1.0], "Gibbs")
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

    def test_sample_conditional_draws_exact_conditionals(self):
        # The unobserved spins of a row belong to a single block, so one Gibbs sweep draws them
        # from their exact conditional distribution given the observed spins, whatever their
        # random initialization
        model = five_cycle_with_chord()
        sampler = BlockSampler(model, None, 1, [1.0])
        block = sampler.partition[0]
        x = torch.tensor([[1.0, -1.0, 1.0, 1.0, -1.0]])
        x[:, block] = torch.nan
        with RecordedBernoulli() as bernoulli:
            samples = sampler.sample(x, num_samples=3)

        filled = torch.nan_to_num(x, nan=1.0).expand(3, -1)
        expected = exact_update_probabilities(
            model, filled, model.linear.detach(), model.quadratic.detach(), block, 1.0, "Gibbs"
        )
        torch.testing.assert_close(bernoulli.probabilities[0], expected)

        with self.subTest("The draws are the free spins; the observed spins are kept"):
            observed = ~torch.isnan(x[0])
            torch.testing.assert_close(
                samples[0][:, block], 2 * (bernoulli.probabilities[0] > 0.5).float() - 1
            )
            torch.testing.assert_close(samples[0][:, observed], x[0, observed].expand(3, -1))

        with self.subTest("The model's conditional expectations agree"):
            torch.testing.assert_close(
                expected, ((1 + model.conditional_expectation(x)[:, block]) / 2).expand(3, -1)
            )

    def test_sample_biases(self):
        model_a = five_cycle_with_chord()
        model_b = five_cycle_with_chord()
        set_weights(model_b, [-0.4, 0.3, 0.1, -0.5, 0.2], [0.5, 0.9, -0.4, 0.6, -0.8, 0.3])
        linear = torch.stack([model_a.linear, model_b.linear]).detach()
        quadratic = torch.stack([model_a.quadratic, model_b.quadratic]).detach()

        # A sampler bound to the bare graph samples any biases on it. With the initial spins and
        # the Bernoulli draws under control, the sweeps of the schedule replay exactly, model by
        # model, from the energies of the models.
        graph = GraphIndex(model_a.nodes, model_a.edges)
        sampler = BlockSampler(graph, schedule=[0.5, 1.0])
        with constant_randspin(1.0), RecordedBernoulli() as bernoulli:
            samples = sampler.sample_biases(linear, quadratic, num_samples=3)
        self.assertEqual((2, 3, 5), tuple(samples.shape))
        n_blocks = len(sampler.partition)
        self.assertEqual(2 * n_blocks, len(bernoulli.probabilities))
        state = torch.ones(2, 3, 5)
        for sweep, beta in enumerate(sampler.schedule):
            recorded = bernoulli.probabilities[sweep * n_blocks:(sweep + 1) * n_blocks]
            state = replay_sweep(graph, state, linear, quadratic, sampler.partition, beta, "Gibbs",
                                 recorded)
        torch.testing.assert_close(samples, state)

        with self.subTest("Unbatched biases give (num_samples, n_nodes)"):
            samples = sampler.sample_biases(model_a.linear, model_a.quadratic, num_samples=7)
            self.assertEqual((7, 5), tuple(samples.shape))
            self.assertTrue(torch.all(samples.abs() == 1))

        with self.subTest("Arbitrary batch dimensions"):
            samples = sampler.sample_biases(linear.reshape(2, 1, 5), quadratic.reshape(2, 1, 5, 5), 3)
            self.assertEqual((2, 1, 3, 5), tuple(samples.shape))

        with self.subTest("Invalid inputs"):
            with self.assertRaisesRegex(ValueError, "linear must have shape"):
                sampler.sample_biases(torch.zeros(2, 4), quadratic)
            with self.assertRaisesRegex(ValueError, "quadratic must have shape"):
                sampler.sample_biases(linear, torch.zeros(2, 5, 4))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                sampler.sample_biases(linear, quadratic, num_samples=0)

        with self.subTest("A graph-bound sampler has no parameters to sample"):
            with self.assertRaisesRegex(TypeError, "GraphRestrictedBoltzmannMachine"):
                sampler.sample()
            with self.assertRaisesRegex(TypeError, "GraphRestrictedBoltzmannMachine"):
                sampler.complete(torch.ones(1, 5))

        with self.subTest("A model-bound sampler samples other biases without touching its chains"):
            sampler = BlockSampler(model_a, schedule=[1.0])
            chains = sampler.state.clone()
            samples = sampler.sample_biases(model_b.linear, model_b.quadratic, num_samples=7)
            self.assertEqual((7, 5), tuple(samples.shape))
            self.assertTrue(torch.equal(chains, sampler.state))

    @parameterized.expand(["Gibbs", "Metropolis"])
    def test_stationary_distribution(self, pac):
        # Statistical integration test of whole sweeps: with 100k chains the expected total
        # variation to the exact distribution is below 0.01 with a standard deviation of about
        # 0.001, so the tolerance holds for any seed; the seed only makes a failure reproducible.
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

    # ------------------------------------------------------------------ module behaviour --------

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
