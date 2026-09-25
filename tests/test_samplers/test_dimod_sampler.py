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
from dimod import SPIN, ExactSolver, IdentitySampler, SampleSet, TrackingComposite

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers.dimod_sampler import DimodSampler
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSampler


def set_weights(bm: GRBM, linear, quadratic) -> None:
    with torch.no_grad():
        bm.linear.copy_(torch.as_tensor(linear, dtype=bm.linear.dtype))
        bm.quadratic[bm.edge_idx_i, bm.edge_idx_j] = torch.as_tensor(
            quadratic, dtype=bm.quadratic.dtype
        )


class TestDimodSampler(unittest.TestCase):
    def setUp(self) -> None:
        # Create a triangle graph with an additional dangling vertex
        #       a
        #     / | \
        #    b--c  d
        # Note the node order is deliberately "dbac" in order to test variable orderings
        self.nodes = list("dbac")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        self.n = 4

        self.bm = GRBM(self.nodes, self.edges)
        set_weights(self.bm, [0.0, 1, 2, 3], [1, 2, 3, 6])

    def test_properties(self):
        sampler = DimodSampler(self.bm, IdentitySampler(), prefactor=2, linear_range=[-1, 1],
                               sample_kwargs=dict(num_reads=3))
        self.assertIs(self.bm, sampler.model)
        self.assertIsInstance(sampler.sampler, IdentitySampler)
        self.assertEqual(2.0, sampler.prefactor)
        self.assertEqual((-1, 1), sampler.linear_range)
        self.assertIsNone(sampler.quadratic_range)
        self.assertDictEqual(dict(num_reads=3), sampler.sample_kwargs)

    def test_to_ising(self):
        set_weights(self.bm, [-3, 0, 1, 3.0], [-1, 1, 2.0, 0])

        with self.subTest("Unscaled and unclipped"):
            h, J = DimodSampler(self.bm, IdentitySampler()).to_ising()
            self.assertDictEqual(h, {"d": -3.0, "b": 0.0, "a": 1.0, "c": 3.0})
            self.assertDictEqual(
                J, {("a", "b"): -1.0, ("a", "c"): 1.0, ("a", "d"): 2.0, ("b", "c"): 0.0}
            )

        with self.subTest("Scaled by the prefactor, then clipped"):
            sampler = DimodSampler(self.bm, IdentitySampler(), prefactor=2,
                                   linear_range=(-1, 5), quadratic_range=(-0.5, 3))
            h, J = sampler.to_ising()
            self.assertDictEqual(h, {"d": -1.0, "b": 0.0, "a": 2.0, "c": 5.0})
            self.assertDictEqual(
                J, {("a", "b"): -0.5, ("a", "c"): 2.0, ("a", "d"): 3.0, ("b", "c"): 0.0}
            )

    def test_sample(self):
        grbm = GRBM(list("abcd"), [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")])

        with self.subTest("Spins should be identical to input."):
            initial_states = [[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [-1, -1, 1, -1]]
            sampler = DimodSampler(grbm, IdentitySampler(),
                                   sample_kwargs=dict(initial_states=(initial_states, "abcd")))
            spins = sampler.sample()
            self.assertIsInstance(spins, torch.Tensor)
            self.assertTupleEqual((3, 4), tuple(spins.shape))
            self.assertListEqual(initial_states, spins.tolist())

        with self.subTest("Prefactor should scale weights up."):
            set_weights(grbm, [1.0] * 4, [-1.0] * 4)
            prefactor = 12345
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker, prefactor=prefactor)
            sampler.sample()
            self.assertDictEqual(tracker.input['h'], dict(zip(grbm.nodes, [prefactor]*4)))
            self.assertDictEqual(tracker.input['J'], dict(zip(grbm.edges, [-prefactor]*4)))

        with self.subTest("Linear weights should be clipped to be 0."):
            set_weights(grbm, [-2, -0.002, 0.002, 3], [-1.0] * 4)
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker, prefactor=100, linear_range=[0, 0])
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['h'].values())),
                torch.tensor([0, 0, 0, 0.0])
            )

        with self.subTest("Linear weights should be clipped to be within range."):
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker, prefactor=100, linear_range=[-1, 1])
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['h'].values())),
                torch.tensor([-1, -0.2, 0.2, 1])
            )

        with self.subTest("Quadratic weights should be clipped to be within range."):
            set_weights(grbm, [0.0] * 4, [-2, -0.002, 0.002, 3])
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker, prefactor=100, quadratic_range=[-1, 1])
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['J'].values())),
                torch.tensor([-1, -0.2, 0.2, 1])
            )

        with self.subTest("Quadratic weights should be clipped to be 0."):
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker, prefactor=100, quadratic_range=[0, 0])
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['J'].values())),
                torch.tensor([0, 0, 0, 0.0])
            )

    def test_aggregated_samples_are_spread(self):
        class AggregatingSampler:
            def sample_ising(self, h, J, **kwargs):
                return SampleSet.from_samples(
                    ([[1, 1, 1, 1], [-1, -1, -1, -1]], list(h)), vartype=SPIN, energy=[0, 0],
                    num_occurrences=[3, 1],
                )

            def sample(self, bqm, **kwargs):
                return SampleSet.from_samples(
                    ([[1] * bqm.num_variables], list(bqm.variables)), vartype=SPIN, energy=[0],
                    num_occurrences=[4],
                )

        sampler = DimodSampler(self.bm, AggregatingSampler())
        spins = sampler.sample()
        self.assertEqual((4, 4), tuple(spins.shape))
        self.assertEqual(3, int((spins[:, 0] == 1).sum()))
        self.assertTrue((sampler.sample_set.record.num_occurrences == 1).all())

        conditional = sampler.sample(torch.tensor([[1.0, float("nan"), -1.0, float("nan")]]))
        self.assertEqual((1, 4, 4), tuple(conditional.shape))

    def test_sample_conditional(self):
        sampler = DimodSampler(self.bm, SimulatedAnnealingSampler(), sample_kwargs=dict(num_reads=1))
        x = torch.tensor([
            [1.0, float("nan"), -1.0, float("nan")],
            [float("nan"), -1.0, float("nan"), 1.0],
        ])
        samples = sampler.sample(x)

        with self.subTest("Conditional sampling returns expected shape"):
            self.assertTupleEqual(samples.shape, (2, 1, 4))
        samples = samples.squeeze(1)

        with self.subTest("Conditional sampling preserves clamped variables"):
            mask = ~torch.isnan(x)
            self.assertTrue(torch.all(samples[mask] == x[mask]))

        with self.subTest("Conditional sampling samples free variables as ±1"):
            free_values = samples[torch.isnan(x)]
            self.assertTrue(torch.all(free_values.abs() == 1), "Free variables should be sampled as ±1")

        with self.subTest("Conditional sampling supports multiple reads."):
            num_reads = 5
            sampler = DimodSampler(self.bm, SimulatedAnnealingSampler(),
                                   sample_kwargs=dict(num_reads=num_reads))
            samples = sampler.sample(x)
            self.assertTupleEqual(samples.shape, (2, 5, 4))
            mask = ~torch.isnan(x)
            for i in range(num_reads):
                self.assertTrue(torch.all(samples[:, i, :][mask] == x[mask]))
                self.assertTrue(torch.all(samples[:, i, :][~mask].abs() == 1))

        with self.subTest("Batch dimensions are preserved."):
            samples = sampler.sample(x.reshape(2, 1, 4))
            self.assertTupleEqual(samples.shape, (2, 1, 5, 4))

        with self.subTest("Conditional sampling with all variables clamped returns input unchanged."):
            x_clamped = torch.tensor([
                [+1.0, -1.0, -1.0, +1.0],
                [-1.0, +1.0, -1.0, -1.0],
            ])
            samples = sampler.sample(x_clamped)
            self.assertTupleEqual(samples.shape, (2, 5, 4))
            for i in range(num_reads):
                torch.testing.assert_close(samples[:, i, :], x_clamped)

        with self.subTest("Conditional sampling supports mixed fully clamped and partially clamped rows."):
            x_mixed = torch.tensor([
                [1.0, -1.0, -1.0, 1.0],          # fully clamped
                [-1.0, float("nan"), 1.0, -1.0],  # partially clamped
            ])
            samples = sampler.sample(x_mixed)
            for i in range(num_reads):
                torch.testing.assert_close(samples[0, i, :], x_mixed[0])
            self.assertTrue(torch.all(samples[1, :, 0] == -1))
            self.assertTrue(torch.all(samples[1, :, 2] == 1))
            self.assertTrue(torch.all(samples[1, :, 3] == -1))

        with self.subTest("Conditional sampling rejects invalid input shape."):
            with self.assertRaisesRegex(ValueError, "x must have shape"):
                sampler.sample(torch.ones((2, self.n - 1)))

        with self.subTest("Conditional sampling rejects invalid spin values."):
            with self.assertRaisesRegex(ValueError, "only ±1 or NaN"):
                sampler.sample(torch.tensor([[1.0, 0.0, -1.0, float("nan")]]))

    def test_sample_conditional_bqm(self):
        with self.subTest("Conditional sampling clips linear biases of free variables."):
            set_weights(self.bm, [100.0, -100.0, 2.0, -2.0], [0.0] * 4)
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(self.bm, tracker, linear_range=[-1, 1])
            # Keep d and b free; clamp a and c.
            sampler.sample(torch.tensor([[float("nan"), float("nan"), 1.0, -1.0]]))
            bqm = tracker.input["bqm"]
            self.assertSetEqual({"d", "b"}, set(bqm.variables))
            self.assertEqual(bqm.get_linear("d"), 1.0)
            self.assertEqual(bqm.get_linear("b"), -1.0)

        with self.subTest("Conditional sampling clips linear biases after conditioning."):
            # Edge order is (a,b), (a,c), (a,d), (b,c); fixing a=+1 adds J_ab = 10 to b's bias.
            set_weights(self.bm, [0.0] * 4, [10.0, 0.0, 0.0, 0.0])
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(self.bm, tracker, linear_range=[-1, 1])
            # Node order is d, b, a, c. Fix a=+1. Variables b, c, d remain free.
            sampler.sample(torch.tensor([[float("nan"), float("nan"), 1.0, float("nan")]]))
            bqm = tracker.input["bqm"]
            self.assertEqual(bqm.get_linear("b"), 1.0)
            self.assertSetEqual({("b", "c")}, {tuple(sorted(e)) for e in bqm.quadratic})

        with self.subTest("Conditional biases are scaled by the prefactor and couplings kept."):
            set_weights(self.bm, [0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 6.0])
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(self.bm, tracker, prefactor=0.5, quadratic_range=[-2, 2])
            # Fix a=+1, d=-1; b and c are free: h_b = 1 + J_ab = 2, h_c = 3 + J_ac = 5
            sampler.sample(torch.tensor([[-1.0, float("nan"), 1.0, float("nan")]]))
            bqm = tracker.input["bqm"]
            self.assertAlmostEqual(bqm.get_linear("b"), 0.5 * 2.0)
            self.assertAlmostEqual(bqm.get_linear("c"), 0.5 * 5.0)
            self.assertAlmostEqual(bqm.get_quadratic("b", "c"), 2.0)  # 0.5 * 6 clipped to 2

    def test_sample_conditional_distribution(self):
        # The conditional samples must follow the model's conditional distribution
        set_weights(self.bm, [0.3, -0.2, 0.5, 0.1], [0.8, -0.6, 0.4, -0.9])
        sampler = DimodSampler(self.bm, ExactSolver())
        x = torch.tensor([[1.0, float("nan"), -1.0, float("nan")]])
        sampler.sample(x)
        sample_set = sampler.sample_set
        probabilities = torch.softmax(-torch.tensor(sample_set.record.energy.copy()), 0).float()
        variables = list(sample_set.variables)
        means = torch.tensor(sample_set.record.sample.copy(), dtype=torch.float32).T @ probabilities
        # Compare with exact conditional means from the full model by enumeration
        states = torch.tensor([[1.0, b, -1.0, c] for b in (-1.0, 1.0) for c in (-1.0, 1.0)])
        weights = torch.softmax(-self.bm(states), 0)
        expected = {node: (weights @ states[:, self.bm.node_to_idx[node]]).item() for node in variables}
        for node, mean in zip(variables, means.tolist()):
            self.assertAlmostEqual(expected[node], mean, places=5)

    def test_sample_conditional_inconsistent_reads(self):
        class InconsistentReadSampler:
            def __init__(self):
                self.calls = 0

            def sample(self, bqm, **kwargs):
                self.calls += 1
                num_reads = 5 if self.calls == 1 else 3
                samples = [{v: 1 for v in bqm.variables} for _ in range(num_reads)]
                return SampleSet.from_samples(samples, vartype=SPIN, energy=[0.0] * num_reads)

        sampler = DimodSampler(self.bm, InconsistentReadSampler())
        x = torch.tensor([
            [1.0, float("nan"), -1.0, float("nan")],
            [float("nan"), -1.0, float("nan"), 1.0],
        ])
        with self.assertRaisesRegex(ValueError, "Expected all samples to have shape"):
            sampler.sample(x)

    def test_sample_set(self):
        grbm = GRBM(list("abcd"), [("a", "b")])
        initial_states = [[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [-1, -1, 1, -1]]
        sampler = DimodSampler(grbm, IdentitySampler(),
                               sample_kwargs=dict(initial_states=(initial_states, "abcd")))
        with self.subTest("Accessing `sample_set` field before sampling should raise an error."):
            with self.assertRaisesRegex(RuntimeError, "no samples found"):
                sampler.sample_set

        sampler.sample()
        with self.subTest("The `sample_set` attribute should be of type `dimod.SampleSet`."):
            self.assertTrue(isinstance(sampler.sample_set, SampleSet))

    def test_device(self):
        sampler = DimodSampler(self.bm, IdentitySampler()).to("meta")
        self.assertEqual("meta", sampler.model.linear.device.type)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda(self):
        sampler = DimodSampler(self.bm, SimulatedAnnealingSampler(),
                               sample_kwargs=dict(num_reads=3)).cuda()
        spins = sampler.sample()
        self.assertTrue(spins.is_cuda)
        self.assertEqual((3, 4), tuple(spins.shape))
        x = torch.tensor([[1.0, float("nan"), -1.0, float("nan")]])
        conditional = sampler.sample(x)
        self.assertTrue(conditional.is_cuda)
        self.assertEqual((1, 3, 4), tuple(conditional.shape))
        self.assertTrue(torch.all(conditional[0, :, [0, 2]].cpu() == torch.tensor([1.0, -1.0])))


if __name__ == "__main__":
    unittest.main()
