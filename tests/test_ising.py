# Copyright 2026 D-Wave
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
from dimod import BQM, ExactSolver

from dwave.plugins.torch.nn.modules.ising import Ising, IsingExpectation
from dwave.plugins.torch.nn.modules.spin_statistic import (IdentityStatistic, IsingStatistic,
                                                           SpinStatistic)
from dwave.plugins.torch.samplers import BlockSampler, DimodSampler
from dwave.plugins.torch.utils import to_bqm
from dwave.samplers import SimulatedAnnealingSampler as Neal
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple
from tests.helper_functions import randspins


class _ConcreteStatistic(SpinStatistic):
    """Minimal concrete implementation for testing the ABC."""

    def _transform(self, x):
        return x[..., :self.dim_out]


class TestStatistic(unittest.TestCase):
    """Verify the Statistic ABC enforces its contract:
    - inputs must be 3D (batch, samples, features)
    - outputs must be 3D with last dim == dim_out
    """

    def test_dim_out(self):
        # The dim_out property should reflect what was passed at construction
        stat = _ConcreteStatistic(dim_out=3)
        self.assertEqual(stat.dim_out, 3)

    def test_rejects_2d_input(self):
        # 2D tensors lack the sample dimension
        stat = _ConcreteStatistic(dim_out=4)
        x = torch.randn(2, 4)
        with self.assertRaisesRegex(ValueError, r"Input tensor.*ndim == 3"):
            stat(x)

    def test_rejects_4d_input(self):
        # 4D tensors have an extra spatial dim
        stat = _ConcreteStatistic(dim_out=4)
        x = torch.randn(2, 3, 4, 5)
        with self.assertRaisesRegex(ValueError, r"Input tensor.*ndim == 3"):
            stat(x)

    def test_rejects_wrong_output_dim(self):
        # If _transform returns a tensor whose last dim != dim_out, catch the bug early
        class BadStatistic(SpinStatistic):
            def _transform(self, x):
                return x[..., :2]

        stat = BadStatistic(dim_out=5)
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(ValueError, r"Output dimension.*does not match"):
            stat(x)

    def test_rejects_wrong_output_ndim(self):
        # If _transform collapses a dimension (e.g. returns 2D), catch it
        class SquashStatistic(SpinStatistic):
            def _transform(self, x):
                return x.mean(dim=1)  # (batch, features) — loses sample dim

        stat = SquashStatistic(dim_out=4)
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(ValueError, r"Output tensor.*ndim == 3"):
            stat(x)

    def test_valid_call(self):
        # Happy path: correct shape in, correct shape out
        stat = _ConcreteStatistic(dim_out=3)
        x = torch.randn(2, 4, 5)
        result = stat(x)
        self.assertEqual(result.shape, (2, 4, 3))


class TestIdentityStatistic(unittest.TestCase):
    """Verify IdentityStatistic passes input through unchanged."""

    def test_dim_out(self):
        stat = IdentityStatistic(5)
        self.assertEqual(stat.dim_out, 5)

    def test_output_equals_input(self):
        stat = IdentityStatistic(4)
        x = torch.randn(2, 3, 4)
        result = stat(x)
        torch.testing.assert_close(result, x)

    def test_output_shape(self):
        stat = IdentityStatistic(7)
        x = torch.randn(5, 10, 7)
        result = stat(x)
        self.assertEqual(result.shape, (5, 10, 7))


class TestIsingStatistic(unittest.TestCase):
    """Verify Ising correctly computes [x[indices], x[indices_j]*x[indices_i]]
    under various edge cases encountered in practice."""

    def test_transform(self):
        # Core behaviour: picks nodes and computes pairwise products
        stat = IsingStatistic(
            node_indices=[0, 2],
            endpoints_1=[1],
            endpoints_2=[3],
        )
        # shape (batch=1, samples=2, nodes=4)
        x = torch.tensor([[[1.0, -1.0, 1.0, -1.0],
                           [1.0,  1.0, -1.0, 1.0]]])
        result = stat(x)
        # expected: [x[...,0], x[...,2], x[...,3]*x[...,1]]
        expected = torch.tensor([[[1.0, 1.0, (-1.0)*(-1.0)],
                                  [1.0, -1.0, 1.0*1.0]]])
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        # Verify shape is (batch, samples, dim_out) for arbitrary input sizes
        stat = IsingStatistic(
            node_indices=[0, 1],
            endpoints_1=[0, 2],
            endpoints_2=[1, 3],
        )
        x = torch.randn(3, 5, 7)
        result = stat(x)
        self.assertEqual(result.shape, (3, 5, 4))

    def test_no_interactions(self):
        # Edge case: only node indices, no interaction terms
        # (e.g. an Ising model with no input_edges)
        stat = IsingStatistic(node_indices=[0, 3], endpoints_1=[], endpoints_2=[])
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]],
                          [[0.1, 2.0, 3.0, 5.0]]])
        result = stat(x)
        torch.testing.assert_close(result, torch.tensor([[[1.0, 4.0]], [[0.1, 5.0]]]))

    def test_no_nodes(self):
        # Edge case: only interaction terms, no direct node indices
        stat = IsingStatistic(node_indices=[], endpoints_1=[0], endpoints_2=[1])
        x = torch.tensor([[[3.0, -2.0]]])
        result = stat(x)
        torch.testing.assert_close(result, torch.tensor([[[-6.0]]]))

    def test_bad_edge_indices(self):
        # Edge indices have different length
        with self.assertRaisesRegex(ValueError, "Interaction indices should be of the same length, got"):
            IsingStatistic(node_indices=[], endpoints_1=[0], endpoints_2=[1, 2])

    def test_from_graph(self):
        # All spins followed by the products along the (canonically oriented) edges
        ising = Ising("abc", [("b", "a"), ("a", "c")])
        stat = IsingStatistic.from_graph(ising)
        self.assertEqual(5, stat.dim_out)
        x = torch.tensor([[[1.0, -1.0, 1.0]]])
        torch.testing.assert_close(stat(x), torch.tensor([[[1.0, -1.0, 1.0, -1.0, 1.0]]]))

    def test_tensor_indices(self):
        # Ising internally passes nn.Parameter tensors as indices;
        # verify IsingStatistic handles them the same as lists
        stat = IsingStatistic(
            node_indices=torch.tensor([1]),
            endpoints_1=torch.tensor([0]),
            endpoints_2=torch.tensor([2]),
        )
        x = torch.nn.Parameter(torch.tensor([[[2.0, 3.0, 4.0]]]))
        result = stat(x)
        torch.testing.assert_close(result, torch.tensor([[[3.0, 8.0]]]))


class TestIsingExpectation(unittest.TestCase):

    def test_forward_backward(self):
        """Test forward and backward evaluations are as expected."""
        spins = torch.tensor([[[-1, -1, -1],
                               [-1,  1,  1],
                               [1,  1,  1]],
                              [[1, -1,  1],
                               [-1, -1,  1],
                               [-1, -1,  1]]]).float()
        # Edges are (0, 1) and (1, 2); interaction terms are col0*col1, col1*col2
        adjacency = torch.zeros(3, 3, dtype=torch.bool)
        adjacency[[0, 1], [1, 2]] = True
        interactions = spins[..., [0, 1]] * spins[..., [1, 2]]
        sufficient_stats = torch.cat([spins, interactions], dim=-1)

        # Statistic is col0 + col1, col1*col2
        output_stats = torch.cat([spins[..., [0]] + spins[..., [1]],
                                  spins[..., [1]] * spins[..., [2]]],
                                 dim=-1)

        linear = torch.tensor([[-0.1, -0.2, -0.3]] * 2, requires_grad=True)
        quadratic = torch.zeros(2, 3, 3, requires_grad=True)

        y = IsingExpectation.apply(spins, output_stats, adjacency, linear, quadratic)

        with self.subTest("Ising aggregation layer produced unexpected output values"):
            with torch.no_grad():
                torch.testing.assert_close(y, output_stats.mean(1))

        # Gradient amounts to summing over gradients per obs
        loss = (y**2).sum()
        loss.backward()

        with torch.no_grad():
            # Manually compute the gradients
            dloss_dy = 2*y
            dy_dhJ = -torch.stack([torch.cat([o, i], -1).mT.cov()[:2, 2:]
                                   for i, o in zip(sufficient_stats, output_stats)])
            dloss_dhJ = torch.einsum("bi, bij -> bj", dloss_dy, dy_dhJ)

        with self.subTest("Linear gradients should match"):
            torch.testing.assert_close(dloss_dhJ[:, :3], linear.grad)

        with self.subTest("Quadratic gradients should match at the edges and vanish elsewhere"):
            torch.testing.assert_close(dloss_dhJ[:, 3:], quadratic.grad[:, [0, 1], [1, 2]])
            self.assertTrue(torch.all(quadratic.grad[:, ~adjacency] == 0))

    def test_rejects_non_3d_statistics(self):
        spins = torch.ones(2, 3, 3)
        with self.assertRaisesRegex(ValueError, "ndim should be 3"):
            IsingExpectation.apply(spins, torch.ones(2, 3), torch.ones(3, 3, dtype=torch.bool),
                                   torch.zeros(2, 3), torch.zeros(2, 3, 3))


class TestIsing(unittest.TestCase):
    NODES = "abc"
    EDGES = [("a", "b"), ("a", "c"), ("b", "c")]

    def test_has_properties(self):
        ising = Ising(self.NODES, self.EDGES, statistic=IsingStatistic([1], [0, 1], [1, 2]))
        self.assertTupleEqual(("a", "b", "c"), ising.nodes)
        self.assertTupleEqual((("a", "b"), ("a", "c"), ("b", "c")), ising.edges)
        self.assertDictEqual({"a": 0, "b": 1, "c": 2}, ising.node_to_idx)
        self.assertEqual(3, ising.n_nodes)
        self.assertEqual(3, ising.n_edges)
        self.assertEqual(3, ising.dim_out)
        self.assertIs(ising.statistic, dict(ising.named_children())["statistic"])
        self.assertEqual(0, len(list(ising.parameters())))
        self.assertIn("n_nodes=3, n_edges=3", repr(ising))

        with self.subTest("The default statistic is the identity"):
            ising = Ising(self.NODES, self.EDGES)
            self.assertIsInstance(ising.statistic, IdentityStatistic)
            self.assertEqual(3, ising.dim_out)

    def test_correct_node_indices_of_edges(self):
        ising = Ising("abc", [("b", "a"), ("a", "c"), ("b", "c")])
        # Canonical (upper-triangular) orientation regardless of the given orientation
        self.assertListEqual([0, 0, 1], ising.edge_idx_i.tolist())
        self.assertListEqual([1, 2, 2], ising.edge_idx_j.tolist())
        expected = torch.zeros(3, 3, dtype=torch.bool)
        expected[[0, 0, 1], [1, 2, 2]] = True
        self.assertTrue(torch.equal(expected, ising.adjacency))

    def test_edge_biases_roundtrip(self):
        ising = Ising("abc", [("b", "a"), ("a", "c"), ("b", "c")])
        per_edge = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        dense = ising.dense_quadratic(per_edge)
        self.assertEqual((2, 3, 3), tuple(dense.shape))
        torch.testing.assert_close(dense[0], torch.tensor([[0.0, 1.0, 2.0],
                                                           [0.0, 0.0, 3.0],
                                                           [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(ising.edge_biases(dense), per_edge)
        with self.assertRaisesRegex(ValueError, "Expected 3 edge biases"):
            ising.dense_quadratic(torch.zeros(2, 2))

    def test_invalid_graph(self):
        with self.assertRaisesRegex(ValueError, "Self-loops"):
            Ising("abc", [("a", "a")])
        with self.assertRaisesRegex(ValueError, "Duplicate edges"):
            Ising("abc", [("a", "b"), ("b", "a")])

    def test_input_validation(self):
        ising = Ising("abc", [("a", "b")])
        linear, quadratic, spins = torch.zeros(2, 3), torch.zeros(2, 3, 3), torch.ones(2, 5, 3)
        with self.assertRaisesRegex(ValueError, r"linear should have shape \(B, 3\)"):
            ising(torch.zeros(2, 4), quadratic, spins)
        with self.assertRaisesRegex(ValueError, r"quadratic should have shape \(B, 3, 3\)"):
            ising(linear, torch.zeros(2, 2), spins)
        with self.assertRaisesRegex(ValueError, r"spins should have shape \(B, M, 3\)"):
            ising(linear, quadratic, torch.ones(3, 5, 3))
        with self.assertRaisesRegex(ValueError, r"spins should have shape \(B, M, 3\)"):
            ising(linear, quadratic, torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, r"spins should have shape \(B, M, 3\)"):
            ising.estimate_betas(linear, quadratic, torch.ones(2, 3))

    def test_forward_backward(self):
        # Two models with three samples each, obtained by whatever means
        spins = torch.tensor([[[-1, -1, -1],
                               [-1,  1,  1],
                               [1,  1,  1]],
                              [[1, -1,  1],
                               [-1, -1,  1],
                               [-1, -1,  1]]]).float().requires_grad_()
        ising = Ising(self.NODES, self.EDGES, statistic=IsingStatistic([1], [0, 1], [1, 2]))

        # The values of the biases do not affect the output; they receive the gradients
        linear = torch.zeros((2, 3), requires_grad=True)
        quadratic = torch.zeros((2, 3, 3), requires_grad=True)
        y = ising(linear, quadratic, spins)
        with self.subTest("Ising layer produced unexpected output values"):
            torch.testing.assert_close(y, torch.tensor([[1/3, 1/3, 1],
                                                        [-1, 1/3, -1]]))

        # Gradient amounts to summing over gradients per obs
        (y**2).sum().backward()

        # Manually compute the gradients: minus the covariance of the output statistic with the
        # sufficient statistics (spins and pairwise products along the edges)
        grads = []
        for b in range(2):
            t = spins[b].detach()
            stat = torch.hstack([t[..., [1]], t[..., [0, 1]] * t[..., [1, 2]]])
            sufficient = torch.hstack([t, t[..., [0, 0, 1]] * t[..., [1, 2, 2]]])
            grads.append(2 * y[b] @ (-torch.cat([stat, sufficient], -1).mT.cov()[:3, 3:]))
        grad = torch.vstack(grads)

        with self.subTest("Linear gradients should match"):
            torch.testing.assert_close(grad[:, :3], linear.grad)

        with self.subTest("Quadratic gradients should match"):
            torch.testing.assert_close(grad[:, 3:], ising.edge_biases(quadratic.grad))
            self.assertTrue(torch.all(quadratic.grad[:, ~ising.adjacency] == 0))

        with self.subTest("Spins receive no gradient"):
            self.assertIsNone(spins.grad)

    def test_energy(self):
        # The layer inherits the batched energies of its graph
        ising = Ising(self.NODES, self.EDGES)
        linear = torch.tensor([[0.1, 0.2, 0.4], [-0.9, -0.8, -0.6]])
        edge_biases = torch.tensor([[1.0, 9.0, 4.0], [0.9, 8.0, -12.0]])
        spins = randspins(2, 6, 3, seed=1)
        energies = ising.energy(spins, linear, ising.dense_quadratic(edge_biases))
        self.assertEqual((2, 6), tuple(energies.shape))
        for b in range(2):
            bqm = to_bqm(ising.nodes, ising.edges, linear[b], edge_biases[b])
            expected = bqm.energies((spins[b].numpy(), list(ising.nodes)))
            torch.testing.assert_close(energies[b], torch.tensor(expected, dtype=torch.float32))

    def test_estimate_betas(self):
        s1 = [[-1, -1, -1],
              [-1,  1,  1],
              [1,  1,  1]]
        s2 = [[1,  -1,  1],
              [-1, -1, 1],
              [-1, -1, 1]]
        linear = torch.tensor([[0.1, 0.2, 0.4],
                               [-0.9, -0.8, -0.6]])
        quadratic = torch.tensor([[1.0, 9.0, 4.0],
                                  [0.9, 8.0, -12.0]])
        ising = Ising(self.NODES, self.EDGES)
        spins = torch.tensor([s1, s2]).float()

        estimated_betas = ising.estimate_betas(linear, ising.dense_quadratic(quadratic), spins)

        dimod_betas = []
        for h, J, s in zip(linear, quadratic, (s1, s2)):
            bqm = BQM.from_ising(dict(zip("abc", h.tolist())), dict(zip(self.EDGES, J.tolist())))
            dimod_betas.append(1 / float(mple(bqm, (s, list("abc")))[0]))
        torch.testing.assert_close(torch.tensor(dimod_betas), estimated_betas)

    def test_sampling_with_block_sampler(self):
        # The layer is sampled on its own graph by a sampler bound to it; strong fields make the
        # Gibbs draws deterministic (a spin flips against a field of 10 with probability e^-20)
        ising = Ising("abc", [("a", "b"), ("a", "c")])
        sampler = BlockSampler(ising, schedule=[1.0] * 5)
        linear = torch.tensor([[10.0] * 3, [-10.0] * 3])
        quadratic = torch.zeros(2, 3, 3)
        spins = sampler.sample_biases(linear, quadratic, num_samples=100)
        self.assertEqual((2, 100, 3), tuple(spins.shape))
        torch.testing.assert_close(
            ising(linear, quadratic, spins), torch.tensor([[-1.0] * 3, [1.0] * 3])
        )

        with self.subTest("The layer has no parameters of its own to sample"):
            with self.assertRaisesRegex(TypeError, "GraphRestrictedBoltzmannMachine"):
                sampler.sample()

    def test_sampling_with_dimod_sampler(self):
        ising = Ising("abc", [("a", "b"), ("a", "c")])
        bs = 4
        quadratic = torch.zeros((bs, 3, 3))

        with self.subTest("An exactly enumerated uniform prior gives exactly zero statistics"):
            # ExactSolver returns every state once, which is the Boltzmann distribution of zero
            # biases, so the layer's averages vanish exactly
            sampler = DimodSampler(ising, ExactSolver())
            linear = torch.zeros(bs, 3)
            spins = sampler.sample_biases(linear, quadratic)
            self.assertEqual((bs, 8, 3), tuple(spins.shape))
            torch.testing.assert_close(ising(linear, quadratic, spins), torch.zeros(bs, 3))

        with self.subTest("Biases are scaled by the prefactor of a sampler at another temperature"):
            # Scaled to 1e14, the fields pin every spin of every read of simulated annealing
            sampler = DimodSampler(ising, Neal(), prefactor=1e20,
                                   sample_kwargs=dict(num_sweeps=1, num_reads=10, beta_range=[1, 1]))
            linear = torch.ones((bs, 3)) * 1e-6
            spins = sampler.sample_biases(linear, quadratic)
            self.assertEqual((bs, 10, 3), tuple(spins.shape))
            torch.testing.assert_close(ising(linear, quadratic, spins), -torch.ones(bs, 3))

    def test_statistic_moves_with_layer(self):
        ising = Ising("abc", [("a", "b")], statistic=IsingStatistic([1], [0], [1])).to("meta")
        self.assertEqual("meta", ising.statistic.node_indices.device.type)
        self.assertNotIn("statistic.node_indices", ising.state_dict())


if __name__ == "__main__":
    unittest.main()
