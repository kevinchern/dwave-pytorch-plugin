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
import copy
import pickle
import unittest

import torch
from dimod import SPIN, BinaryQuadraticModel, SampleSet
from torch import Tensor

from dwave.plugins.torch.utils import (GraphIndex, estimate_beta, randspin, sampleset_to_tensor,
                                       to_bqm, to_ising)
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple
from tests.helper_functions import randspins


class TestUtils(unittest.TestCase):
    def test_sample_to_tensor(self):
        bogus_energy = [999] * 3
        spins_in = [[1, -1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ss = SampleSet.from_samples((spins_in, list("dbca")), SPIN, bogus_energy)
        spins = sampleset_to_tensor(list("cabd"), ss)
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, Tensor)
        self.assertEqual(torch.float32, spins.dtype)
        # Test variable ordering is respected
        self.assertListEqual(
            spins.tolist(), [[1, 1, -1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

        with self.subTest("Aggregated samples are repeated according to their occurrences"):
            ss = SampleSet.from_samples(([[1, -1], [-1, 1]], "ab"), SPIN, [0, 0],
                                        num_occurrences=[3, 1])
            spins = sampleset_to_tensor("ba", ss)
            self.assertListEqual(spins.tolist(), [[-1, 1]] * 3 + [[1, -1]])

        with self.subTest("Empty sample sets"):
            ss = SampleSet.from_samples(([], "ab"), SPIN, [])
            self.assertEqual((0, 2), tuple(sampleset_to_tensor("ab", ss).shape))


class TestRandspin(unittest.TestCase):
    def test_randspin(self):
        spins = randspin((2000,), generator=torch.Generator().manual_seed(0))
        self.assertEqual(torch.int64, spins.dtype)
        self.assertSetEqual({-1, 1}, set(spins.unique().tolist()))

        with self.subTest("Keyword arguments reach torch.randint"):
            spins = randspin((3, 4), dtype=torch.float32, device="meta")
            self.assertEqual((3, 4), tuple(spins.shape))
            self.assertEqual(torch.float32, spins.dtype)
            self.assertEqual("meta", spins.device.type)

        with self.subTest("Seeded draws are reproducible"):
            first = randspin((5, 5), generator=torch.Generator().manual_seed(3))
            second = randspin((5, 5), generator=torch.Generator().manual_seed(3))
            self.assertTrue(torch.equal(first, second))


class TestToIsing(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = list("dbac")
        self.edges = [["a", "b"], ("a", "c"), ("a", "d"), ("b", "c")]
        self.linear = torch.tensor([-3.0, 0.0, 1.0, 3.0], requires_grad=True)
        self.quadratic = torch.tensor([-1.0, 1.0, 2.0, 0.0], requires_grad=True)

    def test_to_ising(self):
        nodes, edges, linear, quadratic = self.nodes, self.edges, self.linear, self.quadratic

        with self.subTest("Dictionaries keyed by nodes and edges, in order"):
            h, J = to_ising(nodes, edges, linear, quadratic)
            self.assertListEqual(list(h), nodes)
            self.assertListEqual(list(h.values()), linear.detach().tolist())
            self.assertListEqual(list(J), [tuple(e) for e in edges])
            self.assertListEqual(list(J.values()), quadratic.detach().tolist())
            self.assertIsInstance(h["d"], float)

        with self.subTest("Prefactor"):
            h, J = to_ising(nodes, edges, linear, quadratic, prefactor=2.0)
            self.assertListEqual(list(h.values()), (2 * linear).detach().tolist())
            self.assertListEqual(list(J.values()), (2 * quadratic).detach().tolist())

        with self.subTest("Clipping after scaling"):
            h, J = to_ising(nodes, edges, linear, quadratic, 1, [-0.1, 1.5], [-0.05, 3])
            for expected, observed in zip([-0.1, 0, 1, 1.5], h.values()):
                self.assertAlmostEqual(expected, observed)
            for expected, observed in zip([-0.05, 1, 2, 0], J.values()):
                self.assertAlmostEqual(expected, observed)

        with self.subTest("Edgeless"):
            h, J = to_ising([0, 1], [], torch.zeros(2), torch.zeros(0))
            self.assertDictEqual(h, {0: 0.0, 1: 0.0})
            self.assertDictEqual(J, {})

        with self.subTest("Shape validation"):
            with self.assertRaisesRegex(ValueError, "Expected 4 linear biases"):
                to_ising(nodes, edges, torch.zeros(3), quadratic)
            with self.assertRaisesRegex(ValueError, "Expected 4 quadratic biases"):
                to_ising(nodes, edges, linear, torch.zeros(4, 4))

    def test_to_bqm(self):
        bqm = to_bqm(self.nodes, self.edges, self.linear, self.quadratic,
                     prefactor=2.0, linear_range=(-1, 5), quadratic_range=(-0.5, 3))
        self.assertIsInstance(bqm, BinaryQuadraticModel)
        self.assertEqual(SPIN, bqm.vartype)
        self.assertDictEqual({"d": -1.0, "b": 0.0, "a": 2.0, "c": 5.0}, dict(bqm.linear))
        self.assertDictEqual(
            {("a", "b"): -0.5, ("a", "c"): 2.0, ("a", "d"): 3.0, ("b", "c"): 0.0},
            {tuple(sorted(e)): b for e, b in bqm.quadratic.items()},
        )


class TestEstimateBeta(unittest.TestCase):
    def test_estimate_beta(self):
        nodes, edges = list("dbac"), [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")]
        linear = torch.tensor([0.0, 1.0, 2.0, 3.0])
        quadratic = torch.tensor([1.0, 2.0, 3.0, 6.0])
        spins = torch.tensor([[1, -1, 1, 1], [-1, -1, 1, 1], [1, -1, -1, 1], [1, 1, 1, -1]])
        beta = estimate_beta(nodes, edges, linear, quadratic, spins)
        self.assertIsInstance(beta, float)
        bqm = to_bqm(nodes, edges, linear, quadratic)
        self.assertEqual(1.0 / mple(bqm, (spins.numpy(), nodes))[0], beta)


class TestGraphIndex(unittest.TestCase):
    def test_index(self):
        graph = GraphIndex("dbac", [("a", "b"), ("a", "c"), ("d", "a"), ("b", "c")])
        self.assertTupleEqual(tuple("dbac"), graph.nodes)
        self.assertTupleEqual((("a", "b"), ("a", "c"), ("d", "a"), ("b", "c")), graph.edges)
        self.assertDictEqual({"d": 0, "b": 1, "a": 2, "c": 3}, graph.node_to_idx)
        self.assertEqual(4, graph.n_nodes)
        self.assertEqual(4, graph.n_edges)
        # canonical orientation: smaller index first
        self.assertListEqual([1, 2, 0, 1], graph.edge_idx_i.tolist())
        self.assertListEqual([2, 3, 2, 3], graph.edge_idx_j.tolist())
        self.assertListEqual([1, 2, 3, 2], graph.degrees().tolist())
        adjacency = graph.adjacency
        self.assertEqual(torch.bool, adjacency.dtype)
        self.assertTrue(torch.equal(adjacency, adjacency.triu(1)))
        self.assertListEqual(adjacency.nonzero().tolist(), [[0, 2], [1, 2], [1, 3], [2, 3]])
        self.assertIn("n_nodes=4, n_edges=4", repr(graph))

    def test_module(self):
        graph = GraphIndex("abc", [("a", "b")])
        self.assertIsInstance(graph, torch.nn.Module)
        self.assertEqual(0, len(list(graph.parameters())))
        self.assertSetEqual({"edge_idx_i", "edge_idx_j", "adjacency"}, set(graph.state_dict()))

        with self.subTest("Buffers move with the module"):
            graph.to("meta")
            self.assertEqual("meta", graph.adjacency.device.type)
            self.assertEqual("meta", graph.edge_idx_i.device.type)

        with self.subTest("The module can be copied and pickled"):
            graph = GraphIndex("abc", [("a", "b")])
            clone = copy.deepcopy(graph)
            self.assertTupleEqual(graph.nodes, clone.nodes)
            self.assertDictEqual(graph.node_to_idx, clone.node_to_idx)
            self.assertTrue(torch.equal(graph.adjacency, clone.adjacency))
            self.assertTupleEqual(graph.edges, pickle.loads(pickle.dumps(graph)).edges)

    def test_edge_biases_roundtrip(self):
        graph = GraphIndex("abc", [("b", "a"), ("a", "c"), ("b", "c")])
        per_edge = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        dense = graph.dense_quadratic(per_edge)
        self.assertEqual((2, 3, 3), tuple(dense.shape))
        torch.testing.assert_close(dense[0], torch.tensor([[0.0, 1.0, 2.0],
                                                           [0.0, 0.0, 3.0],
                                                           [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(graph.edge_biases(dense), per_edge)
        with self.assertRaisesRegex(ValueError, "Expected 3 edge biases"):
            graph.dense_quadratic(torch.zeros(2, 2))

    def test_energy_and_effective_field(self):
        graph = GraphIndex("dbac", [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")])
        linear = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.5, -1.0, 0.0, 2.0]])
        edge_biases = torch.tensor([[1.0, 2.0, 3.0, 6.0], [-1.0, 0.5, 0.0, 2.0]])
        quadratic = graph.dense_quadratic(edge_biases)
        spins = randspins(2, 7, 4, seed=3)
        energies = graph.energy(spins, linear, quadratic)
        self.assertEqual((2, 7), tuple(energies.shape))

        with self.subTest("Batched biases match dimod energies model by model"):
            for b in range(2):
                bqm = to_bqm(graph.nodes, graph.edges, linear[b], edge_biases[b])
                expected = bqm.energies((spins[b].numpy(), list(graph.nodes)))
                torch.testing.assert_close(energies[b], torch.tensor(expected, dtype=torch.float32))

        with self.subTest("Unbatched biases apply to spins of any leading shape"):
            unbatched = graph.energy(spins, linear[0], quadratic[0])
            self.assertEqual((2, 7), tuple(unbatched.shape))
            torch.testing.assert_close(unbatched[0], energies[0])
            torch.testing.assert_close(graph.energy(spins[1, 0], linear[0], quadratic[0]), unbatched[1, 0])

        with self.subTest("One configuration per batched model"):
            torch.testing.assert_close(graph.energy(spins[:, 0], linear, quadratic), energies[:, 0])

        with self.subTest("Effective fields are the gradients of the energy"):
            x = spins.clone().requires_grad_()
            grad, = torch.autograd.grad(graph.energy(x, linear, quadratic).sum(), x)
            fields = graph.effective_field(spins, linear=linear, quadratic=quadratic)
            torch.testing.assert_close(grad, fields)
            torch.testing.assert_close(
                graph.effective_field(spins[:, 0], linear=linear, quadratic=quadratic), fields[:, 0]
            )

        with self.subTest("Subsets of nodes and precomputed couplings"):
            idx = torch.tensor([2, 0])
            coupling = graph.symmetric_coupling(quadratic)
            torch.testing.assert_close(
                graph.effective_field(spins, linear=linear, coupling=coupling, idx=idx), fields[..., idx]
            )

        with self.subTest("Unknown spins contribute nothing"):
            x = spins.clone()
            x[..., 1] = torch.nan
            zeroed = spins.clone()
            zeroed[..., 1] = 0.0
            torch.testing.assert_close(
                graph.effective_field(x, linear=linear, quadratic=quadratic),
                graph.effective_field(zeroed, linear=linear, quadratic=quadratic),
            )

        with self.subTest("Couplings are required"):
            with self.assertRaisesRegex(ValueError, "`quadratic` or `coupling`"):
                graph.effective_field(spins, linear=linear)

    def test_edgeless(self):
        graph = GraphIndex([0, 1], [])
        self.assertEqual(0, graph.n_edges)
        self.assertEqual((0,), tuple(graph.edge_idx_i.shape))
        self.assertListEqual([0, 0], graph.degrees().tolist())
        self.assertFalse(graph.adjacency.any())

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "duplicate entries"):
            GraphIndex("aab", [])
        with self.assertRaisesRegex(ValueError, "not a node"):
            GraphIndex("ab", [("a", "c")])
        with self.assertRaisesRegex(ValueError, r"Self-loops.*\('a', 'a'\)"):
            GraphIndex("ab", [("a", "a")])
        with self.assertRaisesRegex(ValueError, "Duplicate edges"):
            GraphIndex("ab", [("a", "b"), ("b", "a")])


if __name__ == "__main__":
    unittest.main()
