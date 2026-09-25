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
from dimod import SPIN, SampleSet
from torch import Tensor

from dwave.plugins.torch.utils import GraphIndex, sampleset_to_tensor, spread


class TestUtils(unittest.TestCase):
    def test_sample_to_tensor(self):
        bogus_energy = [999] * 3
        spins_in = [[1, -1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ss = SampleSet.from_samples((spins_in, list("dbca")), SPIN, bogus_energy)
        spins = sampleset_to_tensor(list("cabd"), ss)
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, Tensor)
        # Test variable ordering is respected
        self.assertListEqual(
            spins.tolist(), [[1, 1, -1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

    def test_spread(self):
        ss = SampleSet.from_samples(([[1, -1], [-1, 1]], "ab"), SPIN, [0, 0],
                                    num_occurrences=[3, 1], info={"x": 1})
        spread_ss = spread(ss)
        self.assertEqual(4, len(spread_ss))
        self.assertTrue((spread_ss.record.num_occurrences == 1).all())
        self.assertListEqual(spread_ss.record.sample.tolist(),
                             [[1, -1], [1, -1], [1, -1], [-1, 1]])
        self.assertListEqual(list(spread_ss.variables), ["a", "b"])
        self.assertEqual(SPIN, spread_ss.vartype)
        self.assertDictEqual({"x": 1}, spread_ss.info)

        with self.subTest("Sample sets without aggregation are returned as is"):
            ss = SampleSet.from_samples(([[1, -1], [-1, 1]], "ab"), SPIN, [0, 0])
            self.assertIs(ss, spread(ss))


class TestGraphIndex(unittest.TestCase):
    def test_from_graph(self):
        graph = GraphIndex.from_graph("dbac", [("a", "b"), ("a", "c"), ("d", "a"), ("b", "c")])
        self.assertListEqual(list("dbac"), graph.nodes)
        self.assertListEqual([("a", "b"), ("a", "c"), ("d", "a"), ("b", "c")], graph.edges)
        self.assertDictEqual({"d": 0, "b": 1, "a": 2, "c": 3}, graph.node_to_idx)
        self.assertEqual(4, graph.n_nodes)
        self.assertEqual(4, graph.n_edges)
        # canonical orientation: smaller index first
        self.assertListEqual([1, 2, 0, 1], graph.edge_idx_i.tolist())
        self.assertListEqual([2, 3, 2, 3], graph.edge_idx_j.tolist())
        self.assertListEqual([1, 2, 3, 2], graph.degrees().tolist())
        adjacency = graph.adjacency()
        self.assertEqual(torch.bool, adjacency.dtype)
        self.assertTrue(torch.equal(adjacency, adjacency.triu(1)))
        self.assertListEqual(adjacency.nonzero().tolist(), [[0, 2], [1, 2], [1, 3], [2, 3]])

    def test_edgeless(self):
        graph = GraphIndex.from_graph([0, 1], [])
        self.assertEqual(0, graph.n_edges)
        self.assertEqual((0,), tuple(graph.edge_idx_i.shape))
        self.assertListEqual([0, 0], graph.degrees().tolist())
        self.assertFalse(graph.adjacency().any())

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "duplicate entries"):
            GraphIndex.from_graph("aab", [])
        with self.assertRaisesRegex(ValueError, "not a node"):
            GraphIndex.from_graph("ab", [("a", "c")])
        with self.assertRaisesRegex(ValueError, r"Self-loops.*\('a', 'a'\)"):
            GraphIndex.from_graph("ab", [("a", "a")])
        with self.assertRaisesRegex(ValueError, "Duplicate edges"):
            GraphIndex.from_graph("ab", [("a", "b"), ("b", "a")])
