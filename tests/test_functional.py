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
from parameterized import parameterized

from dwave.plugins.torch.nn.functional import bit2spin_soft, gumbel_spins
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.nn.functional import spin2bit_soft
from tests.helper_functions import ConstantKernel


class TestMaximumMeanDiscrepancyLoss(unittest.TestCase):
    def test_mmd_loss_constant(self):
        x = torch.tensor([[1.2], [4.1]])
        y = torch.tensor([[0.3], [0.5]])

        # The resulting kernel matrix will be constant, so (averages) KXX = KYY = 2KXY
        kernel = ConstantKernel()
        # kxx = (4 + 4)/2
        # kyy = (3 + 3)/2
        # kxy = (0 + 1 + 4 + 2)/4
        # kxx + kyy -2kxy = 4 + 3 - 3.5 = 3.5
        self.assertEqual(3.5, mmd_loss(x, y, kernel))

    def test_sample_size_error(self):
        x = torch.tensor([[1.2], [4.1]])
        y = torch.tensor([[0.3]])
        self.assertRaisesRegex(ValueError, "must be at least two", mmd_loss, x, y, None)

    def test_mmd_loss_dim_mismatch(self):
        x = torch.tensor([[1], [4]], dtype=torch.float32)
        y = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6]])
        self.assertRaisesRegex(ValueError,
                               "Input dimensions must match. You are trying to compute ",
                               mmd_loss, x, y, None)

    def test_mmd_loss_arange(self):
        x = torch.tensor([[1.0], [4.0], [5.0]])
        y = torch.tensor([[0.3], [0.4]])

        kernel = ConstantKernel([[150, 22, 39, 34, 28],
                                 [22, 630, 98, 56, 44],
                                 [39, 98, 560, 78, 33],
                                 [-99, -99, -99, 299, 13],
                                 [-99, -99, -99, 13, 970]])
        # NOTE: calculation takes kxy = upper-right corner; no PSD assumption
        # kxx = (22+39+98)/3
        # kyy = 13
        # kxy = (34+28+56+44+78+33)/6
        # kxx + kyy - 2*kxy
        # kxx + kyy - 2*kxy = -25.0
        self.assertEqual(-25, mmd_loss(x, y, kernel))


class TestGumbelSpins(unittest.TestCase):
    def test_shape_and_values(self):
        logits = torch.linspace(-2, 2, 24).reshape(4, 3, 2)
        spins = gumbel_spins(logits, n_samples=5)
        self.assertEqual((4, 5, 3, 2), tuple(spins.shape))
        self.assertTrue(torch.all(spins.abs() == 1))
        self.assertEqual((4, 1, 3, 2), tuple(gumbel_spins(logits).shape))

    def test_strong_logits_are_deterministic(self):
        # A logit of 20 is flipped by the Gumbel noise with probability sigmoid(-20), i.e. never
        logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
        spins = gumbel_spins(logits, n_samples=100)
        expected = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).unsqueeze(1).expand(-1, 100, -1)
        torch.testing.assert_close(spins, expected)

    def test_straight_through_gradient(self):
        # At a high temperature the relaxation never saturates, so every logit receives a positive
        # gradient from every sample while the forward spins are still exactly ±1
        logits = torch.zeros(3, 4, requires_grad=True)
        spins = gumbel_spins(logits, n_samples=7, tau=100.0)
        self.assertTrue(torch.all(spins.abs() == 1))
        spins.sum().backward()
        self.assertTrue(torch.all(logits.grad > 0))

class TestFunctional(unittest.TestCase):

    def test_spin2bit_soft(self):
        self.assertListEqual(spin2bit_soft(torch.tensor([-1.0, 1.0, 0.5])).tolist(), [0, 1, 0.75])

    @parameterized.expand([([-1.1, 1.0],), ([-0.5, 1.1],)])
    def test_spin2bit_raises(self, input):
        self.assertRaises(ValueError, spin2bit_soft, torch.tensor(input))

    def test_bit2spin_soft(self):
        self.assertListEqual(bit2spin_soft(torch.tensor([0.0, 1.0, 0.5])).tolist(), [-1, 1, 0])

    @parameterized.expand([([-0.1, 1.0],), ([0.1, 1.1],)])
    def test_bit2spin_soft_raises(self, input):
        self.assertRaises(ValueError, bit2spin_soft, torch.tensor(input))


if __name__ == "__main__":
    unittest.main()
