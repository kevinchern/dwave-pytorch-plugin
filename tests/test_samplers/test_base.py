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
from dwave.plugins.torch.samplers.base import TorchSampler


class ConstantSampler(TorchSampler):
    """Returns the all-ones state."""

    def sample(self, x=None, num_samples=1):
        if x is None:
            return torch.ones(num_samples, self.model.n_nodes)
        x, clamp_mask, batch_shape = self._validate_conditional_input(x)
        completed = torch.where(clamp_mask, x, torch.ones_like(x))
        completed = completed.unsqueeze(-2).repeat_interleave(num_samples, -2)
        return completed.reshape(*batch_shape, num_samples, -1)


class TestTorchSampler(unittest.TestCase):
    """Test TorchSampler base class."""

    def setUp(self) -> None:
        self.model = GRBM(list("abc"), [("a", "b"), ("b", "c")])

    def test_subclass_without_sample(self):
        class EmptySubClass(TorchSampler):
            def something_else(self):
                pass

        with self.assertRaises(TypeError):
            EmptySubClass(self.model)  # type: ignore

    def test_requires_model(self):
        with self.assertRaisesRegex(TypeError, "GraphRestrictedBoltzmannMachine"):
            ConstantSampler(torch.nn.Linear(3, 3))

    def test_simple_subclass(self):
        sampler = ConstantSampler(self.model)
        torch.testing.assert_close(sampler.sample(), torch.ones(1, 3))
        with self.subTest("Calling the sampler is equivalent to sampling"):
            torch.testing.assert_close(sampler(), torch.ones(1, 3))
            x = torch.tensor([[-1.0, float("nan"), -1.0]])
            torch.testing.assert_close(sampler(x), torch.tensor([[[-1.0, 1.0, -1.0]]]))

    def test_model_is_submodule(self):
        sampler = ConstantSampler(self.model)
        self.assertIsInstance(sampler, torch.nn.Module)
        self.assertIs(self.model, sampler.model)
        self.assertListEqual([self.model], list(sampler.children()))
        self.assertSetEqual({"model.linear", "model.quadratic"},
                            {k for k in sampler.state_dict() if "idx" not in k and "adjacency" not in k})
        self.assertEqual(len(list(self.model.parameters())), len(list(sampler.parameters())))

        with self.subTest("Moving the sampler moves the model"):
            result = sampler.to(torch.device("meta"))
            self.assertIs(sampler, result)
            self.assertEqual(torch.device("meta"), self.model.linear.device)
            self.assertEqual(torch.device("meta"), self.model.adjacency.device)

    def test_complete(self):
        model = GRBM(list("abc"), [("a", "b"), ("b", "c")], hidden_nodes=["b"])
        sampler = ConstantSampler(model)
        x = torch.tensor([[1.0, -1.0], [-1.0, -1.0]], requires_grad=True)
        completed = sampler.complete(x)
        self.assertEqual((2, 1, 3), tuple(completed.shape))
        torch.testing.assert_close(completed[:, 0, [0, 2]], x.detach())
        self.assertTrue(torch.all(completed[..., 1] == 1))

        with self.subTest("Gradients propagate to the observations"):
            completed.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones(2, 2))

        with self.subTest("Keyword arguments are passed on to sample"):
            completed = sampler.complete(x.detach().reshape(2, 1, 2), num_samples=3)
            self.assertEqual((2, 1, 3, 3), tuple(completed.shape))
            self.assertTrue(torch.all(completed[..., [0, 2]] == x.detach().reshape(2, 1, 1, 2)))
            self.assertTrue(torch.all(completed[..., 1] == 1))

        with self.subTest("Wrong number of visible units"):
            with self.assertRaisesRegex(ValueError, "number of visible units"):
                sampler.complete(torch.ones(2, 3))

    def test_validate_conditional_input(self):
        sampler = ConstantSampler(self.model)

        with self.subTest("Valid input"):
            x = torch.tensor([[1.0, float("nan"), -1.0], [float("nan"), float("nan"), 1.0]])
            out, clamp_mask, batch_shape = sampler._validate_conditional_input(x)
            torch.testing.assert_close(out, x, equal_nan=True)
            self.assertListEqual(clamp_mask.tolist(), [[True, False, True], [False, False, True]])
            self.assertEqual((2,), tuple(batch_shape))

        with self.subTest("Batch dimensions are flattened and integer inputs are accepted"):
            out, clamp_mask, batch_shape = sampler._validate_conditional_input(
                torch.ones(2, 5, 3, dtype=torch.int64)
            )
            self.assertEqual((10, 3), tuple(out.shape))
            self.assertEqual((2, 5), tuple(batch_shape))
            self.assertEqual(self.model.linear.dtype, out.dtype)
            self.assertTrue(clamp_mask.all())

        with self.subTest("Wrong shape"):
            with self.assertRaisesRegex(ValueError, r"x must have shape \(\.\.\., 3\)"):
                sampler._validate_conditional_input(torch.ones(2, 2))

        with self.subTest("Non-spin values"):
            with self.assertRaisesRegex(ValueError, "only ±1 or NaN"):
                sampler._validate_conditional_input(torch.tensor([[0.5, 1.0, float("nan")]]))


if __name__ == "__main__":
    unittest.main()
