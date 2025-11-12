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
#
import torch


class RadialBasis(torch.nn.Module):

    def __init__(self, n=1, base=2.0, bandwidth=None):
        super().__init__(self, vars())
        factors = base ** (torch.arange(n) - n // 2)
        self.register_buffer("factors", factors)
        self.bandwidth = bandwidth

    @torch.no_grad
    def get_bandwidth(self, l2):
        if self.bandwidth is None:
            n = l2.shape[0]
            # diagonal is zero
            avg = l2.sum() / (n**2 - n)
            return avg

        return self.bandwidth

    def forward(self, X):
        l2 = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(l2.detach()) * self.factors
        res = torch.exp(-l2.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)).sum(dim=0)
        return res


class MaximumMeanDiscrepancy(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__(self, vars())
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X.flatten(1), Y.flatten(1)]))
        n = X.shape[0]
        m = Y.shape[0]
        XX = (K[:n, :n].sum() - K[:n, :n].trace()) / (n*(n-1))
        YY = (K[n:, n:].sum() - K[n:, n:].trace()) / (m*(m-1))
        XY = K[:n, n:].mean()
        mmd = XX - 2 * XY + YY
        return mmd
