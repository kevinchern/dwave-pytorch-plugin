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

import torch


class Bounded(torch.nn.Module):
    def __init__(self, lower_bound, upper_bound, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if lower_bound > upper_bound:
            raise ValueError("Lower bound must be less than or equal to upper bound.")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x: torch.Tensor):
        return torch.clip(x, self.lower_bound, self.upper_bound)
