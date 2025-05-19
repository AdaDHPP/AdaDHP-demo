# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose


class AdaDHPLayer(BaseTunerLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("my_l", "my_r")

    def __init__(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self.my_l = nn.ParameterDict({})
        self.my_r = nn.ParameterDict({})
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, init_adadhp_weights):
        self.my_l[adapter_name] = nn.Parameter(torch.randn(1, self.in_features))
        self.my_r[adapter_name] = nn.Parameter(torch.randn(self.out_features, 1))
        if init_adadhp_weights:
            self.reset_adadhp_parameters(adapter_name)
        self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_adadhp_parameters(self, adapter_name):
        if adapter_name in self.my_l.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.my_l[adapter_name], 1.0)
            nn.init.constant_(self.my_r[adapter_name], 1.0)


class AdaDHPLinear(nn.Module, AdaDHPLayer):
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str,
            fan_in_fan_out: bool = False,
            init_adadhp_weights: bool = True,
            **kwargs,
    ) -> None:
        super().__init__()
        AdaDHPLayer.__init__(self, base_layer)
        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, init_adadhp_weights=True)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.my_l.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights *= self.my_l * self.my_r

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data *= self.my_l * self.my_r
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.my_l.keys():
                self.get_base_layer().weight.data /= self.my_l * self.my_r

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            for active_adapter in self.active_adapters:
                my_l = self.my_l[active_adapter]
                my_r = self.my_r[active_adapter]

                if isinstance(self.base_layer, torch.nn.Linear):
                    if self.base_layer.bias is not None:
                        result = torch.matmul(x, ((self.base_layer.weight * my_l * my_r).T).to(x.dtype)) + self.base_layer.bias
                        # result = torch.matmul(x * my_l.to(x.dtype), ((self.base_layer.weight).T)) * (
                        #     (my_r.T).to(x.dtype)) + self.base_layer.bias
                    else:
                        result = torch.matmul(x, ((self.base_layer.weight * my_l * my_r).T).to(x.dtype))
                        # result = torch.matmul(x * my_l.to(x.dtype), ((self.base_layer.weight).T)) * (
                            # (my_r.T).to(x.dtype))
                elif isinstance(self.base_layer, Conv1D):
                    size_out = x.size()[:-1] + (self.base_layer.nf,)
                    # result = torch.addmm(self.base_layer.bias, x.view(-1, x.size(-1)), self.base_layer.weight * my_l.T * my_r.T)
                    result = torch.addmm(self.base_layer.bias, x.view(-1, x.size(-1)) * my_l.to(x.dtype),
                                         self.base_layer.weight) * ((my_r.T).to(x.dtype))
                    result = result.view(size_out)
        return result


class RankAllocator:
    """
    The RankAllocator for AdaLoraModel. Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        config ([`AdaLoraConfig`]): The configuration of the AdaLora model.
        model: the model that we apply AdaLoRA to.

    """

    def __init__(self, model, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self.beta1 = peft_config.beta1
        self.beta2 = peft_config.beta2
        assert self.beta1 > 0 and self.beta1 < 1
        assert self.beta2 > 0 and self.beta2 < 1
        self.last_current_rank = None

        self.reset_ipt()
        self._set_budget_scheduler(model)

    def set_total_step(self, total_step):
        self.peft_config.total_step = total_step
        print("-" * 100)
        print(total_step)
        print(total_step * (100 / 1720))
        print(total_step * (800 / 1720))
        print("-" * 100)
        assert self.peft_config.total_step > self.peft_config.tfinal

    def reset_ipt(self):
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}

    def _set_budget_scheduler(self, model):
        self.total_num = 0
        self.name_set = set()
        for n, p in model.named_parameters():
            if f"my_l.{self.adapter_name}" in n and p.requires_grad:
                self.total_num += 1
                self.name_set.add(n.replace("_l", "%s"))
        print("total_num" * 10)
        print(self.total_num)
        self.name_set = sorted(self.name_set)
        # The total final trainable num
        self.target_num = self.peft_config.target_num

    def budget_schedule(self, step: int):
        tinit = self.peft_config.tinit
        tfinal = self.peft_config.tfinal
        total_step = self.peft_config.total_step
        # Initial warmup
        if step <= tinit:
            curr_num = self.total_num
            mask_ind = False
        # Final fine-tuning
        elif step > tfinal:
            curr_num = self.target_num
            mask_ind = True
        else:
            mul_coeff = (tfinal - step) / (tfinal - tinit)
            curr_num = int((self.total_num - self.target_num) * (mul_coeff ** 3) + self.target_num)
            mask_ind = True if step % self.peft_config.deltaT == 0 else False
        return curr_num, mask_ind

    def update_ipt(self, model):
        # Update the sensitivity and uncertainty for every weight
        for n, p in model.named_parameters():
            if "my_" in n and self.adapter_name in n and p.requires_grad:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Sensitivity smoothing
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                    # Uncertainty quantification
                    self.exp_avg_unc[n] = (
                            self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (
                            self.ipt[n] - self.exp_avg_ipt[n]).abs()
                    )

    def _element_score(self, n):
        return self.exp_avg_ipt[n] * self.exp_avg_unc[n]

    def mask_to_budget(self, model, curr_rank):
        vector_ipt = {}
        triplet_ipt = {}
        # Get the importance score for A, E, B
        for n, p in model.named_parameters():
            if f"my_l.{self.adapter_name}" in n and p.requires_grad:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("_l", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)
            if f"my_r.{self.adapter_name}" in n and p.requires_grad:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("_r", "%s")
                if name_m not in vector_ipt:
                    vector_ipt[name_m] = [comb_ipt]
                else:
                    vector_ipt[name_m].append(comb_ipt)

        all_score = []
        # Calculate the score for each triplet
        for name_m in vector_ipt:
            ipt_AB = torch.cat(vector_ipt[name_m], dim=0)
            sum_ipt = ipt_AB.mean()
            triplet_ipt[name_m] = sum_ipt.view(-1, 1)
            all_score.append(sum_ipt.view(-1))

        # Get the threshold by ranking ipt
        if self.last_current_rank is None:
            mask_threshold = torch.kthvalue(torch.cat(all_score), k=self.total_num - curr_rank)[0].item()
        else:
            print(curr_rank)
            if self.last_current_rank > curr_rank:
                mask_threshold = torch.kthvalue(torch.cat(all_score), k=self.last_current_rank - curr_rank)[0].item()
            else:
                return 0

        # Mask the unimportant triplets
        with torch.no_grad():
            for n, p in model.named_parameters():
                if f"my_l.{self.adapter_name}" in n and p.requires_grad:
                    name_m = n.replace("_l", "%s")
                    if triplet_ipt[name_m] <= mask_threshold:
                        p.requires_grad = False
                        p.data.fill_(1.0)
                elif f"my_r.{self.adapter_name}" in n and p.requires_grad:
                    name_m = n.replace("_r", "%s")
                    if triplet_ipt[name_m] <= mask_threshold:
                        p.requires_grad = False
                        p.data.fill_(1.0)
        return mask_threshold

    def update_and_allocate(self, model, global_step):
        # # Update the importance score and allocate the budget
        if global_step < self.peft_config.tfinal:
            self.update_ipt(model)
        curr_rank, mask_ind = self.budget_schedule(global_step)
        # Allocate the budget according to importance scores
        if mask_ind:
            mask_threshold = self.mask_to_budget(model, curr_rank)
            self.last_current_rank = curr_rank
            print("\n")
            print("curr_rank" * 10)
            print(curr_rank)
            # print(all)
            print("curr_rank" * 10)
        else:
            mask_threshold = None
        return curr_rank, mask_threshold
