# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, List, Optional

import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

def sparsify(tensor, sparsity):
    with torch.no_grad():
        k = int(sparsity * tensor.numel())
        threshold = torch.topk(torch.abs(tensor).flatten(), k, largest=True).values.min()
        tensor = torch.where(torch.abs(tensor) >= threshold, tensor, torch.zeros_like(tensor))
    return tensor

class LinearMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    normalize: bool
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        keys = list(tensors.keys())

        tensors = [tensors[key] for key in keys]
        weights = [self.tensor_parameters[key]["weight"] for key in keys]

        rectify_embed_sizes(self.weight_info, tensors)

        unique_shapes = set(t.shape for t in tensors)
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}"
            )

        tensors = torch.stack(tensors, dim=0)
        weights = torch.tensor(weights, dtype=tensors.dtype, device=tensors.device)
        while len(weights.shape) < len(tensors.shape):
            weights.unsqueeze_(-1)

        # print(tensors.shape)
        # print(weights)

        num_models = weights.size(0)
        if num_models == 2:
            res = (weights * tensors).sum(dim=0)
            print("naive merge")
            if self.normalize:
                res = res / weights.sum(dim=0)
        
        # if num_models == 3:
        #     print("task vector")
        #     delta_1 = tensors[1] - tensors[0]
        #     delta_2 = tensors[2] - tensors[0]
        #     # sparsity = 0.3
        #     delta_1 = sparsify(delta_1, 0.3)
        #     delta_2 = sparsify(delta_2, 0.3)

        #     res = weights[0]*tensors[0] + weights[1]*delta_1 + weights[2]*delta_2

            # print("before", torch.sum(pruning_vector))
            # with torch.no_grad():

            #     sign_pruning = torch.sign(pruning_vector)
            #     sign_reflective = torch.sign(reflective_vector)

            #     sign_mismatch = sign_pruning != sign_reflective
            #     sign_mismatch = torch.logical_and(sign_mismatch, torch.abs(sign_pruning) == 1)
            #     sign_mismatch = torch.logical_and(sign_mismatch, torch.abs(sign_reflective) == 1)
            #     mismatch_count = torch.sum(sign_mismatch).item()

            #     total_count = pruning_vector.numel()

            #     # 计算不一致比例
            #     mismatch_ratio = mismatch_count / total_count
            #     print("mismatch_ratio:",mismatch_ratio)
            #     pruning_vector[sign_mismatch] *= 0
            #     # pruning_vector = torch.where(torch.sign(pruning_vector) != torch.sign(reflective_vector), 
            #     #                  torch.zeros_like(pruning_vector), pruning_vector)
            #     # pruning_vector = torch.where(torch.sign(pruning_vector) != torch.sign(reflective_vector), 
            #     #                  torch.zeros_like(pruning_vector), pruning_vector)
            #     # pruning_vector = torch.where(torch.sign(pruning_vector) != torch.sign(reflective_vector), 
            #     #                 0.5*pruning_vector, pruning_vector)
            #     print("after", torch.sum(pruning_vector))
            # print("-"*10)
            # res = weights[0]*tensors[0] + weights[1]*pruning_vector

            # print("task vector merge")
            
        if num_models == 5:

            #0 base model
            #1 model 1
            #2 model 2
            #3 model 1 grad
            #4 model 2 grad 

            delta_1 = tensors[1] - tensors[0]
            delta_2 = tensors[2] - tensors[0]

            # sparsity = 0.3
            # delta_1 = sparsify(delta_1, 0.3)
            # delta_2 = sparsify(delta_2, 0.3)

            # fisher_1 = torch.square(tensors[3])
            # fisher_2 = torch.square(tensors[4])

            # fisher_1 = sparsify(fisher_1, 0.3)
            # fisher_2 = sparsify(fisher_2, 0.3)









        # weights: 1.0, 0.75, -0.75, 1.0
        # tensors: N * (E, E)
                
        # a = (weights[0:3] * tensors[0:3].to(torch.float32)).sum(dim=0).to(torch.bfloat16)
        # b = (weights[0]*tensors[0].to(torch.float32) + weights[1]*(tensors[1].to(torch.float32) - tensors[2].to(torch.float32))).to(torch.bfloat16)
        # print(a.shape, b.shape)
        # print(torch.sum(a), torch.sum(b))

        if num_models == 4:
            # print(weights)
            # print(weights[0],weights[1])
            # model 0: base model,
            # model 1: simplified model,
            # model 2: common model,
            # model 3: enhanced model,
            print("constraint merge")
            pruning_vector = tensors[1] - tensors[2]
            # print(tensors[1].shape)
            reflective_vector = tensors[3] - tensors[2]

            sparsity = 0.3
            pruning_vector = sparsify(pruning_vector, sparsity)
            reflective_vector = sparsify(reflective_vector, 0.05)
            


            print("before", torch.sum(pruning_vector))
            with torch.no_grad():

                sign_pruning = torch.sign(pruning_vector)
                # print(pruning_vector)
                # print(sign_pruning)
                sign_reflective = torch.sign(reflective_vector)

                sign_mismatch = sign_pruning != sign_reflective
                sign_mismatch = torch.logical_and(sign_mismatch, torch.abs(sign_pruning) == 1)
                sign_mismatch = torch.logical_and(sign_mismatch, torch.abs(sign_reflective) == 1)


                mismatch_count = torch.sum(sign_mismatch).item()

                total_count = pruning_vector.numel()

                # 计算不一致比例
                mismatch_ratio = mismatch_count / total_count
                print("mismatch_ratio:",mismatch_ratio)
                pruning_vector[sign_mismatch] *= 0
                # pruning_vector = torch.where(torch.sign(pruning_vector) != torch.sign(reflective_vector), 
                #                  torch.zeros_like(pruning_vector), pruning_vector)
                # pruning_vector = torch.where(torch.sign(pruning_vector) != torch.sign(reflective_vector), 
                #                  torch.zeros_like(pruning_vector), pruning_vector)
                # pruning_vector = torch.where(torch.sign(pruning_vector) != torch.sign(reflective_vector), 
                #                 0.5*pruning_vector, pruning_vector)
                print("after", torch.sum(pruning_vector))
            print("-"*10)
            res = weights[0]*tensors[0] + weights[1]*pruning_vector


        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class LinearMerge(MergeMethod):
    def name(self) -> str:
        return "linear"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Linear"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/2203.05482"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="normalize", required=False, default_value=True),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return LinearMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            normalize=parameters["normalize"],
            weight_info=output_weight,
        )
