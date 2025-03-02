"""
This script contains the general implementation of the Task Arithmetic method.

http://arxiv.org/abs/2212.04089
"""

import logging
import os
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Mapping, TypeVar, Union, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial
import functools
from collections import OrderedDict
from omegaconf import OmegaConf
from transformers import CLIPVisionModel

from .tie_merging_utils import *
from .clip_concrete_adamerging import *
from .clip_concrete_task_arithmetic import *

import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench.compat.method import ModelFusionAlgorithm
from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.mixins import CLIPClassificationMixin
from fusion_bench.compat.modelpool import ModelPool
from fusion_bench.mixins.lightning_fabric import LightningFabricMixin
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel,
    get_layer_wise_weights,
)
from fusion_bench.utils.data import load_tensor_from_file
from fusion_bench.utils.type import TorchModelType

if TYPE_CHECKING:
    from fusion_bench.programs.fabric_fusion_program import FabricModelFusionProgram

from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)
from fusion_bench.utils.type import StateDictType
from fusion_bench.utils import instantiate

log = logging.getLogger(__name__)


@torch.no_grad()
def task_arithmetic_merge(
    pretrained_model: nn.Module,
    finetuned_models: List[Dict[str, Tensor]],
    scaling_factor: float,
    inplace: bool = True,
) -> nn.Module:
    """
    Merges the task vectors from multiple fine-tuned models into a single pre-trained model.

    Args:
        pretrained_model (nn.Module): The pre-trained model to which the task vectors will be added.
        finetuned_models (List[nn.Module]): A list of fine-tuned models from which task vectors will be calculated.
        scaling_factor (float): A factor by which the task vectors will be scaled before merging.
        inplace (bool, optional): If True, the pre-trained model will be modified in place.
                                  If False, a copy of the pre-trained model will be modified. Defaults to True.

    Returns:
        nn.Module: The pre-trained model with the merged task vectors.
    """
    if not inplace:
        pretrained_model = deepcopy(pretrained_model)
    if isinstance(finetuned_models[0], nn.Module):
        finetuned_models = [deepcopy(model.state_dict(keep_vars=True)) for model in finetuned_models]
    task_vector: StateDictType = None
    # Calculate the total task vector
    for model in finetuned_models:
        if task_vector is None:
            task_vector = state_dict_sub(
                model,
                pretrained_model.state_dict(keep_vars=True),
            )
        else:
            task_vector = state_dict_add(
                task_vector,
                state_dict_sub(
                    model,
                    pretrained_model.state_dict(keep_vars=True),
                ),
            )
    # scale the task vector
    task_vector = state_dict_mul(task_vector, scaling_factor)
    # add the task vector to the pretrained model
    state_dict = state_dict_add(
        pretrained_model.state_dict(keep_vars=True), task_vector
    )
    pretrained_model.load_state_dict(state_dict)
    return pretrained_model


def entropy_loss(logits: Tensor, pred = None, eps: float = 1e-8) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.
        eps (float): A small value to avoid log(0). Default is 1e-8.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    # Ensure the logits tensor has 2 dimensions
    assert (
        logits.dim() == 2
    ), f"Expected logits to have 2 dimensions, found {logits.dim()}, {logits.size()=}"

    # Compute the softmax probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute the entropy loss
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()


class FrankWolfeAdamergingAlgorithm(
    CLIPClassificationMixin,
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
):
    def __init__(self, 
                 max_iters: int,
                 dataset_size:int,
                 ada_iters: int,
                 ada_coeff: float,
                 merge_fn: str,
                 max_num_models: int = 100,
                 step_size: float = 0.3,
                 tasks: List[str] = [],
                 init_weight: str = "",
                 **kwargs):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor.

        Args:
            step_size (int): The factor by which the task vectors will be scaled before merging.
        """
        self.merge_fn = merge_fn
        
        self.init_weight = init_weight
        self.max_iters = max_iters
        self.ada_iters = ada_iters
        self.ada_coeff = ada_coeff
        self.tasks = tasks
        self.step_size = step_size
        self.dataset_size = dataset_size
        self.max_num_models = max_num_models
        super().__init__(**kwargs)


    def on_frank_wolfe_iteration_start(self):
        self.setup_zero_shot_classification_head()
    
    @functools.cache
    def get_shuffled_loader_iter(self, task: str, batch_size: int = 1):
        # get dataloader kwargs
        dataloader_kwargs = self._dataloader_kwargs.copy()
        dataloader_kwargs["shuffle"] = True
        dataloader_kwargs["batch_size"] = batch_size

        # get the test dataset
        clip_dataset = CLIPDataset(
            self.modelpool.load_train_dataset(task), self.clip_processor
        )
        # create the dataloader
        loader = DataLoader(clip_dataset, **dataloader_kwargs)
        loader = self.fabric.setup_dataloaders(loader)
        return iter(InfiniteDataLoader(loader))
    

    @functools.cache
    def get_shuffled_test_loader_iter(self, task: str, batch_size: int = 1):
        return super().get_shuffled_test_loader_iter(
                task,
                batch_size=batch_size
        )
    
    
    def run_adamerging(self, module: TaskWiseMergedModel):
        optimizer = torch.optim.Adam(
            [module.merge_weight], lr=1e-3
        )
        module, optimizer = self.fabric.setup(module, optimizer)
        module.train()
        for step_idx in (
            pbar := tqdm(
                range(self.ada_iters),
                "AdaMerging (2/2)",
                dynamic_ncols=True,
                disable=not self.fabric.is_global_zero,
            )
        ):
            with self.profile("merge weights"):
                module.merge_weights()

            metrics = {}
            total_loss = None
            tasks = self.modelpool.model_names if self.tasks == [] else self.tasks
            for task in tasks:
                with self.profile("data loading"):
                    batch = next(self.get_shuffled_test_loader_iter(task, batch_size=16))
                    # NOTE: The labels are not allowed to be used during test-time adaptation
                    images = batch[0]
                with self.profile("forward pass"):
                    logits = self.compute_logits(module, images, task)
                    loss = entropy_loss(logits)
                    total_loss = loss if total_loss is None else total_loss + loss

            optimizer.zero_grad()
            with self.profile("compute grad"):
                self.fabric.backward(total_loss)

            with self.profile("base optimizer step"):
                optimizer.step()

            metrics.update({"train/loss": loss.item()})
            self.fabric.log_dict(metrics, step=step_idx)
            pbar.set_postfix(metrics)
        return module
    
    
    def frank_wolfe_iteration(self, merged_model, task):
        
        merged_model.train()
        # zero the gradients
        requires_grad_dict = {}
        for name, param in merged_model.named_parameters():
            requires_grad_dict[name] = param.requires_grad
            param.requires_grad = True
            param.grad = None
        
        loss_fn = nn.CrossEntropyLoss()
        avg_loss = defaultdict(list)
        # tasks = self.tasks if self.tasks else self.modelpool.model_names
        # for task in tasks:
        log.info(f"Processing task {task}")
        for _ in range(self.dataset_size):
            with self.profile("data loading"):
                batch = next(self.get_shuffled_loader_iter(task))
            with self.profile("forward pass"):
                logits = self.compute_logits(merged_model, batch[0], task)
                loss = loss_fn(logits, batch[1]) / (self.dataset_size * len(self.modelpool.model_names))
            with self.profile("backward pass"):
                # self.fabric.backward(loss, retain_graph=True)
                loss.backward()
            avg_loss[task].append(loss.item())
        
        # calculate the loss
        avg_loss = {task: sum(losses) / len(losses) for task, losses in avg_loss.items()}
        log.info(f"Average Loss: {avg_loss}, Total Loss: {sum(avg_loss.values()) / len(avg_loss)}")
        
        gradients = {name: param.grad.clone().to('cpu') for name, param in merged_model.named_parameters() if param.requires_grad}
        for name, param in merged_model.named_parameters():
            param.requires_grad = requires_grad_dict[name]
            param.grad = None
        merged_model.eval()
        
        return gradients

    def run(self, modelpool: HuggingFaceClipVisionPool):
        log.info("Fusing models using FW merging.")
        self.modelpool = modelpool
        self.log_hyperparams(self.config)
        self.on_frank_wolfe_iteration_start()

        assert modelpool.has_pretrained, "Pretrained model is required."
        finetuned_models = {name: modelpool.load_model(name) for name in modelpool.model_names[:self.max_num_models]}
        
        if self.init_weight == 'base' or self.init_weight == '':
            merged_model = modelpool.load_model("_pretrained_")
        else:
            log.info("Initializing the merged model with the initial weight")
            if isinstance(self.init_weight, str):
                # self.config.weights is a path to a saved tensor
                layer_wise_weight = load_tensor_from_file(self.init_weight)
            else:
                raise ValueError(f"Unsupported weights format: {self.init_weight}")

            pretrained_model = modelpool.load_model("_pretrained_")
            layerwise_merged_model = LayerWiseMergedModel(
                layer_wise_weight=layer_wise_weight,
                pretrained_model=pretrained_model,
                finetuned_models=list(finetuned_models.values())[:self.max_num_models],
                clamp_weights=False,
                tie_weights=True,
                strict=False,
            ).cuda()
            merged_model = layerwise_merged_model.merge_and_unload()

        initial_model = modelpool.load_model("_pretrained_")
        self.set_requires_grad(merged_model, initial_model)
        # initial_model.load_state_dict(deepcopy(merged_model.state_dict()))
        # finetuned_models['initial'] = initial_model
        for step_idx in (
            pbar := tqdm(
                range(self.max_iters if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Frank-Wolfe Merging",
                dynamic_ncols=True,
            )
        ):

            # Find the task vector with the most alignment to the gradient
            models_dict_to_merge = []
            model_to_merge_names = []
            tasks = self.tasks if self.tasks else self.modelpool.model_names
            for task in tasks:
                torch.set_grad_enabled(True)
                torch.cuda.empty_cache()
                gradients = self.frank_wolfe_iteration(merged_model.cuda(), task)
                grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients.values()]))

                torch.set_grad_enabled(False)
                min_inner_product = float("inf")
                min_model = None 
                min_model_name = None
                for model_name, model_to_merge in finetuned_models.items():
                    model_to_merge = model_to_merge.to('cpu').state_dict()
                    inner_product_sum = 0
                    for param_name, param_value in model_to_merge.items():
                        # caclulate consine similarity
                        grad = gradients[param_name]
                        ckpt = model_to_merge[param_name]
                        param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / (torch.norm(grad) * torch.norm(ckpt))
                        inner_product_sum += param_alignment
                    if inner_product_sum < min_inner_product and model_name not in model_to_merge_names:
                        min_inner_product = inner_product_sum
                        min_model = deepcopy(model_to_merge)
                        min_model_name = model_name
                # if min_model is None:
                #     # fill with zero parameters
                #     min_model = {name: torch.zeros_like(param) for name, param in model_to_merge.items()}
                #     min_model_name = 'zero'
                models_dict_to_merge.append(min_model)
                model_to_merge_names.append(min_model_name)
                
                if len(models_dict_to_merge) >= len(self.modelpool.model_names) or len(models_dict_to_merge) >= self.max_num_models:
                    log.info(f"Breaking at {len(models_dict_to_merge)}")
                    break
            
            # print iteration information
            log.info(f"Iteration {step_idx+1}, Task Vector: {model_to_merge_names}, Gradient Norm: {grad_norm:.6f}")
            
            if self.merge_fn == 'adamerging':
                models_to_merge = [modelpool.load_model('_pretrained_') for _ in range(len(models_dict_to_merge))]
                layer_wise_weight = get_layer_wise_weights(
                    num_models=len(models_to_merge),
                    num_layers=len(
                        tuple(
                            filter(lambda p: p.requires_grad, models_to_merge[0].parameters())
                        )
                    ),
                    init_values=self.ada_coeff if step_idx > 0 else 0.3,
                )
                for model_to_merge, model_to_merge_dict in zip(models_to_merge, models_dict_to_merge):
                    model_to_merge.load_state_dict(model_to_merge_dict)
                layerwise_merged_model = LayerWiseMergedModel(
                    layer_wise_weight=layer_wise_weight,
                    pretrained_model=merged_model.to('cpu'),
                    finetuned_models=models_to_merge,
                    clamp_weights=False,
                    tie_weights=True,
                    strict=False,
                ).cuda()
                torch.set_grad_enabled(True)
                layerwise_merged_model = self.run_adamerging(layerwise_merged_model)
                torch.set_grad_enabled(False)
                with torch.no_grad():
                    merged_model = layerwise_merged_model.merge_and_unload()
                    self.set_requires_grad(merged_model, initial_model)
                del models_to_merge, layerwise_merged_model, layer_wise_weight, models_dict_to_merge
            else:
                step = 2 / (step_idx + 2) * self.step_size if step_idx > 0 else 1
                merged_model = task_arithmetic_merge(merged_model.to('cpu'), models_dict_to_merge, 0.3*step)
                del models_dict_to_merge
                
    
        torch.set_grad_enabled(False)
        merged_model = merged_model.cuda().eval()
        return merged_model

    def set_requires_grad(self, merged_model, initial_model):
        for name, param in initial_model.named_parameters():
            for n, p in merged_model.named_parameters():
                if name == n:
                    p.requires_grad = param.requires_grad
