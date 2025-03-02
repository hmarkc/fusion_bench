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


@torch.no_grad()
def ties_merge(
    pretrained_model: nn.Module,
    finetuned_models: List[Dict[str, Tensor]],
    scaling_factor: float,
    threshold: float,
) -> nn.Module:    
    remove_keys = []
    merge_func = "sum"
    if isinstance(finetuned_models[0], nn.Module):
        finetuned_models = [deepcopy(model.state_dict(keep_vars=True)) for model in finetuned_models]

    ptm_check = pretrained_model.state_dict(keep_vars=True)

    # Compute the task vectors
    flat_ft = torch.vstack(
            [state_dict_to_vector(check, remove_keys) for check in finetuned_models]
        )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm

    # Perform TIES Merging
    merged_tv = ties_merging(
        tv_flat_checks,
        reset_thresh=threshold,
        merge_func=merge_func,
    )
    merged_check = flat_ptm + scaling_factor * merged_tv
    merged_state_dict = vector_to_state_dict(
        merged_check, ptm_check, remove_keys=remove_keys
    )

    # Load the merged state dict into the pretrained model
    pretrained_model.load_state_dict(merged_state_dict)
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


class FrankWolfeAlgorithm(
    CLIPClassificationMixin,
    ModelFusionAlgorithm,
    SimpleProfilerMixin,
):


    def __init__(self, 
                 merge_fn: str,
                 step_size: float,
                 max_iters: int,
                 dataset_size:int,
                 tasks: List[str] = [],
                 max_num_models: int = 100,
                 loss_fn: str = "cross_entropy",
                 init_weight: str = "",
                 scaling_factor: float = 1.,
                 threshold: int = 20,
                 **kwargs):
        """
        Initializes the TaskArithmeticAlgorithm with the given scaling factor.

        Args:
            scaling_factor (int): The factor by which the task vectors will be scaled before merging.
        """
        self.merger = merge_fn
        if merge_fn == "task_arithmetic":
            self.merge_fn = task_arithmetic_merge
        elif merge_fn == "ties":
            self.merge_fn = partial(ties_merge, threshold=threshold)
        # elif merge_fn == "concrete_ta":
        #     self.merge_fn = ConcreteTaskArithmeticAlgorithmForCLIP(
        #         instantiate(OmegaConf.load("config/method/concrete_subspace/clip_concrete_task_arithmetic.yaml"))
        #     )
        else:
            raise ValueError(f"Unsupported merge_fn: {merge_fn}")
        self.scaling_factor = scaling_factor
        
        self.init_weight = init_weight
        self.step_size = step_size
        self.max_iters = max_iters
        self.loss_fn = loss_fn
        self.tasks = tasks
        self.dataset_size = dataset_size
        self.max_num_models = max_num_models
        super().__init__(**kwargs)


    def on_frank_wolfe_iteration_start(self):
        self.setup_zero_shot_classification_head()
    
    @functools.cache
    def get_shuffled_loader_iter(self, task: str):
        if self.loss_fn == "cross_entropy":
            # get dataloader kwargs
            dataloader_kwargs = self._dataloader_kwargs.copy()
            dataloader_kwargs["shuffle"] = True
            dataloader_kwargs["batch_size"] = 1

            # get the test dataset
            clip_dataset = CLIPDataset(
                self.modelpool.load_train_dataset(task), self.clip_processor
            )
            # create the dataloader
            loader = DataLoader(clip_dataset, **dataloader_kwargs)
            loader = self.fabric.setup_dataloaders(loader)
            return iter(InfiniteDataLoader(loader))
        elif self.loss_fn == "entropy":
            return super().get_shuffled_test_loader_iter(
                task,
                batch_size=1,
            )
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_fn}")

    
    def frank_wolfe_iteration(self, merged_model):
        
        merged_model.train()
        # zero the gradients
        for name, param in merged_model.named_parameters():
            param.requires_grad = True
            param.grad = None
        
        if self.loss_fn == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss()
        elif self.loss_fn == "entropy":
            loss_fn = entropy_loss
        avg_loss = defaultdict(list)
        tasks = self.tasks if self.tasks else self.modelpool.model_names
        for task in tasks:
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
        pretrained_model = modelpool.load_model("_pretrained_")
        
        if self.init_weight:
            if self.init_weight == 'base':
                log.info("Initializing the merged model with the base model")
                merged_model = pretrained_model
            else:
                log.info("Initializing the merged model with the initial weight")
                if isinstance(self.init_weight, str):
                    # self.config.weights is a path to a saved tensor
                    layer_wise_weight = load_tensor_from_file(self.init_weight)
                else:
                    raise ValueError(f"Unsupported weights format: {self.init_weight}")

                merged_model = LayerWiseMergedModel(
                    layer_wise_weight=layer_wise_weight,
                    pretrained_model=modelpool.load_model("_pretrained_"),
                    finetuned_models=list(finetuned_models.values()),
                    clamp_weights=False,
                    tie_weights=True,
                    strict=False,
                ).cuda()
                merged_model = merged_model.merge_and_unload()
        else:
            log.info("Initializing the merged model with merge function")
            merged_model = self.merge_fn(
                pretrained_model=modelpool.load_model("_pretrained_"),
                finetuned_models=list(finetuned_models.values()),
                scaling_factor=self.scaling_factor
            ).cuda()
        # merged_model = self.fabric.setup(merged_model)

        initial_model = modelpool.load_model("_pretrained_")
        initial_model.load_state_dict(deepcopy(merged_model.state_dict()))
        finetuned_models['initial'] = initial_model
        for step_idx in (
            pbar := tqdm(
                range(self.max_iters if not self.is_debug_mode else 1),
                ("[DEBUG MODE] " if self.is_debug_mode else "")
                + "Frank-Wolfe Merging",
                dynamic_ncols=True,
            )
        ):
            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()
            gradients = self.frank_wolfe_iteration(merged_model.cuda())
            grad_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients.values()]))

            torch.set_grad_enabled(False)
            # Find the task vector with the most alignment to the gradient
            model_to_merge_dict = {}
            min_alignment = {}
            min_idx = {}
            for model_name, model_to_merge in finetuned_models.items():
                model_to_merge = model_to_merge.to('cpu').state_dict()
                for param_name, param_value in model_to_merge.items():
                    # caclulate consine similarity
                    grad = gradients[param_name]
                    ckpt = model_to_merge[param_name]
                    param_alignment = torch.dot(grad.flatten(), ckpt.flatten()) / (torch.norm(grad) * torch.norm(ckpt))
                    if param_name not in min_alignment or param_alignment < min_alignment[param_name]:
                        min_alignment[param_name] = param_alignment
                        if min_alignment[param_name] < 0:
                            model_to_merge_dict[param_name] = param_value
                            min_idx[param_name] = model_name
                        else:
                            model_to_merge_dict[param_name] = torch.zeros_like(param_value)
                            # model_to_merge_dict[param_name] = merged_model.state_dict()[param_name].to('cpu')
            chosen_model = {model_name: 0 for model_name in finetuned_models.keys()}
            for k in min_idx.values():
                chosen_model[k] += 1

            # Determine step size
            step = 2 / (step_idx + 2) * self.step_size
            
            # print iteration information
            log.info(f"Iteration {step_idx+1}, Task Vector: {chosen_model}, Gradient Norm: {grad_norm:.6f}, Step Size: {step:.6f}, Negative Alignment: {sum([1 for v in min_alignment.values() if v < 0])}/{len(min_alignment)}")
            
            merged_model = self.merge_fn(
                    pretrained_model=merged_model.to('cpu'),
                    finetuned_models=[model_to_merge_dict],
                    scaling_factor=step*self.scaling_factor,
                )
    
        torch.set_grad_enabled(False)
        merged_model = merged_model.cuda().eval()
        return merged_model
