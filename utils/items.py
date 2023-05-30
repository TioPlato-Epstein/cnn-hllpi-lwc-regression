"""This file contains the supported items for the experiments."""
import json
import os
from typing import Any, Dict, Optional, Tuple

import neptune
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

SUPPORTED_ITEMS = {
  "model": {
  },
  "criterion": {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
    "cross_entropy": nn.CrossEntropyLoss,
  },
  "optimizer": {
    "adam": optim.Adam,
    "sgd": optim.SGD,
  },
  "scheduler": {
    "step_lr": lr_scheduler.StepLR,
    "multi_step_lr": lr_scheduler.MultiStepLR,
  },
}


def register_model(model_name: str, model_cls: nn.Module) -> None:
  SUPPORTED_ITEMS["model"].setdefault(model_name, model_cls)


def _parse_json_config(
        config_path: str, logger: neptune.Run) -> Dict[str, Any]:
  with open(config_path, "r", encoding="utf-8") as config_file:
    config = json.load(config_file)
  logger["configurations"] = config
  return config


_ExperimentItems = Tuple[nn.Module, nn.Module, optim.Optimizer,
                         Optional[lr_scheduler._LRScheduler]]  # pylint: disable=protected-access


def _construct_items(
    device: torch.device,
    model_name: str,
    model_params: Dict[str, Any],
    criterion_name: str,
    criterion_params: Dict[str, Any],
    optimizer_name: Optional[str],
    optimizer_params: Optional[Dict[str, Any]],
    scheduler_name: Optional[str],
    scheduler_params: Optional[Dict[str, Any]],
  ) -> _ExperimentItems:
  model_cls = SUPPORTED_ITEMS["model"][model_name]
  model = model_cls(**model_params)
  criterion_cls = SUPPORTED_ITEMS["criterion"][criterion_name]
  criterion = criterion_cls(**criterion_params)
  optimizer = None
  if optimizer_name is not None:
    optimizer_cls = SUPPORTED_ITEMS["optimizer"][optimizer_name]
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
  scheduler = None
  if scheduler_name is not None:
    scheduler_cls = SUPPORTED_ITEMS["scheduler"][scheduler_name]
    scheduler = scheduler_cls(optimizer, **scheduler_params)
  return model.to(device), criterion.to(device), optimizer, scheduler


def construct_experiment_items(
    experiment_dir: str,
    config_filename: str,
    device: torch.device,
    logger: neptune.Run,
  ) -> _ExperimentItems:
  config_path = os.path.join(experiment_dir, config_filename)
  config = _parse_json_config(config_path, logger)
  return _construct_items(device, **config)
