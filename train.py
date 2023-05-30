"""Training loop for the model."""
import os
from typing import Any, Dict, Optional, Tuple

import neptune
import torch
import tqdm
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

import utils


def _train_step(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[lr_scheduler._LRScheduler],
    dataloader: data.DataLoader,
    device: torch.device,
    logger: neptune.Run,
  ) -> None:
  model.train()
  total_loss = 0.0
  for images, targets in dataloader:
    images, targets = images.to(device), targets.to(device)
    optimizer.zero_grad()
    output = model(images)
    loss = criterion(output, targets)
    total_loss += loss.item()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
      scheduler.step()
      logger["metrics/lr"].append(scheduler.get_last_lr()[0])
  epoch_loss = total_loss / len(dataloader)
  logger["metrics/train_loss"].append(epoch_loss)


def _validate_step(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: data.DataLoader,
    device: torch.device,
    logger: neptune.Run,
  ) -> None:
  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for images, targets in dataloader:
      images, targets = images.to(device), targets.to(device)
      output = model(images)
      loss = criterion(output, targets)
      total_loss += loss.item()
  epoch_loss = total_loss / len(dataloader)
  logger["metrics/val_loss"].append(epoch_loss, wait=True)


def _refresh_model_archive(
    best_loss: float,
    model: nn.Module,
    experiment_dir: str,
    epoch: int,
    logger: neptune.Run,
    last_filename: Optional[str],
  ) -> Tuple[float, str]:
  current_loss = logger["metrics/val_loss"].fetch_last()
  if current_loss < best_loss:
    best_loss = current_loss
    best_filename = (f"{model._get_name().lower()}"  # pylint: disable=protected-access
                     f"_step{epoch}_loss{current_loss:.4f}.pth")
    if last_filename is not None and last_filename != best_filename:
      last_filepath = os.path.join(experiment_dir, last_filename)
      os.remove(last_filepath)
    best_filepath = os.path.join(experiment_dir, best_filename)
    torch.save(model.state_dict(), best_filepath)
    return best_loss, best_filename
  return best_loss, last_filename


def train_model(
    experiment_dir: str,
    config_filename: str,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    device: torch.device,
    logger: neptune.Run,
    max_epochs: int,
    validation_freq: int,
  ) -> Dict[str, Any]:
  model, criterion, optimizer, scheduler = utils.construct_experiment_items(
    experiment_dir, config_filename, device, logger)
  best_loss = float("inf")
  archive_filename = None
  for epoch in tqdm.tqdm(range(max_epochs)):
    _train_step(
      model, criterion, optimizer, scheduler, train_dataloader, device, logger)
    if epoch % validation_freq == 0 or epoch == max_epochs - 1:
      _validate_step(model, criterion, val_dataloader, device, logger)
      best_loss, archive_filename = _refresh_model_archive(
        best_loss, model, experiment_dir, epoch, logger, archive_filename)
  logger["checkpoints"].upload(os.path.join(experiment_dir, archive_filename))
  print(f"Best model archive: {archive_filename}.")
  return {"best_loss": best_loss, "best_model": archive_filename}
