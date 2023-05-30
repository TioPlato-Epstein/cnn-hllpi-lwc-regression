"""Training loop for the model."""
import os
from typing import Any, Dict, Optional, Tuple

import neptune
import torch
import tqdm
from matplotlib import pyplot as plt
from matplotlib import ticker
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
    epochs: int,
    val_every: int,
  ) -> Dict[str, Any]:
  model, criterion, optimizer, scheduler = utils.construct_experiment_items(
    experiment_dir, config_filename, device, logger)
  best_loss = float("inf")
  archive_filename = None
  for epoch in tqdm.tqdm(range(epochs)):
    _train_step(
      model, criterion, optimizer, scheduler, train_dataloader, device, logger)
    if epoch % val_every == 0 or epoch == epochs - 1:
      _validate_step(model, criterion, val_dataloader, device, logger)
      best_loss, archive_filename = _refresh_model_archive(
        best_loss, model, experiment_dir, epoch, logger, archive_filename)
  logger["checkpoints"].upload(os.path.join(experiment_dir, archive_filename))
  print(f"Best model archive: {archive_filename}.")
  return {"best_loss": best_loss, "best_model": archive_filename}


def _plot_result_scatter(
    title: str,
    target_list: list[float],
    predict_list: list[float],
    error_toleration: float,
  ):
  result_figure = plt.figure()
  axis = plt.gca()
  axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
  axis.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
  plt.scatter(target_list, predict_list, marker="x")
  plt.title(title)
  plt.xlabel("Target")
  plt.ylabel("Predicted")
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.plot([0, 1], [0, 1], color="red")
  plt.plot(
    [0, 1], [0 + error_toleration, 1 + error_toleration],
    color="orange", linestyle="--")
  plt.plot(
    [0, 1], [0 - error_toleration, 1 - error_toleration],
    color="orange", linestyle="--")
  return result_figure


def test_model(
    experiment_dir: str,
    config_filename: str,
    archive_filename: str,
    test_dataloader: data.DataLoader,
    device: torch.device,
    logger: neptune.Run,
    error_toleration: float = 0.05,
  ) -> Dict[str, float]:
  model, criterion, _, _ = utils.construct_experiment_items(
    experiment_dir, config_filename, device, logger)
  archive_path = os.path.join(experiment_dir, archive_filename)
  model.load_state_dict(torch.load(archive_path))
  model.eval()
  predicted_list = []
  target_list = []
  total_loss = 0.0
  total_near_to = 0
  with torch.no_grad():
    for images, targets in test_dataloader:
      images, targets = images.to(device), targets.to(device)
      output = model(images)
      loss = criterion(output, targets)
      total_loss += loss.item()
      near_to = torch.abs(output - targets) < error_toleration
      total_near_to += torch.sum(near_to).item()
      target_list.extend(targets.tolist())
      predicted_list.extend(output.tolist())
  epoch_loss = total_loss / len(test_dataloader)
  logger["metrics/test_loss"] = epoch_loss
  near_to_accuracy = total_near_to / len(test_dataloader.dataset)
  logger["metrics/accuracy"] = near_to_accuracy
  result_figure = _plot_result_scatter(
    f"{model._get_name()}", target_list,  # pylint: disable=protected-access
    predicted_list, error_toleration)
  logger["plots/target_vs_predicted"].upload(result_figure)
  return {"loss": epoch_loss, "accuracy": near_to_accuracy}
