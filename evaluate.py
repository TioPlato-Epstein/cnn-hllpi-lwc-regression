"""Training loop for the model."""
import os
from typing import Dict

import neptune
import torch
from matplotlib import pyplot as plt
from matplotlib import ticker
from torch.utils import data

import utils


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
  colors = ["green" if abs(target - predict) <= error_toleration else "blue"
            for target, predict in zip(target_list, predict_list)]
  plt.scatter(target_list, predict_list, c=colors, marker="x")
  plt.grid(visible=True, linestyle="--", which="major", axis="both")
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
    title: str,
    experiment_dir: str,
    config_filename: str,
    archive_filename: str,
    test_dataloader: data.DataLoader,
    device: torch.device,
    logger: neptune.Run,
    error_toleration: float = 0.1,
  ) -> Dict[str, float]:
  model, criterion, _, _ = utils.construct_experiment_items(
    experiment_dir, config_filename, device, logger)
  archive_path = os.path.join(experiment_dir, archive_filename)
  model.load_state_dict(torch.load(archive_path))
  model.eval()
  total_loss = 0.0
  total_near_to = 0
  predicted_list, target_list = [], []
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
    title, target_list, predicted_list, error_toleration)
  logger["plots/target_vs_predicted"].upload(result_figure)
  return {"loss": epoch_loss, "accuracy": near_to_accuracy}
