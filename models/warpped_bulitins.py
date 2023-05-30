"""This module contains the warpped bulit-in models from torchvision.models"""
import torch
from torch import nn
from torchvision import models

class AlexNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.bulit_in_alexnet = models.AlexNet(num_classes=1)

  def forward(self, images) -> torch.Tensor:
    output = self.bulit_in_alexnet(images)
    return output.reshape(-1)

class DenseNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.bulit_in_densenet = models.DenseNet(num_classes=1)

  def forward(self, images) -> torch.Tensor:
    output = self.bulit_in_densenet(images)
    return output.reshape(-1)
