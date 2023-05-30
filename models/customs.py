"""Custom models for the project."""
import torch
from torch import nn

class CustomVggNet(nn.Module):
  """Custom VGG model for the project."""
  def __init__(self) -> None:
    super().__init__()
    # input: B x 3 x 512 x 512
    self.vgg_block1 = nn.Sequential(
      nn.Conv2d(3, 16, 4, 2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.Conv2d(16, 16, 4, 2),
      nn.BatchNorm2d(16),
      nn.Dropout2d(0.1),
      nn.ReLU(),
      nn.AvgPool2d(2),
    )
    self.vgg_block2 = nn.Sequential(
      nn.Conv2d(16, 32, 4, 2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 4, 2),
      nn.BatchNorm2d(32),
      nn.Dropout2d(0.1),
      nn.ReLU(),
      nn.AvgPool2d(2),
    )
    self.vgg_block3 = nn.Sequential(
      nn.Conv2d(32, 64, 2),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, 2),
      nn.BatchNorm2d(64),
      nn.Dropout2d(0.1),
      nn.ReLU(),
      nn.AvgPool2d(2),
    )
    self.convolutional_layers = nn.Sequential(
      self.vgg_block1,
      self.vgg_block2,
      self.vgg_block3,
    )
    self.flatten_layer = nn.Flatten()
    self.full_connection_layers = nn.Sequential(
      nn.Dropout(0.4),
      nn.Linear(256, 16),
      nn.Dropout(0.5),
      nn.ReLU(),
      nn.Linear(16, 1),
      nn.ReLU(),
    )

  def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
    feature_map = self.convolutional_layers(image_batch)
    feature_vector = self.flatten_layer(feature_map)
    return self.full_connection_layers(feature_vector).reshape(-1)
