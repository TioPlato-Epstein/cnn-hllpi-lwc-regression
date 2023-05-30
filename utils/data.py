"""Define dataset and config data load pipeline."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from torch.utils import data
from torchvision import datasets, utils


def remove_surrounding_border(image: Image.Image) -> Image.Image:
  crop_box = (0, 30, image.width, image.height)
  image = image.crop(crop_box)
  invert_image = ImageOps.invert(image)
  crop_box = invert_image.getbbox()
  image = image.crop(crop_box)
  crop_box = (10, 10, image.width - 10, image.height - 10)
  image = image.crop(crop_box)
  return image


class HollyLeafPolarizationImages(datasets.VisionDataset):
  """Dataset of Holly leaf polarization images.

  Args:

    root (str): Root directory of dataset.
    train (bool, optional): If True, creates dataset from training set,
      otherwise creates from test set. Default: True.
    transform (callable, optional): A function/transform that takes in an PIL
      image and returns a transformed version. E.g, ``transforms.RandomCrop``.
      Default: None.
    target_transform (callable, optional): A function/transform that takes in
      the target and transforms it. Default: None.
  """
  def __init__(
      self,
      root: str,
      train: bool = True,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
    ) -> None:
    super().__init__(
      root, transform=transform, target_transform=target_transform,
    )
    self.train = train
    self.images = []
    self.targets = []
    self._load_data()

  def _load_data(self) -> None:
    meta_folder = os.path.join(self.root, "meta")
    if self.train:
      meta_file = os.path.join(meta_folder, "train.csv")
      image_folder = os.path.join(self.root, "train")
    else:
      meta_file = os.path.join(meta_folder, "test.csv")
      image_folder = os.path.join(self.root, "test")
    meta = pd.read_csv(meta_file)
    for image_name, target in zip(meta.iloc[:, 0], meta.iloc[:, 1]):
      if not image_name.endswith(datasets.folder.IMG_EXTENSIONS):
        image_name += ".png"
      image_path = os.path.join(image_folder, image_name)
      image = Image.open(image_path).convert("RGB")

      image = remove_surrounding_border(image)
      target = torch.tensor(target, dtype=torch.float32)

      self.images.append(image)
      self.targets.append(target)

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    image, target = self.images[index], self.targets[index]
    if self.transform is not None:
      image = self.transform(image)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return image, target

  def extra_repr(self) -> str:
    return f"Is training dataset: {self.train}"


def compute_std_mean(
    data_dir: str,
    transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
  dataset = HollyLeafPolarizationImages(
    data_dir, train=True, transform=transform)
  tensor_images = torch.cat([image for image, _ in dataset], dim=1)
  return torch.std_mean(tensor_images, dim=(1, 2))


def prepare_dataloaders(
    data_dir: str,
    train_transform: Callable,
    test_transform: Callable,
    batch_sizes: Union[int, List[int]],
    num_workers: Union[int, List[int]],
    train_split_ratio: float,
  ) -> Dict[str, data.DataLoader]:
  if isinstance(batch_sizes, int):
    batch_sizes = [batch_sizes] * 3
  if isinstance(num_workers, int):
    num_workers = [num_workers] * 3

  train_dataset = HollyLeafPolarizationImages(
    data_dir, train=True, transform=train_transform)
  print(train_dataset)

  train_length = int(len(train_dataset) * train_split_ratio)
  val_length = len(train_dataset) - train_length
  train_dataset, val_dataset = data.random_split(
    train_dataset, [train_length, val_length])

  train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_sizes[0], shuffle=True,
    num_workers=num_workers[0], pin_memory=True, drop_last=True)
  val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_sizes[1],
    shuffle=False, num_workers=num_workers[1])

  test_dataset = HollyLeafPolarizationImages(
    data_dir, train=False, transform=test_transform)
  print(test_dataset)

  test_dataloader = data.DataLoader(
    test_dataset, batch_size=batch_sizes[2],
    shuffle=False, num_workers=num_workers[2])

  return {"train": train_dataloader,
          "val": val_dataloader,
          "test": test_dataloader}


def check_dataloaders(dataloaders: Dict[str, data.DataLoader]) -> None:
  for iterator in map(iter, dataloaders.values()):
    images, targets = next(iterator)
    grid = utils.make_grid(images, nrow=10)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    print(f"Targets: {targets}.")
