"""Define dataset and config data load pipeline."""
import os
from typing import Any, Callable, Optional, Tuple, Union

import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from torch.utils import data
from torchvision import datasets, transforms, utils


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
    - `root` (str): Root directory of dataset.
    - `color_mode` (str): Color mode of images.
    - `train` (bool): Whether to load training or test dataset.
    - `transform` (Optional[Callable]): A function/transform that takes in an
      PIL image and returns a transformed version. E.g,
      `transforms.RandomCrop`.
    - `target_transform` (Optional[Callable]): A function/transform that takes
      in the target and transforms it.
    - `preprocessing_function` (Optional[Callable]): A function that takes in
      a PIL image and returns a transformed version. E.g,
      `torchvision.transforms.ToTensor`.
  """

  def __init__(
      self,
      root: str,
      color_mode: str = "RGB",
      train: bool = True,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      preprocessing_function: Optional[Callable] = None,
      transform_callback: Optional[Callable] = None,
    ) -> None:
    super().__init__(
      root, transform=transform, target_transform=target_transform,
    )
    self.color_mode = color_mode
    self.train = train
    self.preprocessing_function = preprocessing_function
    self.transform_callback = transform_callback
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
      image = Image.open(image_path).convert(self.color_mode)

      image = self.preprocessing_function(image)
      target = torch.tensor(target, dtype=torch.float32)

      self.images.append(image)
      self.targets.append(target)

  def __len__(self) -> int:
    return len(self.images)

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    image, target = self.images[index], self.targets[index]
    if self.transform is not None:
      image = self.transform(image)
    if self.transform_callback is not None:
      image = self.transform_callback(image)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return image, target

  def extra_repr(self) -> str:
    return f"Is training dataset: {self.train}"


def compute_normalize_arguments(
    directory: str,
    color_mode: str = "RGB",
    transform: Optional[Callable] = None,
    preprocessing_function: Optional[Callable] = None,
    abandonment_ratio: Optional[float] = None,
    image_size: Union[int, Tuple[int, int]] = (512, 512),
  ) -> Tuple[torch.Tensor, torch.Tensor]:
  dataset = HollyLeafPolarizationImages(
    directory, color_mode, train=True, transform=transform,
    transform_callback=transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
    ]),
    preprocessing_function=preprocessing_function)
  if abandonment_ratio is not None:
    remain = int(len(dataset) * (1 - abandonment_ratio))
    adandon = len(dataset) - remain
    dataset, _ = data.random_split(dataset, [remain, adandon])
  concate_list = []
  for i in range(len(dataset)):
    image, _ = dataset[i]
    concate_list.append(image)
  tensor_images = torch.cat(concate_list, dim=1)
  normalize_args = torch.std_mean(tensor_images, dim=(1, 2))
  print(f"Normalize arguments: {normalize_args}.")
  return normalize_args


def vision_dataloaders_from_directory(
    directory: str,
    color_mode: str = "RGB",
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    preprocessing_function: Optional[Callable] = remove_surrounding_border,
    abandonment_ratio: Optional[float] = None,
    image_size: Union[int, Tuple[int, int]] = (512, 512),
    batch_size: int = 10,
    num_workers: int = 0,
    shuffle: bool = True,
    split_ratio: Optional[float] = None,
    repr_datasets: bool = True,
  ) -> Any:
  # Set the split ratio to 0.8 if it is None
  if split_ratio is None:
    split_ratio = 0.8

  # Compute the normalize arguments
  normalize_args = compute_normalize_arguments(
    directory, color_mode, transform,
    preprocessing_function, abandonment_ratio, image_size,
  )

  # Create the train dataset
  origin_train_dataset = HollyLeafPolarizationImages(
    directory, color_mode, train=True, transform=transform,
    transform_callback=transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
      transforms.Normalize(*normalize_args),
    ]),
    target_transform=target_transform,
    preprocessing_function=preprocessing_function,
  )

  # Split the train dataset into train and validation datasets
  total_length = len(origin_train_dataset)
  abandon_length = 0
  if abandonment_ratio is not None:
    abandon_length = int(total_length * abandonment_ratio)
  remain_length = total_length - abandon_length
  train_length = int(remain_length * split_ratio)
  val_length = remain_length - train_length

  train_dataset, val_dataset, _ = data.random_split(
    origin_train_dataset, [train_length, val_length, abandon_length])
  if repr_datasets:
    print(f"Length of train dataset: {len(train_dataset)}.")
    print(f"Length of validation dataset: {len(val_dataset)}.")

  # Create the train and validation dataloaders
  train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle,
    num_workers=num_workers, pin_memory=True, drop_last=True)

  val_dataloader = data.DataLoader(
    val_dataset, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)

  # Create the test dataset
  test_dataset = HollyLeafPolarizationImages(
    directory, color_mode, train=False,
    transform_callback=transforms.Compose([
      transforms.Resize(image_size),
      transforms.ToTensor(),
      transforms.Normalize(*normalize_args),
    ]),
    target_transform=target_transform,
    preprocessing_function=preprocessing_function,
  )
  if repr_datasets:
    print(f"Length of test dataset: {len(test_dataset)}.")

  # Create the test dataloader
  test_dataloader = data.DataLoader(
    test_dataset, batch_size=batch_size,
    shuffle=False, num_workers=num_workers)

  # Return the three dataloaders
  return train_dataloader, val_dataloader, test_dataloader


def check_dataloaders(dataloaders: Tuple[data.DataLoader, ...]) -> None:
  for iterator in map(iter, dataloaders):
    images, targets = next(iterator)
    grid = utils.make_grid(images, nrow=10)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
    print(f"Targets: {targets}.")
