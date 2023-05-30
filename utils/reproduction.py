"""Reproduction utilities."""
import torch
import numpy as np
from torch import cuda
from torch.backends import cudnn


def ensure_reproduction(seed: int = 0) -> None:
  """
  Ensure reproducibility of the experiment.

  Args:
    - `seed`: The random seed to use for reproducibility.
  """
  np.random.seed(seed)
  torch.manual_seed(seed)  # The returned generator is ignored
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if cuda.is_available():
    cudnn.deterministic = True
    cudnn.benchmark = False
  print(f"Random seed set to {seed} and cudnn deterministic mode set to True.")
