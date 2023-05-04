import torch
from dataclasses import dataclass

@dataclass
class Options:
    normalize: bool = True
    batch_size: int = 32
    input_size: int = 40000
    sample_rate: int = 8000
    n_epochs: int = 10
    lr: float = 0.0001
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")