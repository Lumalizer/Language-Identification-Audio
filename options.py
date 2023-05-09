import torch
from dataclasses import dataclass


@dataclass
class Options:
    use_all_languages: bool
    normalize: bool = True
    batch_size: int = 32
    input_size: int = 40000
    sample_rate: int = 8000
    n_epochs: int = 20
    lr: float = 0.001
    device: str = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model_path: str = "model"
