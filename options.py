import torch
from dataclasses import dataclass


@dataclass
class Options:
    use_all_languages: bool
    normalize: bool = True
    batch_size: int = 64
    input_size: int = 40000
    sample_rate: int = 16000
    n_epochs: int = 10
    lr: float = 0.001
    model_path: str = "model"
    record_intermediate_losses: bool = False
    device: str = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
