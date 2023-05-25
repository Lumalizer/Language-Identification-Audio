# Replace the string below:
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
MODEL_PATH = "model/JIT_model_state_dict.pt"


# disable gradient calculation
torch.set_grad_enabled(False)

N_CLASSES = 6
N = 1200

X_full, y_full = np.load("dataset/inputs_test_fp16.npy"), np.load(
    "dataset/targets_test_int8.npy")

X_full = X_full.astype(np.float32)
X_full = torch.from_numpy(X_full)
targets_full = torch.from_numpy(y_full)
# one hot encode
y_full = torch.nn.functional.one_hot(targets_full.long(), N_CLASSES).float()
dataset = TensorDataset(X_full, y_full)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


def handle_uploaded_model():
    try:
        model = torch.jit.load(MODEL_PATH).to("cpu")
        model.eval()
        outputs = []
        for x, _ in dataloader:
            outputs.append(model(x).argmax(dim=1).long())
        outputs = torch.cat(outputs)
        accuracy = (outputs == targets_full).sum().item() / N
        accuracy *= 100
        print(f"accuracy is {accuracy:.3f}")

    except Exception as e:
        print(f"error running model")
        print(e)


if __name__ == "__main__":
    handle_uploaded_model()
