import numpy as np
from options import Options
from q1b_normalize_data import normalize_data
from torch.utils.data import DataLoader,TensorDataset
import torch 

def load_data(options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampling_rate = 8_000
    languages = ["de", "en", "es", "fr", "nl", "pt"]
    language_dict = {languages[i]: i for i in range(len(languages))}

    X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
        "dataset/targets_train_int8.npy")
    X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
        "dataset/targets_test_int8.npy")

    if options.normalize:
        X_train, X_test = normalize_data(X_train, X_test)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    return X_train, y_train, X_test, y_test




def convert_tensors(X_train,y_train, X_test, y_test):
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create PyTorch data loaders
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        return train_loader, test_loader
