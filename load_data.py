import torch 
import numpy as np
from options import Options
from q1b_normalize_data import normalize_data
from torch.utils.data import DataLoader,TensorDataset

def load_data(options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def convert_tensors(X_train, y_train, X_test, y_test, options: Options) -> tuple[DataLoader, DataLoader]:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # Create TensorDatasets
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create PyTorch data loaders
        train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=True)
    
        return train_loader, test_loader

def english_spanish_data(X_train, y_train, X_test, y_test) -> tuple[DataLoader, DataLoader]:
    X_train = np.array([X_train[i] for i in range(len(X_train)) if y_train[i] == 1 or y_train[i] == 2])
    y_train = np.array([y_train[i] for i in range(len(y_train)) if y_train[i] == 1 or y_train[i] == 2]) - 1

    X_test = np.array([X_test[i] for i in range(len(X_test)) if y_test[i] == 1 or y_test[i] == 2])
    y_test = np.array([y_test[i] for i in range(len(y_test)) if y_test[i] == 1 or y_test[i] == 2]) - 1

    return X_train, y_train, X_test, y_test

def get_dataloaders(options: Options) -> tuple[DataLoader, DataLoader]:
    if options.use_all_languages:
        train_loader, test_loader = convert_tensors(*load_data(options), options)
    else:
        train_loader, test_loader = convert_tensors(*english_spanish_data(*load_data(options)), options)
    return train_loader, test_loader