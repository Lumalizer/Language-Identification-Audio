import torch
import numpy as np
from options import Options
from torch.utils.data import DataLoader, TensorDataset


def load_data(options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    languages = ["de", "en", "es", "fr", "nl", "pt"]
    language_dict = {languages[i]: i for i in range(len(languages))}

    X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
        "dataset/targets_train_int8.npy")
    X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
        "dataset/targets_test_int8.npy")

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    if not options.use_all_languages:
        X_train = np.array([X_train[i] for i in range(
            len(X_train)) if y_train[i] == 1 or y_train[i] == 2])
        y_train = np.array([y_train[i] for i in range(
            len(y_train)) if y_train[i] == 1 or y_train[i] == 2]) - 1

        X_test = np.array([X_test[i] for i in range(
            len(X_test)) if y_test[i] == 1 or y_test[i] == 2])
        y_test = np.array([y_test[i] for i in range(
            len(y_test)) if y_test[i] == 1 or y_test[i] == 2]) - 1

    return X_train, y_train, X_test, y_test


def get_dataloaders(options: Options) -> tuple[DataLoader, DataLoader]:
    X_train, y_train, X_test, y_test = load_data(options)

    train_dataset = TensorDataset(torch.tensor(
        X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(
        X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(
        train_dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=options.batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader
