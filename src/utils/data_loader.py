"""
Loads Thyroid, Arrhythmia, Cardio from ODDS via PyOD.
Semi-supervised split: TRAIN = normal only, TEST = normal + anomalies.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_odds_dataset(name):
    name = name.lower()
    if name == 'thyroid':
        from pyod.datasets.data import load_thyroid
        X, y = load_thyroid()
    elif name == 'arrhythmia':
        from pyod.datasets.data import load_arrhythmia
        X, y = load_arrhythmia()
    elif name == 'cardio':
        from pyod.datasets.data import load_cardio
        X, y = load_cardio()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    print(f"[{name.upper()}] {X.shape[0]} samples, {X.shape[1]} features | Anomaly rate: {y.mean():.1%}")
    return X, y.astype(np.float32)

def make_train_test_split(X, y, test_size=0.2, seed=42):
    X_normal = X[y == 0]; X_anomaly = X[y == 1]
    X_norm_train, X_norm_test = train_test_split(X_normal, test_size=test_size, random_state=seed)
    X_test = np.vstack([X_norm_test, X_anomaly])
    y_test = np.concatenate([np.zeros(len(X_norm_test)), np.ones(len(X_anomaly))])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_test))
    print(f"  Train: {len(X_norm_train)} normal | Test: {len(X_norm_test)} normal + {len(X_anomaly)} anomaly")
    return X_norm_train, X_test[idx], y_test[idx]

def make_dataloaders(X_train, X_test, batch_size=64):
    train_ds = TensorDataset(torch.tensor(X_train))
    test_ds  = TensorDataset(torch.tensor(X_test))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False))
