"""
Loads Thyroid, Arrhythmia, Cardio from ODDS via PyOD or direct download.
Semi-supervised split: TRAIN = normal only, TEST = normal + anomalies.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import urllib.request
import os
from scipy.io import loadmat

def load_odds_dataset(name):
    """Load ODDS datasets. Falls back to direct download if pyod datasets unavailable."""
    name = name.lower()
    
    # Try PyOD first
    try:
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
    except (ImportError, ModuleNotFoundError):
        # Fallback: Download from ODDS repository
        print(f"PyOD datasets not available, downloading {name} directly...")
        X, y = _download_odds_dataset(name)
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    print(f"[{name.upper()}] {X.shape[0]} samples, {X.shape[1]} features | Anomaly rate: {y.mean():.1%}")
    return X, y.astype(np.float32)

def _download_odds_dataset(name):
    """Download dataset from ODDS repository."""
    name = name.lower()
    base_url = "https://www.dropbox.com/s/hvhkkz0zlm6n9e6"
    
    files = {
        'thyroid': ('thyroid.mat', 'thyroid'),
        'arrhythmia': ('arrhythmia.mat', 'arrhythmia'),
        'cardio': ('cardio.mat', 'cardio')
    }
    
    if name not in files:
        raise ValueError(f"Unknown dataset: {name}")
    
    filename, dataset_key = files[name]
    url = f"{base_url}/{filename}?dl=1"
    
    # Create temp directory
    os.makedirs("/tmp/odds_data", exist_ok=True)
    filepath = f"/tmp/odds_data/{filename}"
    
    # Download if not exists
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    
    # Load .mat file
    mat = loadmat(filepath)
    X = mat['X'].astype(np.float32)
    y = mat['y'].ravel().astype(np.float32)
    
    return X, y

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
