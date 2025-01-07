import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from load_preprocessed_data import load_preprocessed_data
from sklearn.model_selection import train_test_split


class EEGDatasetWGAN(Dataset):
    def __init__(self, features, labels, transform=None):
        
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label

def prepare_wgan_data(loaded_data, batch_size=64, shuffle=True):

    X = loaded_data['ts_features']
    y = loaded_data['labels']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_channels = 32
    n_frames = 16

    X_train_cut = X_train[:, :512]
    X_test_cut = X_test[:, :512]

    mean = X_train_cut.mean(axis=0)
    std = X_train_cut.std(axis=0) + 1e-8
    X_train_norm = (X_train_cut - mean) / std
    X_test_norm  = (X_test_cut  - mean) / std

    X_train_3d = X_train_norm.reshape(-1, n_channels, n_frames)
    X_test_3d = X_test_norm.reshape(-1, n_channels, n_frames)

    np.savez(os.path.join(derivatives_path, 'train_proj.npz'), features=X_train_3d, labels=y_train)
    np.savez(os.path.join(derivatives_path, 'test_proj.npz'), features=X_test_3d, labels=y_test)

    features_tensor = torch.tensor(X_train_3d, dtype=torch.float32)  
    labels_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = EEGDatasetWGAN(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


    print(f"[INFO] WGAN DataLoader created with {len(dataset)} samples.")
    print(f"[DEBUG] features_tensor shape = {features_tensor.shape}")
    return dataloader


if __name__ == "__main__":
    datapath = './ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')
    subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    loaded_data = load_preprocessed_data(subject_ids, derivatives_path)
    ts_features = loaded_data['ts_features']
    labels = loaded_data['labels']

    wgan_dataloader = prepare_wgan_data(loaded_data, batch_size=64, shuffle=True)

    for batch_features, batch_labels in wgan_dataloader:
        print(f"Batch features shape: {batch_features.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        break