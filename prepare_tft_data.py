import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from load_preprocessed_data import load_preprocessed_data
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class EEGSequenceDatasetTFT(Dataset):
    def __init__(self, features, labels, sequence_length=10, transform=None):

        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence = self.features[idx:idx + self.sequence_length]
        sequence = sequence.view(self.sequence_length, -1)

        label = self.labels[idx + self.sequence_length - 1]

        if self.transform:
            sequence = self.transform(sequence)

        return sequence.clone().detach(), label.clone().detach()

def add_gaussian_noise(data, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def time_warp(data, factor_range=(0.8, 1.2)):

    factor = np.random.uniform(*factor_range)
    indices = np.round(np.arange(0, data.shape[1], factor)).astype(int)
    indices = np.clip(indices, 0, data.shape[1] - 1)
    warped_data = data[:, indices]

    M = warped_data.shape[1]
    target_frames = data.shape[1]

    if M > target_frames:
        start = np.random.randint(0, M - target_frames + 1)
        warped_data = warped_data[:, start : start + target_frames]
    elif M < target_frames:
        pad_width = target_frames - M
        warped_data = np.pad(
            warped_data,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant'
        )

    return warped_data

def scale_data(data, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range, size=(data.shape[0], 1))
    return data * scale

def permute_channels(data):
    perm = np.random.permutation(data.shape[0])
    return data[perm, :]

def augment_data(features, n_augments=1):
    if n_augments == 0:
          return np.array([])

    features_augmented = []
    for feature in features:
        for _ in range(n_augments):
            transformed = add_gaussian_noise(feature)
            transformed = time_warp(transformed)
            transformed = scale_data(transformed)
            transformed = permute_channels(transformed)
            features_augmented.append(transformed)
    if not features_augmented:
        raise ValueError("No augmented data generated.")
    return np.array(features_augmented).reshape(-1, *features[0].shape)

def prepare_tft_data(loaded_data, sequence_length=20, batch_size=64, shuffle=True, n_augments=5, validation_split=0.2):

    X = loaded_data['ts_features']
    y = loaded_data['labels']

    n_channels = 32
    n_frames = 16


    total_features = n_channels * n_frames
    X_cut = X[:, :total_features]

    mean = X_cut.mean(axis=0)
    std = X_cut.std(axis=0) + 1e-8
    X_zscore = (X_cut - mean) / std

    X_3d = X_zscore.reshape(-1, n_channels, n_frames)

    X_augmented = augment_data(X_3d, n_augments=n_augments)

    if X_augmented.size > 0:
        X_combined = np.concatenate((X_3d, X_augmented), axis=0)
        y_combined = np.concatenate((y, np.tile(y, n_augments)), axis=0)
    else:
        X_combined = X_3d
        y_combined = y

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_combined)

    num_classes = len(label_encoder.classes_)

    print(f"[INFO] Original data shape: {X_3d.shape}")
    print(f"[INFO] Augmented data shape: {X_augmented.shape}")
    print(f"[INFO] Combined data shape: {X_combined.shape}")
    print(f"[INFO] Number of classes: {num_classes}")

    features_tensor = torch.tensor(X_combined, dtype=torch.float32)
    labels_tensor = torch.tensor(y_combined, dtype=torch.long)


    dataset = EEGSequenceDatasetTFT(features_tensor, labels_tensor, sequence_length=sequence_length)
    if validation_split > 0:
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42, shuffle=shuffle)

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        print(f"[INFO] TFT Train DataLoader created with {len(train_dataset)} samples.")
        print(f"[INFO] TFT Validation DataLoader created with {len(val_dataset)} samples.")

        return train_dataloader, val_dataloader, label_encoder
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
        print(f"[INFO] TFT DataLoader created with {len(dataset)} samples.")
        return dataloader, label_encoder


if __name__ == "__main__":

    datapath = './ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')


    subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    loaded_data = load_preprocessed_data(subject_ids, derivatives_path)

    sequence_length = 10
    train_loader, val_loader, label_encoder = prepare_tft_data(
        loaded_data,
        sequence_length=sequence_length,
        batch_size=64,
        shuffle=True,
        n_augments=5,
        validation_split=0.2
    )

    for batch_sequences, batch_labels in train_loader:
        print(f"Train batch sequences shape: {batch_sequences.shape}")
        print(f"Train batch labels shape: {batch_labels.shape}")
        break

    for batch_sequences, batch_labels in val_loader:
        print(f"Validation batch sequences shape: {batch_sequences.shape}")
        print(f"Validation batch labels shape: {batch_labels.shape}")
        break