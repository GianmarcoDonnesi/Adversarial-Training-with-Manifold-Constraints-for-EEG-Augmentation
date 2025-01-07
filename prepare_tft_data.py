import os
import torch
from torch.utils.data import Dataset, DataLoader
from load_preprocessed_data import load_preprocessed_data


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

def prepare_tft_data(loaded_data, sequence_length=10, batch_size=64, shuffle=True):


    X = loaded_data['ts_features']
    y = loaded_data['labels']

    n_channels = 32
    n_frames = 16


    total_features = n_channels * n_frames
    X_cut = X[:, :total_features]

    min_val = X_cut.min(axis=0)
    max_val = X_cut.max(axis=0) + 1e-8
    X_norm = (X_cut - min_val) / (max_val - min_val)

    X_3d = X_norm.reshape(-1, n_channels, n_frames)

    features_tensor = torch.tensor(X_3d, dtype=torch.float32)
    labels_tensor = torch.tensor(loaded_data['labels'], dtype=torch.long)


    dataset = EEGSequenceDatasetTFT(features_tensor, labels_tensor, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    print(f"[INFO] TFT DataLoader created with {len(dataset)} samples.")
    return dataloader


if __name__ == "__main__":

    datapath = './ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')


    subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    loaded_data = load_preprocessed_data(subject_ids, derivatives_path)

    sequence_length = 10
    tft_dataloader = prepare_tft_data(loaded_data, sequence_length=sequence_length, batch_size=64, shuffle=True)

    for batch_sequences, batch_labels in tft_dataloader:
        print(f"Batch sequences shape: {batch_sequences.shape}")  
        print(f"Batch labels shape: {batch_labels.shape}")
        break