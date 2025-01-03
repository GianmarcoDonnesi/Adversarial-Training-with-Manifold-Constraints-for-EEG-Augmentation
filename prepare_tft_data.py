import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from load_preprocessed_data import load_preprocessed_data
from tqdm.notebook import tqdm

class EEGSequenceDatasetTFT(Dataset):
    def __init__(self, features, labels, sequence_length=10, transform=None):
        """
        Dataset per TFT.

        Parameters:
        - features (numpy.ndarray): Array delle features (ts_features).
        - labels (numpy.ndarray): Array delle etichette.
        - sequence_length (int): Lunghezza della sequenza temporale.
        - transform: Trasformazioni da applicare ai dati.
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence = self.features[idx:idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length - 1]

        if self.transform:
            sequence = self.transform(sequence)

        return sequence.clone().detach(), label.clone().detach()

def prepare_tft_data(loaded_data, sequence_length=10, batch_size=64, shuffle=True):
    """
    Prepara i DataLoader per TFT.

    Parameters:
    - loaded_data (dict): Dati caricati tramite load_preprocessed_data.
    - sequence_length (int): Lunghezza della sequenza temporale.
    - batch_size (int): Dimensione del batch.
    - shuffle (bool): Se mescolare i dati.

    Returns:
    - DataLoader: DataLoader per l'addestramento del TFT.
    """

    ts_features = loaded_data['ts_features']

    #Normalizzazione: min-max per ogni feature
    min_val = ts_features.min(axis=0)
    max_val = ts_features.max(axis=0) + 1e-8
    ts_features_norm = (ts_features - min_val) / (max_val - min_val)

    #Converti le features e le etichette in PyTorch tensors
    features_tensor = torch.tensor(ts_features_norm, dtype=torch.float32)
    labels_tensor = torch.tensor(loaded_data['labels'], dtype=torch.long)


    dataset = EEGSequenceDatasetTFT(features_tensor, labels_tensor, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    print(f"[INFO] TFT DataLoader created with {len(dataset)} samples.")
    return dataloader


if __name__ == "__main__":

    datapath = '/content/drive/MyDrive/AIRO/Projects/EAI_Project/ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')


    subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    loaded_data = load_preprocessed_data(subject_ids, derivatives_path)

    #Prepara il DataLoader per TFT
    sequence_length = 10
    tft_dataloader = prepare_tft_data(loaded_data, sequence_length=sequence_length, batch_size=32, shuffle=True)

    for batch_sequences, batch_labels in tft_dataloader:
        print(f"Batch sequences shape: {batch_sequences.shape}")  #(batch_size, sequence_length, feature_dim)
        print(f"Batch labels shape: {batch_labels.shape}")
        break