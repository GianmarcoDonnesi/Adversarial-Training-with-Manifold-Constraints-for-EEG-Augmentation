import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

def load_preprocessed_data(subject_ids, derivatives_path):
    """
    Carica i dati preprocessati da file .npz per tutti i soggetti.

    Returns:
    - dict: Un dizionario contenente tutte le features e le etichette combinate.
    """
    all_ts_features = []
    all_labels = []
    all_subjects = []
    all_chan_labels = None
    all_times = None
    all_onsets_sec = []

    print("[INFO] Loading preprocessed data...")
    for subject_id in tqdm(subject_ids, desc="Loading subjects"):
        out_fn = os.path.join(derivatives_path, f"sub-{subject_id}_preprocessed_tangentspace.npz")
        if not os.path.exists(out_fn):
            print(f"[WARNING] Preprocessed file for sub-{subject_id} not found. Skipping...")
            continue

        data = np.load(out_fn)
        ts_features = data['ts_features']
        labels = data['labels']
        subjects = data['subjects']
        onsets_sec = data['onsets_sec']

        all_ts_features.append(ts_features)
        all_labels.append(labels)
        all_subjects.append(subjects)
        all_onsets_sec.append(onsets_sec)

        if all_chan_labels is None:
            all_chan_labels = data['chan_labels']
        if all_times is None:
            all_times = data['times']

    all_ts_features = np.concatenate(all_ts_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_subjects = np.concatenate(all_subjects, axis=0)
    all_onsets_sec = np.concatenate(all_onsets_sec, axis=0)

    loaded_data = {
        'ts_features': all_ts_features,
        'labels': all_labels,
        'subjects': all_subjects,
        'chan_labels': all_chan_labels,
        'times': all_times,
        'onsets_sec': all_onsets_sec
    }

    print(f"[INFO] Loaded ts_features shape: {all_ts_features.shape}")
    print(f"[INFO] Loaded labels shape: {all_labels.shape}")
    print(f"[INFO] Loaded subjects shape: {all_subjects.shape}")
    return loaded_data


if __name__ == "__main__":

    datapath = '/content/drive/MyDrive/AIRO/Projects/EAI_Project/ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')

    subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    loaded_data = load_preprocessed_data(subject_ids, derivatives_path)

    ts_features = loaded_data['ts_features']
    labels = loaded_data['labels']
    subjects = loaded_data['subjects']
    chan_labels = loaded_data['chan_labels']
    times = loaded_data['times']
    onsets_sec = loaded_data['onsets_sec']

    print(f"ts_features shape = {ts_features.shape}")
    print(f"labels shape = {labels.shape}")
    print(f"subjects shape = {subjects.shape}")
    print(f"channels = {chan_labels}")