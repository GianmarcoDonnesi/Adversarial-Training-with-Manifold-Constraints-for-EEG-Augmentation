import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from prepare_wgan_data import EEGDatasetWGAN
from wgan_gp import generate_signals 
from tft_model import TemporalFusionTransformer
from prepare_tft_data import prepare_tft_data


def riemannian_distance_metric(real_data: np.ndarray, gen_data: np.ndarray) -> float:
    """
    Calcola la distanza Riemanniana media tra la covarianza media dei real_data e gen_data.
    
    Parametri:
    - real_data:  shape (N, C, T) => N epoche, C canali, T campioni
    - gen_data:   shape (M, C, T)
    
    Ritorna:
    - dist_rg: Distanza Riemanniana (float)
    """
    if Covariances is None or distance_riemann is None:
        print("[ERROR] PyRiemann non disponibile, impossibile calcolare la distanza Riemanniana.")
        return None

    # Stima covarianze SPD
    cov_r = Covariances(estimator='oas').fit_transform(real_data)  
    cov_g = Covariances(estimator='oas').fit_transform(gen_data)   

    cov_r_mean = np.mean(cov_r, axis=0)  
    cov_g_mean = np.mean(cov_g, axis=0)  

    # Distanza Riemanniana tra le due SPD medie
    dist_rg = distance_riemann(cov_r_mean, cov_g_mean)
    return dist_rg


def classify_real_vs_fake(real_data: np.ndarray, gen_data: np.ndarray) -> float:
    """
    Addestra un classificatore (LogisticRegression) per distinguere
    real_data vs gen_data e valuta l'accuracy.
    
    Ritorna:
    - accuracy (float): se ~0.5 => real e fake difficili da distinguere
                        se ~1.0 => generati facilmente distinguibili
    """
    # Etichette: 1 = reale, 0 = fake
    y_real = np.ones(len(real_data))
    y_fake = np.zeros(len(gen_data))


    data_all = np.concatenate((real_data, gen_data), axis=0)
    labels_all = np.concatenate((y_real, y_fake), axis=0)

    N, C, T = data_all.shape
    data_flat = data_all.reshape(N, C*T)

    X_train, X_test, y_train, y_test = train_test_split(
        data_flat, labels_all,
        test_size=0.2,
        random_state=42,
        stratify=labels_all
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc


def evaluate_wgan_eeg(real_data: np.ndarray, gen_data: np.ndarray):
    results = {}

    dist_riem = riemannian_distance_metric(real_data, gen_data)
    results['RiemannianDistance'] = dist_riem

    acc_rf = classify_real_vs_fake(real_data, gen_data)
    results['RealVsFakeAccuracy'] = acc_rf

    return results


def validate_tft(
    model_path: str,
    loaded_data_test: dict,
    sequence_length: int = 10,
    batch_size: int = 64,
    device: str = 'cuda'
):

    test_dataloader = prepare_tft_data(
        loaded_data_test,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False
    )

  
    model = TemporalFusionTransformer(
        input_size=512,   
        d_model=128,
        num_lstm_layers=2,
        n_heads=4,
        d_ff=256,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1,
        output_size=200,  
        use_pos_enc=True,
        use_lstm=True
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_sequences, batch_labels in test_dataloader:
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_sequences) 
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm
    }
    return results


if __name__ == "__main__":
    logging.basicConfig(
        filename="validate_models.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    real_data = np.random.randn(500, 32, 128) 
    gen_data = np.random.randn(500, 32, 128)  

    wgan_eeg_results = evaluate_wgan_eeg(real_data, gen_data)
    print("[WGAN EEG Validation Results]", wgan_eeg_results)

    loaded_data_test = {
        'ts_features': np.random.randn(1000, 512),
        'labels': np.random.randint(0, 200, size=(1000,))
    }
    tft_model_path = "tft_model.pth"

    tft_results = validate_tft(
        model_path=tft_model_path,
        loaded_data_test=loaded_data_test,
        sequence_length=10,
        batch_size=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("[TFT Validation Results]", tft_results)
