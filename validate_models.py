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
from sklearn.neighbors import NearestNeighbors
from wgan_gp import WGAN_GP_Generator, WGAN_GP_Discriminator
from tft_model import TemporalFusionTransformer
from load_preprocessed_data import load_preprocessed_data
from prepare_tft_data import prepare_tft_data
from scipy.linalg import sqrtm



def riemannian_distance_metric(real_data: np.ndarray, gen_data: np.ndarray) -> float:

    # Stima covarianze SPD
    cov_r = Covariances(estimator='oas').fit_transform(real_data)
    cov_g = Covariances(estimator='oas').fit_transform(gen_data)

    cov_r_mean = np.mean(cov_r, axis=0)
    cov_g_mean = np.mean(cov_g, axis=0)

    # Distanza Riemanniana tra le due SPD medie
    dist_rg = distance_riemann(cov_r_mean, cov_g_mean)
    return dist_rg


def classify_real_vs_fake(real_data: np.ndarray, gen_data: np.ndarray) -> float:
    
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


def evaluate_wgan_eeg(real_data, gen_data, gen_model_path, disc_model_path, metrics=['FID', 'IS', 'KID', 'Precision and Recall'], device='cuda'):
    generator = WGAN_GP_Generator(nz=300, ngf=100, s_length=50, device=device).to(device)
    generator.load_state_dict(torch.load(gen_model_path, map_location=device, weights_only=True))
    generator.eval()

    discriminator = WGAN_GP_Discriminator(ndf=64, gp_scale=50).to(device)
    discriminator.load_state_dict(torch.load(disc_model_path, map_location=device, weights_only=True))
    discriminator.eval()

    synthetic_data = generator.forward_fake_signals(torch.randn(len(gen_data), 300, device=device)).detach().cpu().numpy()

    results = {}

    dist_riem = riemannian_distance_metric(real_data, gen_data)
    results['RiemannianDistance'] = dist_riem

    acc_rf = classify_real_vs_fake(real_data, gen_data)
    results['RealVsFakeAccuracy'] = acc_rf


    return results


def validate_tft(model_path, loaded_data_test, label_encoder, sequence_length=10, batch_size=64, device='cuda'):

    test_dataloader, _ = prepare_tft_data(
        loaded_data_test,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=False,
        n_augments=0
    )


    model = TemporalFusionTransformer(
        input_size=512,
        d_model=128,
        num_lstm_layers=2,
        n_heads=4,
        d_ff=256,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.2,
        output_size=len(label_encoder.classes_),
        use_pos_enc=True,
        use_lstm=True
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
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

    logging.info("Inizio validazione dei modelli.")
    # Percorso dei dati e dei modelli
    datapath = '/content/drive/MyDrive/AIRO/Projects/EAI_Project/ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')

    # Carica i dati
    subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]
    loaded_data_test = load_preprocessed_data(subject_ids, derivatives_path)


    # Estrai dati reali preprocessati
    real_data = loaded_data['ts_features']
    real_labels = loaded_data['labels']

    n_channels = 32 
    n_frames = 16    
    total_features = n_channels * n_frames

    real_data_cut = real_data[:, :total_features]

    # Normalizzazione Z-Score
    mean = real_data_cut.mean(axis=0)
    std = real_data_cut.std(axis=0) + 1e-8
    real_data_zscore = (real_data_cut - mean) / std

    # Reshape dei dati reali per ottenere (n_trials, n_channels, n_frames)
    real_data_reshaped = real_data_zscore.reshape(-1, n_channels, n_frames)
    logging.info(f"Dati reali caricati e reshaped. Shape: {real_data_reshaped.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # **1. Validazione WGAN-GP**
        logging.info("Inizio validazione WGAN-GP.")
        
        gen_model_path = os.path.join(derivatives_path, "wgan_generator.pth")
        disc_model_path = os.path.join(derivatives_path, "wgan_discriminator.pth")

        # Esegui la validazione WGAN-GP
        wgan_eeg_results = evaluate_wgan_eeg(
            real_data=real_data_reshaped,
            gen_data=real_data_reshaped,
            gen_model_path=gen_model_path,
            disc_model_path=disc_model_path,
            metrics=['FID', 'IS', 'KID', 'Precision and Recall'],
            device=device
        )
        logging.info(f"Risultati validazione WGAN-GP: {wgan_eeg_results}")
        print("[WGAN-GP Validation Results]", wgan_eeg_results)

    except Exception as e:
        logging.error(f"Errore durante la validazione WGAN-GP: {e}", exc_info=True)
        print("[ERRORE] Validazione WGAN-GP fallita.")

    try:
        # **2. Validazione TFT**
        logging.info("Inizio validazione TFT.")

        # Prepara il DataLoader e il LabelEncoder
        sequence_length = 10
        tft_dataloader, label_encoder = prepare_tft_data(
            loaded_data_test,
            sequence_length=sequence_length,
            batch_size=64,
            shuffle=False,
            n_augments=0
        )
        
        tft_model_path = os.path.join(derivatives_path, "tft_model.pth")
        
        # Esegui la validazione TFT
        tft_results = validate_tft(
            model_path=tft_model_path,
            loaded_data_test=loaded_data_test,
            label_encoder=label_encoder,
            sequence_length=10,
            batch_size=64,
            device=device
        )
        logging.info(f"Risultati validazione TFT: {tft_results}")
        print("[TFT Validation Results]", tft_results)

    except Exception as e:
        logging.error(f"Errore durante la validazione TFT: {e}", exc_info=True)
        print("[ERRORE] Validazione TFT fallita.")

    logging.info("Validazione completata.")
    print("Validazione completata. Controlla il file di log per i dettagli.")