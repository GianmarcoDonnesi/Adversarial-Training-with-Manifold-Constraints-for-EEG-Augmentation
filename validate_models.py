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

class EEG_net(nn.Module):

    def __init__(self, num_classes=200):
        super(EEG_net, self).__init__()

        self.model_eeg = nn.Sequential(
            nn.Conv1d(32, 20, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),

            nn.Conv1d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(40, affine=False),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(40, 80, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(80, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(160, affine=False),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool1d(1)
        )

        self.linear1 = nn.Linear(160, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h = self.model_eeg(x)
        h = torch.flatten(h, start_dim=1)
        h = self.linear1(h)
        return self.softmax(h)

#Frèchet Inception Distance
def fid(real_signals, generated_signals, model):
    features_r = model(torch.tensor(real_signals).float()).detach().numpy()
    features_g = model(torch.tensor(generated_signals).float()).detach().numpy()

    m_1, m_2 = np.mean(features_r, axis=0), np.mean(features_g, axis=0)
    s_1, s_2 = np.cov(features_r, rowvar=False), np.cov(features_g, rowvar=False)

    cov_m = sqrtm(s_1.dot(s_2))
    if np.iscomplexobj(cov_m):
        cov_m = cov_m.real

    fid1 = np.sum((m_1 - m_2)**2)
    fid2 = np.trace(s_1 + s_2 - 2 * cov_m)
    return fid1 + fid2

#Inception Score
def inception_s(signals, model, batch_size = 32, s = 10):
    scores = []

    for _ in range(s):
        s_probs = []
        for i in range(0, len(signals), batch_size):
            batch = torch.tensor(signals[i:i + batch_size]).float()
            with torch.no_grad():
                y = F.softmax(model(batch), dim=1)
            s_probs.append(y.cpu().numpy())

        s_probs = np.concatenate(s_probs)
        y = np.mean(s_probs, axis=0)

        score = np.exp(np.mean(np.sum(s_probs * np.log(s_probs / y), axis=1)))
        scores.append(score)

    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std

#Kernel Inception Distance
def kid(features_r, features_g, d = 3, g = None, c_0 = 1):
    features_r = torch.tensor(features_r)
    features_g = torch.tensor(features_g)

    if g is None:
        g = 1.0 / features_r.shape[1]

    k_xx = (g * features_r.mm(features_r.t()) + c_0) ** d
    k_yy = (g * features_g.mm(features_g.t()) + c_0) ** d
    k_xy = (g * features_r.mm(features_g.t()) + c_0) ** d

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

#Precision and Recall
def precision_recall(features_r, features_g, k = 3):

    #Precision
    nn_r = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(features_r)
    _, indices_r = nn_r.kneighbors(features_g)
    precision = np.mean([np.any(np.isin(idx, indices_r)) for idx in indices_r])

    #Recall
    nn_g = NearestNeighbors(n_neighbors = k, algorithm = 'auto').fit(features_g)
    _, indices_g = nn_g.kneighbors(features_r)
    recall = np.mean([np.any(np.isin(idx, indices_g)) for idx in indices_g])

    return precision, recall


def riemannian_distance_metric(real_data: np.ndarray, gen_data: np.ndarray) -> float:

    #SPD Covariances
    cov_r = Covariances(estimator='oas').fit_transform(real_data)
    cov_g = Covariances(estimator='oas').fit_transform(gen_data)

    cov_r_mean = np.mean(cov_r, axis=0)
    cov_g_mean = np.mean(cov_g, axis=0)

    dist_rg = distance_riemann(cov_r_mean, cov_g_mean)
    return dist_rg


def classify_real_vs_fake(real_data: np.ndarray, gen_data: np.ndarray) -> float:

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

    results = {}

    dist_riem = riemannian_distance_metric(real_data, gen_data)
    results['RiemannianDistance'] = dist_riem

    acc_rf = classify_real_vs_fake(real_data, gen_data)
    results['RealVsFakeAccuracy'] = acc_rf

    model = EEG_net(num_classes = 200).to(device)

    real_data_t = torch.tensor(real_data).float().to(device)
    print(f"[DEBUG] Shape dei dati reali: {real_data_t.shape}")
    gen_data_t = torch.tensor(gen_data).float().to(device)

    features_r = model(real_data_t).detach().cpu().numpy()
    features_g = model(gen_data_t).detach().cpu().numpy()

    if 'FID' in metrics:
        v = fid(real_data, gen_data, model)
        results['FID'] = v

    if 'IS' in metrics:
        mean, std = inception_s(gen_data, model)
        results['InceptionScore'] = {'mean': mean, 'std': std}

    if 'KID' in metrics:
        v = kid(features_r, features_g)
        results['KID'] = v

    if 'Precision and Recall' in metrics:
        precision, recall = precision_recall(features_r, features_g)
        results['Precision'] = precision
        results['Recall'] = recall

    return results


def validate_tft(model_path, loaded_data, label_encoder, sequence_length=10, batch_size=64, device='cuda'):

    test_dataloader, _ = prepare_tft_data(
        loaded_data,
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

    logging.info("Start Validation models.")
    datapath = './ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')

    subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]
    loaded_data = load_preprocessed_data(subject_ids, derivatives_path)

    real_data = loaded_data['ts_features']
    real_labels = loaded_data['labels']

    n_channels = 32
    n_frames = 16
    total_features = n_channels * n_frames

    real_data_cut = real_data[:, :total_features]

    mean = real_data_cut.mean(axis=0)
    std = real_data_cut.std(axis=0) + 1e-8
    real_data_zscore = (real_data_cut - mean) / std

    real_data_reshaped = real_data_zscore.reshape(-1, n_channels, n_frames)
    logging.info(f"Loaded real data. Shape: {real_data_reshaped.shape}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:

        logging.info("Start WGAN-GP Validation.")

        gen_model_path = os.path.join(derivatives_path, "wgan_generator.pth")
        disc_model_path = os.path.join(derivatives_path, "wgan_discriminator.pth")

        wgan_eeg_results = evaluate_wgan_eeg(
            real_data=real_data_reshaped,
            gen_data=real_data_reshaped,
            gen_model_path=gen_model_path,
            disc_model_path=disc_model_path,
            metrics=['FID', 'IS', 'KID', 'Precision and Recall'],
            device=device
        )
        logging.info(f"WGAN-GP Validation Results: {wgan_eeg_results}")
        print("WGAN-GP Validation Results", wgan_eeg_results)

    except Exception as e:
        logging.error(f"Error during WGAN-GP Validation: {e}", exc_info=True)
        print("[Error] WGAN-GP Validation failed.")

    try:

        logging.info("Start TFT Validation.")

        sequence_length = 10
        tft_dataloader, label_encoder = prepare_tft_data(
            loaded_data,
            sequence_length=sequence_length,
            batch_size=64,
            shuffle=False,
            n_augments=0
        )

        tft_model_path = os.path.join(derivatives_path, "tft_model.pth")

        tft_results = validate_tft(
            model_path=tft_model_path,
            loaded_data=loaded_data,
            label_encoder=label_encoder,
            sequence_length=10,
            batch_size=64,
            device=device
        )
        logging.info(f"TFT Validation Results: {tft_results}")
        print("TFT Validation Results", tft_results)

    except Exception as e:
        logging.error(f"Error during TFT Validation: {e}", exc_info=True)
        print("[Error] TFT Validation failed.")

    logging.info("Validation complete.")
    print("Validation complete. Check log file for details.")
