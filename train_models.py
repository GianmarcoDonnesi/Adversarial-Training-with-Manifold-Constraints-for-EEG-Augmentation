import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from load_preprocessed_data import load_preprocessed_data
from prepare_wgan_data import prepare_wgan_data
from riemannian_manifold import Spatial
from wgan_gp import (
    WGAN_GP_Discriminator,
    WGAN_GP_Generator,
    wgan_gp_init_w,
    train, generate_signals
)
from prepare_tft_data import prepare_tft_data
from tft_model import TFTRefinedModel, train_tft

def train_wgan_synthetic(
    datapath='/content/drive/MyDrive/AIRO/Projects/EAI_Project/ds005106',
    subject_ids=None,
    n_rank=24,
    batch_size=64,
    num_steps=10000,
    synth_file_name='synthetic_data.npz',
    train_data_file = 'train_proj.npz',
    test_data_file = 'test_proj.npz'
):

    if subject_ids is None:
        subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')

    train_file_path = os.path.join(derivatives_path, train_data_file)
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"[ERROR] File {train_file_path} non trovato. Assicurati di aver eseguito prepare_wgan_data.")

    print(f"[INFO] Caricamento dei dati di training da {train_file_path}.")
    train_data = np.load(train_file_path)
    features = torch.tensor(train_data['features'], dtype=torch.float32)
    labels = torch.tensor(train_data['labels'], dtype=torch.float32)

    dataset = EEGDatasetWGAN(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
    total_count = counts.sum()
    proportions = counts / total_count

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netD = WGAN_GP_Discriminator(ndf = 64, gp_scale = 10).to(device)
    netG = WGAN_GP_Generator(nz = 300, ngf = 100, s_length = 50).to(device)

    wgan_gp_init_w(netD)
    wgan_gp_init_w(netG)

    optD = torch.optim.Adam(netD.parameters(), lr = 0.001, betas = (0.5, 0.99))
    optG = torch.optim.Adam(netG.parameters(), lr = 0.001, betas = (0.5, 0.99))

    train(
        D_net = netD,
        G_net = netG,
        D_opt = optD,
        G_opt = optG,
        lr_D = optD.param_groups[0]['lr'],
        lr_G = optG.param_groups[0]['lr'],
        n_ots = 5,
        n_steps = num_steps,
        dataloader = dataloader
    )

    netG.eval()
    num_synth_signals = 4000

    with torch.no_grad():
        synthetic_data = generate_signals(netG, num_synth_signals, device)

    synthetic_data_np = synthetic_data.cpu().numpy()

    synthetic_labels = np.random.choice(
        unique_labels,
        size=num_synth_signals,
        p=proportions
    ).astype(int)

    out_synth_file = os.path.join(derivatives_path, synth_file_name)
    np.savez(
        out_synth_file,
        ts_features=synthetic_data_np,
        labels=synthetic_labels
    )

    print(f"\nSaved {num_synth_signals} synthetic signals in {out_synth_file}.\n")

def train_tft(
    datapath='/content/drive/MyDrive/AIRO/Projects/EAI_Project/ds005106',
    subject_ids=None,
    synth_file_name='synthetic_data.npz',
    sequence_length=10,
    batch_size=32,
    num_outputs=5,
    num_epochs=10
):
    """
    1) Carica dati reali preprocessati.
    2) Carica dati sintetici.
    3) Combina i dataset.
    4) Crea DataLoader TFT.
    5) Inizializza e addestra il TFT sul dataset arricchito.
    """

    if subject_ids is None:
        subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')

    # 1)Carica dati reali
    loaded_data_real = load_preprocessed_data(subject_ids, derivatives_path)
    real_features = loaded_data_real['ts_features']
    real_labels = loaded_data_real['labels']

    # 2)Carica dati sintetici
    synthetic_file = os.path.join(derivatives_path, synth_file_name)
    synth_data = np.load(synthetic_file)
    synth_features = synth_data['ts_features']
    synth_labels = synth_data['labels']

    # 3)Combina dataset
    combined_features = np.concatenate((real_features, synth_features), axis=0)
    combined_labels   = np.concatenate((real_labels,   synth_labels),   axis=0)

    print(f"[INFO] Dati reali shape: {real_features.shape}")
    print(f"[INFO] Dati sintetici shape: {synth_features.shape}")
    print(f"[INFO] Totale dataset combinato: {combined_features.shape}")

    # 4)Crea DataLoader per TFT
    loaded_data_combined = {
        'ts_features': combined_features,
        'labels': combined_labels
    }

    tft_dataloader = prepare_tft_data(
        loaded_data_combined,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True
    )

    # 5)Inizializza e addestra TFT

    feature_dim = combined_features.shape[1]

    model = TFTRefinedModel(
        input_size=feature_dim,
        hidden_size=64,
        num_heads=4,
        num_outputs=num_outputs,
        num_encoder_layers=2,
        dropout=0.1,
        use_pos_enc=True
    )

    train_tft(
        model=model,
        dataloader=tft_dataloader,
        num_epochs=num_epochs,
        learning_rate=1e-3,
        device=device,
        task='classification'  # o 'regression'
    )

    print("\n[INFO] Training completato su dataset arricchito.\n")


def main():
    """
    Esegue in sequenza:
    1) Training WGAN e generazione dati sintetici.
    2) Training TFT sul dataset (reale + sintetico).
    """
    # Parametri comuni
    DATAPATH = '/content/drive/MyDrive/AIRO/Projects/EAI_Project/ds005106'
    SUBJECT_IDS = ['{:03d}'.format(i) for i in range(1, 52)]

    train_wgan_synthetic(
        datapath=DATAPATH,
        subject_ids=SUBJECT_IDS,
        n_rank=24,           #rank per proiezione Riemann
        batch_size=64,
        num_steps=10000,
        synth_file_name='synthetic_data.npz'
    )
    
    #Allena il TFT sul dataset arricchito (reale + sintetico)
    train_tft(
        datapath=DATAPATH,
        subject_ids=SUBJECT_IDS,
        synth_file_name='synthetic_data.npz',
        sequence_length=10,
        batch_size=32,
        num_outputs=5,
        num_epochs=10
    )


if __name__ == "__main__":
    main()