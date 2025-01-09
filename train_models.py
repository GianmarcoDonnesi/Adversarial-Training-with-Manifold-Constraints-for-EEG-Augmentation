import os
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from load_preprocessed_data import load_preprocessed_data
from prepare_wgan_data import EEGDatasetWGAN
from torch.utils.data import DataLoader
import torch.nn as nn
from wgan_gp import (
    WGAN_GP_Discriminator,
    WGAN_GP_Generator,
    wgan_gp_init_w,
    train, generate_signals
)
from prepare_tft_data import prepare_tft_data
from tft_model import TemporalFusionTransformer, train_tft


def train_wgan_synthetic(
    datapath='./ds005106',
    subject_ids=None,
    n_rank=24,
    batch_size=64,
    num_steps=1000,
    synth_file_name='synthetic_data.npz',
    train_data_file = 'train_proj.npz',
    test_data_file = 'test_proj.npz'
):

    if subject_ids is None:
        subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')

    train_file_path = os.path.join(derivatives_path, train_data_file)
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"[ERROR] File {train_file_path} not found.")

    print(f"[INFO] Loading training data from {train_file_path}.")
    train_data = np.load(train_file_path)
    features = torch.tensor(train_data['features'], dtype=torch.float32)
    labels = torch.tensor(train_data['labels'], dtype=torch.float32)

    dataset = EEGDatasetWGAN(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
    total_count = counts.sum()
    proportions = counts / total_count

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netD = WGAN_GP_Discriminator(ndf = 64, gp_scale = 50).to(device)
    netG = WGAN_GP_Generator(nz = 300, ngf = 100, s_length = 50, device=device).to(device)

    wgan_gp_init_w(netD)
    wgan_gp_init_w(netG)

    optD = torch.optim.Adam(netD.parameters(), lr = 1e-4, betas = (0.5, 0.9))
    optG = torch.optim.Adam(netG.parameters(), lr = 1e-4, betas = (0.5, 0.9))

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
    num_synth_signals = 8000

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

    print(f"\n[INFO] Saved {num_synth_signals} synthetic data in {out_synth_file}.\n")

    gen_path = os.path.join(derivatives_path, "wgan_generator.pth")
    disc_path = os.path.join(derivatives_path, "wgan_discriminator.pth")

    torch.save(netG.state_dict(), gen_path)
    torch.save(netD.state_dict(), disc_path)

    print(f"[INFO] Saved Generator weights in: {gen_path}")
    print(f"[INFO] Saved Discriminator weights in: {disc_path}")


def train_tft_main(
    datapath='./ds005106',
    subject_ids=None,
    synth_file_name='synthetic_data.npz',
    sequence_length=10,
    batch_size=64,
    num_outputs=200,
    num_epochs=15
):

    if subject_ids is None:
        subject_ids = ['{:03d}'.format(i) for i in range(1, 52)]

    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')

    loaded_data_real = load_preprocessed_data(subject_ids, derivatives_path)
    real_features = loaded_data_real['ts_features']
    real_labels = loaded_data_real['labels']

    synthetic_file = os.path.join(derivatives_path, synth_file_name)
    synth_data = np.load(synthetic_file)
    synth_features = synth_data['ts_features']
    synth_labels = synth_data['labels']

    real_features_cut = real_features[:, :512]
    real_features_3d  = real_features_cut.reshape(-1, 32, 16)

    combined_features_3d = np.concatenate((real_features_3d, synth_features), axis=0)
    combined_labels = np.concatenate((real_labels, synth_labels), axis=0)

    print(f"[INFO] Real data shape: {real_features.shape}")
    print(f"[INFO] Synthetic data shape: {synth_features.shape}")
    print(f"[INFO] Combined dataset shape: {combined_features_3d.shape}")

    label_encoder = LabelEncoder()
    combined_labels_encoded = label_encoder.fit_transform(combined_labels)
    combined_labels = combined_labels_encoded
    num_classes = len(label_encoder.classes_)
    num_outputs = num_classes  

    print(f"[INFO] Number of classes after encoding: {num_classes}")

    assert combined_labels.max() < num_outputs, "Some labels exceed the expected number of classes."
    print(f"[INFO] Number of classes: {num_classes}")

    combined_features_2d = combined_features_3d.reshape(-1, 32*16)

    loaded_data_combined = {
        'ts_features': combined_features_2d,
        'labels': combined_labels
    }

    tft_dataloader, _ = prepare_tft_data(
        loaded_data_combined,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True
    )

    feature_dim = combined_features_2d.shape[1]

    model = TemporalFusionTransformer(
        input_size=feature_dim,
        d_model=128,
        num_lstm_layers=2,
        n_heads=8,
        d_ff=256,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.2,
        output_size=num_classes,
        use_pos_enc=True,
        use_lstm=True
)

    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    model.apply(initialize_weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tft(
        model=model,
        dataloader=tft_dataloader,
        num_epochs=500,
        learning_rate=0.001,
        device=device,
        task='classification', # or 'regression'
        grad_clip=1.0,
        use_scheduler=True
    )

    print("\n[INFO] Training complete.\n")

    tft_model_path = os.path.join(derivatives_path, "tft_model.pth")
    torch.save(model.state_dict(), tft_model_path)
    print(f"[INFO] Saved TFT model in: {tft_model_path}")


def main():
 
    DATAPATH = './ds005106'
    SUBJECT_IDS = ['{:03d}'.format(i) for i in range(1, 52)]

    train_wgan_synthetic(
        datapath=DATAPATH,
        subject_ids=SUBJECT_IDS,
        n_rank=24,          
        batch_size=64,
        num_steps=1000,
        synth_file_name='synthetic_data.npz'
    )

    train_tft_main(
        datapath=DATAPATH,
        subject_ids=SUBJECT_IDS,
        synth_file_name='synthetic_data.npz',
        sequence_length=50,
        batch_size=64,
        num_outputs=200,
        num_epochs=1000
    )


if __name__ == "__main__":
    main()