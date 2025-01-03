import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from prepare_tft_data import load_preprocessed_data, prepare_tft_data
from tqdm.notebook import tqdm


#1) Gated Residual Network (GRN)

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network come descritto nel paper TFT.
    - Esegue una trasformazione non lineare su input e poi un gating,
      con skip connection verso l'uscita.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # gating + skip
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()

        # per lo skip connection
        if input_size != output_size:
            # Proiezione del residuo (skip)
            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = None

    def forward(self, x):
        # x.shape = (batch_size, seq_length, input_size) oppure (batch_size, input_size)
        hidden = self.fc1(x)
        hidden = self.elu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)

        # gating
        gate = self.sigmoid(self.gate(hidden))

        # skip connection
        if self.skip_proj is not None:
            x_skip = self.skip_proj(x)
        else:
            x_skip = x

        return (hidden * gate) + (1 - gate) * x_skip



#2) Multi-Head Attention


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention base per TFT.
    Utilizza la built-in nn.MultiheadAttention di PyTorch con batch_first=True.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x.shape = (batch_size, seq_length, d_model)
        """
        attn_output, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        out = self.dropout(attn_output)
        out = self.layernorm(x + out)  # residuo
        return out



#3) Positional Encoding (sin/cos standard)

class PositionalEncoding(nn.Module):
    """
    Implementazione standard di Positional Encoding (Transformer).
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # dimensioni pari
        pe[:, 1::2] = torch.cos(position * div_term)  # dimensioni dispari
        pe = pe.unsqueeze(0)  # shape = (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x.shape = (batch_size, seq_length, d_model)
        Ritorna x + positional_encoding
        """
        seq_len = x.size(1)
        # Aggiunge la PE alle feature
        # self.pe[:, :seq_len, :] => shape = (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


#4) ENCODER + DECODER LSTM

class TFTEncoder(nn.Module):
    """
    Encoder del TFT con:
    - Multi-layer LSTM (num_encoder_layers) per catturare info temporali
    - (Opzionale) Positional Encoding
    - Multi-Head Self-Attention
    - Gated Residual post-attn
    """
    def __init__(self,
                 input_size,
                 hidden_size=64,
                 num_heads=4,
                 dropout=0.1,
                 num_encoder_layers=2,
                 use_positional_encoding=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.use_positional_encoding = use_positional_encoding

        # LSTM multi-layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            batch_first=True,
            bidirectional=False
        )

        # Positional Encoding (opzionale)
        if self.use_positional_encoding:
            self.pos_enc = PositionalEncoding(d_model=hidden_size)
        else:
            self.pos_enc = None

        # Self-attention + residual
        self.self_attn = MultiHeadAttention(d_model=hidden_size, n_heads=num_heads, dropout=dropout)

        # GRN post-attn
        self.grn_post_attn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout
        )

    def forward(self, x):
        """
        x.shape = (batch_size, seq_length, input_size)
        Ritorna hidden_seq, (h_n, c_n) come ultimi stati LSTM
        """
        # LSTM multi-layer
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out.shape = (batch_size, seq_length, hidden_size)

        # Positional Encoding
        if self.pos_enc is not None:
            lstm_out = self.pos_enc(lstm_out)

        # Self-Attention
        attn_out = self.self_attn(lstm_out)

        # Gated Residual
        enc_out = self.grn_post_attn(attn_out)

        return enc_out, (h_n, c_n)


class TFTDecoder(nn.Module):
    """
    Decoder LSTM che riceve l'ultimo hidden state dell'encoder
    e produce la predizione.
    """
    def __init__(self, hidden_size=64, output_size=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Decoder LSTM (singolo layer)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # GRN feedforward finale
        self.grn_ff = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=dropout
        )

        # Proiezione finale
        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, enc_out, encoder_hidden):
        """
        enc_out: (batch_size, seq_length, hidden_size) [non sempre usato in uno schema semplificato]
        encoder_hidden: (h_n, c_n) dimensioni => (num_layers, batch_size, hidden_size)
        Ritorna logits => (batch_size, output_size)

        """
        # Usiamo lo stato hidden dell'encoder come initial state del decoder
        # h_n shape: (num_encoder_layers, batch_size, hidden_size)
        # Per la LSTM decoder, prendiamo l'ultimo layer dell'encoder:
        h_0 = encoder_hidden[0][-1].unsqueeze(0)
        c_0 = encoder_hidden[1][-1].unsqueeze(0)

        batch_size, seq_len, _ = enc_out.shape
        context_vec = torch.mean(enc_out, dim=1, keepdim=True)  # (batch_size, 1, hidden_size)

        dec_out, (h_dec, c_dec) = self.lstm(context_vec, (h_0, c_0))
        # dec_out.shape = (batch_size, 1, hidden_size)

        # GRN feedforward
        ff_out = self.grn_ff(dec_out)  # shape (batch_size, 1, hidden_size)

        logits = self.output_proj(ff_out).squeeze(1)  # (batch_size, output_size)

        return logits



class TFTRefinedModel(nn.Module):
    """
    Versione raffinata del TFT:
    - Encoder LSTM multilayer + Positional Encoding + Self-Attn
    - Decoder LSTM semplice (1 step)
    - Gated Residual Networks
    - Output finale (classif/regressione)
    """
    def __init__(self,
                 input_size,       # dimensione input di ogni time step
                 hidden_size=64,
                 num_heads=4,
                 num_outputs=2,
                 num_encoder_layers=2,
                 dropout=0.1,
                 use_pos_enc=True):
        super().__init__()
        self.encoder = TFTEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            num_encoder_layers=num_encoder_layers,
            use_positional_encoding=use_pos_enc
        )
        self.decoder = TFTDecoder(
            hidden_size=hidden_size,
            output_size=num_outputs,
            dropout=dropout
        )

    def forward(self, x):
        """
        x.shape = (batch_size, seq_length, input_size)
        """
        enc_out, (h_n, c_n) = self.encoder(x)  # (batch_size, seq_len, hidden_size), (h_n, c_n)
        logits = self.decoder(enc_out, (h_n, c_n))  # (batch_size, num_outputs)
        return logits


'''
if __name__ == "__main__":

    datapath = '/content/drive/MyDrive/AIRO/Projects/EAI_Project/ds005106'
    derivatives_path = os.path.join(datapath, 'derivatives', 'preprocessing')
    subject_ids = ['{:03d}'.format(i) for i in range(1, 10)]

    # Carica dati preprocessati
    loaded_data = load_preprocessed_data(subject_ids, derivatives_path)
    real_labels = loaded_data['labels']

    sequence_length = 10
    batch_size = 32

    # Prepara DataLoader TFT
    tft_dataloader = prepare_tft_data(
        loaded_data,
        sequence_length=sequence_length,
        batch_size=batch_size,
        shuffle=True
    )

    # Dimensione feature
    feature_dim = loaded_data['ts_features'].shape[1]

    # Inizializza il modello
    model = TFTRefinedModel(
        input_size=feature_dim,
        hidden_size=64,
        num_heads=4,
        num_outputs=5,
        num_encoder_layers=2,
        dropout=0.1,
        use_pos_enc=True
    )

    # Addestramento (classificazione) su CPU
    train_tft(
        model,
        tft_dataloader,
        num_epochs=10,
        learning_rate=1e-3,
        device='cpu',
        task='classification'
    )

    print("Training completato")
'''