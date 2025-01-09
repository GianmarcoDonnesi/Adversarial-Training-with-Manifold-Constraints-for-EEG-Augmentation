import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import logging  

logging.basicConfig(
    filename=os.path.join('./ds005106/derivatives/preprocessing', 'tft_training.log'),
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class GatedResidualNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()

        if input_size != output_size:

            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = None

    def forward(self, x):
       
        hidden = self.fc1(x)
        hidden = self.elu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)

        gate = self.sigmoid(self.gate(hidden))

        if self.skip_proj is not None:
            x_skip = self.skip_proj(x)
        else:
            x_skip = x

        return (hidden * gate) + (1 - gate) * x_skip

class SimpleFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
    
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)

        out = self.layernorm(residual + out)
        return out


class MultiHeadAttention(nn.Module):
  
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                         dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
    
        attn_output, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        out = self.dropout(attn_output)
        out = self.layernorm(x + out)  
        return out


class PositionalEncoding(nn.Module):
   
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class TFTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ff = SimpleFeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.grn = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)

    def forward(self, x, mask=None):
        x = self.self_attn(x, attn_mask=mask)
        x = self.ff(x)
        x = self.grn(x)
        return x

class TFTEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 d_model=64,
                 num_lstm_layers=2,
                 n_heads=4,
                 d_ff=256,
                 num_encoder_layers=2,
                 dropout=0.1,
                 use_positional_encoding=True,
                 use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm
        self.use_pos_enc = use_positional_encoding
        self.d_model = d_model

        if input_size != d_model:
            self.feature_proj = nn.Linear(input_size, d_model)
        else:
            self.feature_proj = None

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_lstm_layers,
                batch_first=True,
                bidirectional=False
            )

        if self.use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model, max_len=10000)

        self.layers = nn.ModuleList([
            TFTEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

    def forward(self, x, mask=None):
  
        if self.feature_proj is not None:
            x = self.feature_proj(x)  
       
        if self.use_lstm:
            x, (h_n, c_n) = self.lstm(x)  
        else:
            h_n = None
            c_n = None

        if self.use_pos_enc:
            x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x, (h_n, c_n)


class TFTDecoder(nn.Module):
    def __init__(self,
                 d_model=128,
                 output_size=200,
                 num_decoder_layers=1,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.output_size = output_size


        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_decoder_layers,
            batch_first=True
        )

        self.grn_ff = GatedResidualNetwork(d_model, d_model, d_model, dropout=dropout)
        self.output_proj = nn.Linear(d_model, output_size)

    def forward(self, enc_out, encoder_hidden):
        
        if encoder_hidden[0] is not None:
            
            h_0 = encoder_hidden[0][-1].unsqueeze(0)  
            c_0 = encoder_hidden[1][-1].unsqueeze(0)  
        else:
          
            batch_size = enc_out.size(0)
            h_0 = torch.zeros((1, batch_size, self.d_model), device=enc_out.device)
            c_0 = torch.zeros((1, batch_size, self.d_model), device=enc_out.device)

        context_vec = enc_out.mean(dim=1, keepdim=True) 

        dec_out, (h_dec, c_dec) = self.lstm(context_vec, (h_0, c_0))
       
        ff_out = self.grn_ff(dec_out) 
        logits = self.output_proj(ff_out).squeeze(1)  
        return logits



class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 input_size,
                 d_model=64,
                 num_lstm_layers=2,
                 n_heads=4,
                 d_ff=256,
                 num_encoder_layers=2,
                 num_decoder_layers=1,
                 dropout=0.1,
                 output_size=200,
                 use_pos_enc=True,
                 use_lstm=True):

        super().__init__()

        self.encoder = TFTEncoder(
            input_size=input_size,
            d_model=d_model,
            num_lstm_layers=num_lstm_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            use_positional_encoding=use_pos_enc,
            use_lstm=use_lstm
        )

        self.decoder = TFTDecoder(
            d_model=d_model,
            output_size=output_size,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout
        )

    def forward(self, x, mask=None):
        enc_out, (h_n, c_n) = self.encoder(x, mask=mask)  
        logits = self.decoder(enc_out, (h_n, c_n)) 
        return logits


def train_tft(model, dataloader, num_epochs=20, learning_rate=1e-3, device='cuda', task='classification', grad_clip=1.0, use_scheduler=True):
    model.to(device)
    if task == 'classification':
        criterion = nn.CrossEntropyLoss()
    elif task == 'regression':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Il task deve essere 'classification' o 'regression'.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if use_scheduler:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(dataloader), epochs=num_epochs, pct_start=0.3, anneal_strategy='cos', cycle_momentum=False )
    else:
        scheduler = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_sequences, batch_labels in progress_bar:
            torch.cuda.synchronize()
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_sequences)

            if task == 'classification':
                loss = criterion(outputs, batch_labels)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            else:
                loss = criterion(outputs, batch_labels.float())

            loss.backward()
            torch.cuda.synchronize()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if task == 'classification':
                progress_bar.set_postfix({'Loss': running_loss / (progress_bar.n + 1), 'Accuracy': 100 * correct / total})
            else:
                progress_bar.set_postfix({'Loss': running_loss / (progress_bar.n + 1)})

        epoch_loss = running_loss / len(dataloader)
        if task == 'classification':
            epoch_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Addestramento completato.")
    logging.info("Addestramento completato.")