import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import logging
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(
    filename=os.path.join('./ds005106/derivatives/preprocessing', 'training.log'),
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

#Generate a number of signals randomly.
def generate_signals(G_net, n_signals, device = device):

    noise = torch.randn((n_signals, G_net.nz), device = device)
    fake_signals = G_net.forward_fake_signals(noise)

    return fake_signals

def sequential_init_w(i):
    if isinstance(i, nn.Conv1d):
        nn.init.normal_(i.weight.data, 0.0, 0.02)

def wgan_gp_init_w(model):
    for i in model.modules():
      if isinstance(i, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
        nn.init.normal_(i.weight.data, 0.0, 0.02)

class WGAN_GP_Generator(nn.Module):

    def __init__(self, nz, ngf, s_length, device = None, upsample = False):
        super().__init__()
        self.in_ch = ngf
        self.out_ch = ngf
        self.h_ch = self.out_ch
        self.nz = nz
        self.ngf = ngf
        self.kernel_size = 2
        self.s_length = s_length
        self.G_err_a = []
        self.count = 0
        self.device = device

        self.l1 = nn.Linear(self.nz, self.s_length * self.ngf)

        #First
        #Shortcut Layer
        self.l_sc1 = self.in_ch != self.out_ch or True

        self.c1 = nn.Conv1d(self.in_ch, self.h_ch, self.kernel_size, 1)
        nn.init.normal_(self.c1.weight.data, 0.0, 0.02)

        layers1 = [nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, 1),
                   nn.BatchNorm1d(self.in_ch),
                   nn.BatchNorm1d(self.h_ch),
                   nn.LeakyReLU(0.2),
                   nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc1:
            layers1.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, padding = 0))

        self.s1 = nn.Sequential(*layers1)

        #Second
        #Shortcut Layer
        self.l_sc2 = self.in_ch != self.out_ch or False

        self.c2 = nn.Conv1d(self.in_ch, self.h_ch, self.kernel_size, 1)
        nn.init.normal_(self.c2.weight.data, 0.0, 0.02)

        layers2 = [nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, 1),
                   nn.BatchNorm1d(self.in_ch),
                   nn.BatchNorm1d(self.h_ch),
                   nn.LeakyReLU(0.2),
                   nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc2:
            layers2.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, padding = 0))

        self.s2 = nn.Sequential(*layers2)

        #Third
        #Shortcut Layer
        self.l_sc3 = self.in_ch != self.out_ch or True

        self.c3 = nn.Conv1d(self.in_ch, self.h_ch, self.kernel_size, 1)
        nn.init.normal_(self.c3.weight.data, 0.0, 0.02)

        layers3 = [nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, 1),
                   nn.BatchNorm1d(self.in_ch),
                   nn.BatchNorm1d(self.h_ch),
                   nn.LeakyReLU(0.2),
                   nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc3:
            layers3.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, padding = 0))

        self.s3 = nn.Sequential(*layers3)

        #Fourth
        #Shortcut Layer
        self.l_sc4 = self.in_ch != self.out_ch or False

        self.c4 = nn.Conv1d(self.in_ch, self.h_ch, self.kernel_size, 1)
        nn.init.normal_(self.c4.weight.data, 0.0, 0.02)

        layers4 = [nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, 1),
                   nn.BatchNorm1d(self.in_ch),
                   nn.BatchNorm1d(self.h_ch),
                   nn.LeakyReLU(0.2),
                   nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc4:
            layers4.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, padding = 0))

        self.s4 = nn.Sequential(*layers4)

        self.c13 = nn.Conv1d(self.ngf, 32, 1, 1, 0)
        self.end = nn.Linear(42, 16)

        #Initialise the weights
        self.s1.apply(sequential_init_w)
        self.s2.apply(sequential_init_w)
        self.s3.apply(sequential_init_w)
        self.s4.apply(sequential_init_w)

        nn.init.normal_(self.l1.weight.data, 0.0, 0.02)
        nn.init.normal_(self.c13.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)

    def forward(self, x):

        h = x

        h = self.c1(F.interpolate(h, scale_factor = 2, mode = 'linear', align_corners = False))
        h = self.s1(h)

        h = self.c2(h)
        h = self.s2(h)

        h = self.c3(F.interpolate(h, scale_factor = 2, mode = 'linear', align_corners = False))
        h = self.s3(h)

        h = self.c4(h)
        h = self.s4(h)

        return h

    def forward_fake_signals(self, x):

        h = self.l1(x)
        h = h.view(x.shape[0], self.ngf, self.s_length)

        h = self.c1(h)
        h = self.s1(h)
        h = self.c2(h)
        h = self.s2(h)
        h = self.c3(h)
        h = self.s3(h)
        h = self.c4(h)
        h = self.s4(h)

        h = self.c13(h)
        h = self.end(h)

        return h

    def train_one_step(self, batch_r, D_net, G_net, G_opt, log_data, device = None):

        batch_size = batch_r[0].shape[0]

        fake_signals = generate_signals(G_net, n_signals = batch_size, device = self.device)
        out_fake_signals = D_net(fake_signals)

        self.zero_grad()

        G_err = -out_fake_signals.mean()
        G_err.backward()
        G_opt.step()

        self.G_err_a.append(G_err.item())

        self.count += 1

        add_metric(log_data, 'Generator_err', G_err.item(), group='loss')

        return log_data

class WGAN_GP_Discriminator(nn.Module):

    def __init__(self, ndf, gp_scale, stride = 1, downsample = False):

        super().__init__()
        self.in_ch = ndf
        self.out_ch = ndf
        self.h_ch = self.in_ch
        self.kernel_size = 2
        self.stride = stride
        self.ndf = ndf
        self.gp_scale = gp_scale
        self.count = 0
        self.D_err_a = []

        #First
        #Shortcut Layer
        self.l_sc1 = (self.in_ch != self.out_ch) or False

        layers1 = [nn.Conv1d(32, self.out_ch, self.kernel_size, self.stride),
                   nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, self.stride),
                   nn.LeakyReLU(0.2),
                   nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc1:
            layers1.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, 0))

        self.s1 = nn.Sequential(*layers1)

        #Second
        #Shortcut Layer
        self.l_sc2 = (self.in_ch != self.out_ch) or True

        layers2 = [nn.Conv1d(self.in_ch, self.out_ch, self.kernel_size, self.stride),
                  nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, self.stride),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc2:
            layers2.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, 0))

        self.s2 = nn.Sequential(*layers2)

        #Third
        #Shortcut Layer
        self.l_sc3 = (self.in_ch != self.out_ch) or False

        layers3 = [nn.Conv1d(self.in_ch, self.out_ch, self.kernel_size, self.stride),
                  nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, self.stride),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc3:
            layers3.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, 0))

        self.s3 = nn.Sequential(*layers3)

        #Fourth
        #Shortcut Layer
        self.l_sc4 = (self.in_ch != self.out_ch) or True

        layers4 = [nn.Conv1d(self.in_ch, self.out_ch, self.kernel_size, self.stride),
                  nn.Conv1d(self.h_ch, self.out_ch, self.kernel_size, self.stride),
                  nn.LeakyReLU(0.2),
                  nn.Dropout(0.2)]

        #Shortcut layer
        if self.l_sc4:
            layers4.append(nn.Conv1d(self.in_ch, self.out_ch, 1, 1, 0))

        self.s4 = nn.Sequential(*layers4)

        self.c = nn.Conv1d(self.ndf, 32, 1, 1, 0)
        self.end = nn.Linear(8, 1)

        self.s1.apply(sequential_init_w)
        self.s2.apply(sequential_init_w)
        self.s3.apply(sequential_init_w)
        self.s4.apply(sequential_init_w)

        nn.init.normal_(self.c.weight.data, 0.0, 0.02)
        nn.init.normal_(self.end.weight.data, 0.0, 0.02)


    def forward(self, x):

        h = x
        h = self.s1(h)

        h = self.s2(h)
        h = F.avg_pool1d(h, 2)

        h = self.s3(h)

        h = self.s4(h)
        h = F.avg_pool1d(h, 2)

        return h

    def forward_discriminator(self, x):

        x = x.float()

        h = self.s1(x)
        h = self.s2(h)
        h = self.s3(h)
        h = self.s4(h)
        h = self.c(h)
        h = self.end(h)

        return h

    def train_one_step(self, batch_r, G_net, D_opt, log_data, device = None):

        batch_size = batch_r.shape[0]

        real_signals = batch_r

        out_real_signals = self.forward_discriminator(real_signals)

        fake_signals = generate_signals(G_net, n_signals = batch_size, device = device).detach()
        out_fake_signals = self.forward_discriminator(fake_signals)
        D_opt.zero_grad()

        D_err = -1.0 * out_real_signals.mean() + out_fake_signals.mean()
        D_err_gp = self.gp_loss(self.gp_scale, real_signals = real_signals, fake_signals = fake_signals)

        D_err_total = D_err + D_err_gp
        D_err_total.backward()
        D_opt.step()

        if (self.count != 0 and self.count % 5 == 0):
            self.D_err_a.append(D_err_total.item())
            add_metric(log_data, 'Discriminator_err', D_err.item(), group='loss')
            add_metric(log_data, 'Discriminator_err_gp', D_err_gp.item(), group='loss')

        self.count += 1

        return log_data

    #Gradient penalty loss
    def gp_loss(self, gp_scale, real_signals, fake_signals):

        N, _, L = real_signals.shape
        device = real_signals.device

        a = torch.rand(N, 1)
        a = a.expand(N, int(real_signals.nelement() / N)).contiguous()
        a = a.view(N, 32, L)
        a = a.to(device)

        i = a * real_signals.detach() + ((1 - a) * fake_signals.detach())
        i = i.to(device)
        i.requires_grad_(True)

        disc_i = self.forward_discriminator(i)
        grad = autograd.grad(outputs=disc_i, inputs=i, grad_outputs=torch.ones(disc_i.size()).to(device), create_graph = True, retain_graph = True, only_inputs = True)[0]
        grad = grad.view(grad.size(0), -1)

        gp = ((grad.norm(2, dim=1) - 1)**2).mean() * gp_scale

        return gp

def log_print(step_g, log_data, dataset_size, n_epochs, n_steps):

    epoch = max(int(step_g / dataset_size), 1)
    headers = ["Epoch", "Discriminator_err", "Discriminator_err_gp", "Generator_err", "lr_Discriminator", "lr_Generator"]
    values = [
        epoch,
        log_data.get('Discriminator_err', {}).get('v', 0),
        log_data.get('Discriminator_err_gp', {}).get('v', 0),
        log_data.get('Generator_err', {}).get('v', 0),
        log_data.get('lr_Discriminator', {}).get('v', 0),
        log_data.get('lr_Generator', {}).get('v', 0)
    ]

    # Formattazione delle metriche con precisione limitata
    formatted_metrics = " | ".join([f"{value:.4f}" for value in values[1:]])
    log_message = f"Epoch {values[0]} of {n_epochs} | {formatted_metrics}"

    print(log_message)
    logging.info(log_message)

def add_metric(log_data, name, v, group = None, precision = 5):
     log_data[name] = {'v': v}

def linear_decay(optimizer, step_g, lr_range, lr_range_step):

    v0, v1 = lr_range
    s0, s1 = lr_range_step

    if step_g <= s0:
        lr_update = v0

    elif step_g >= s1:
        lr_update = v1

    else:
        scale = (step_g - s0) / (s1 - s0)
        lr_update = v0 + scale * (v1 - v0)

    optimizer.param_groups[0]['lr'] = lr_update

    return lr_update

def step(D_opt, G_opt, lr_D, lr_G, step_init, log_data, step_g, n_steps):

    lr_D = linear_decay(optimizer=D_opt, step_g=step_g, lr_range=(lr_D, 0.0), lr_range_step=(step_init, n_steps))
    lr_G = linear_decay(optimizer=G_opt, step_g=step_g, lr_range=(lr_G, 0.0), lr_range_step=(step_init, n_steps))

    add_metric(log_data, 'lr_Discriminator', lr_D, group = 'lr', precision = 5)
    add_metric(log_data, 'lr_Generator', lr_G, group = 'lr', precision = 5)

    return log_data

def data_fetch(dataloader_i, dataloader, device):

    try:
        batch_r = next(dataloader_i)

    except StopIteration:
        dataloader_i = iter(dataloader)
        batch_r = next(dataloader_i)

    batch_r = batch_r[0].to(device)

    return batch_r

def train(D_net, G_net, D_opt, G_opt, lr_D, lr_G, n_ots, n_steps, dataloader):

    dataset_size= len(dataloader)
    n_epochs = max(int(n_steps / dataset_size), 1)
    step_init = 0
    step_g = 0
    dataloader_i = iter(dataloader)

    with tqdm(total=n_steps, desc="Training WGAN-GP", ncols=100) as pbar:
        for step_g in range(1, n_steps + 1):

            log_data = {}

            #One training step
            for i in range(n_ots):
                batch_r = data_fetch(dataloader_i = dataloader_i, dataloader = dataloader, device = device)

                #Discriminator update
                log_data = D_net.train_one_step(batch_r = batch_r, G_net = G_net, D_opt=D_opt, log_data = log_data, device = device)

                #Generator update
                if i == (n_ots - 1):
                    log_data = G_net.train_one_step(batch_r = batch_r, D_net = D_net, G_net = G_net, G_opt = G_opt, log_data = log_data, device = device)

            step_g += 1
            log_data = step(D_opt=D_opt, G_opt= G_opt, lr_D=lr_D, lr_G = lr_G, step_init=step_init, log_data=log_data, step_g=step_g, n_steps=n_steps)

            if step_g % 20 == 0:
                metrics = {key: f"{log_data[key]['v']:.4f}" for key in sorted(log_data.keys())}
                pbar.set_postfix(metrics)

                # Stampa log formattato
                log_print(step_g=step_g, log_data=log_data, dataset_size=dataset_size, n_epochs=n_epochs, n_steps=n_steps)

            pbar.update(1)

    print("Training finished")
    logging.info("Training finished")