import torch
from torch import nn
from torchvision import models
import numpy as np
from scipy import linalg
import torch.nn.functional as F
import faiss

m_inception = models.inception_v3(pretrained = True).eval()
m_inception.fc = nn.Identity()

vgg = models.vgg16(pretrained = True).features.eval().cuda()

#Frèchet Inception Distance
def fid(features_r, features_g):

    m_1, m_2 = features_r.mean(axis = 0), features_g.mean(axis = 0)
    s_1, s_2 = np.cov(features_r, rowvar = False), np.cov(features_g, rowvar = False)

    cov_m = linalg.sqrtm(s_1.dot(s_2))
    if np.iscomplexobj(cov_m):
        cov_m = cov_m.real

    fid1 = (m_1 - m_2).dot((m_1 - m_2))
    fid2 = np.trace(s_1 + s_2 - 2 * cov_m)
    return fid1 + fid2

#Inception Score
def inception_s(signals, m_inception, batch_size = 32, s = 10):
    scores = []

    for _ in range(s):
        s_probs = []
        for i in range(0, len(signals), batch_size):
            batch = signals[i:i + batch_size].cuda()

            with torch.no_grad():
                y = F.softmax(m_inception(batch), dim = 1)
            s_probs.append(y.cpu().numpy())

        s_probs = np.concatenate(s_probs)
        x = np.mean(s_probs, axis = 0)

        scores.append(np.exp(np.mean(np.sum(s_probs * np.log(s_probs / x), axis = 1))))

    mean = np.mean(scores)
    std = np.std(scores)
    return mean, std

#Kernel Inception Distance
def kid(features_r, features_g, d = 3, g = None, c_0 = 1):

    features_r = torch.tensor(features_r)
    features_g = torch.tensor(features_g)

    if g is None:
        g = 1.0 / features_r.shape[1]

    k_xx = (g*features_r.mm(features_r.t()) + c_0) ** d
    k_yy = (g*features_g.mm(features_g.t()) + c_0) ** d
    k_xy = (g*features_r.mm(features_g.t()) + c_0) ** d

    y = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

    return y

#Perceptual Path Length
def ppl(generator, latent_dim, vgg, num_samples = 100):

    d = []
    for _ in range(num_samples):
        z_1 = torch.randn(1, latent_dim).cuda()
        z_2 = z_1 + torch.randn(1, latent_dim).cuda() * 0.1
        signal_1, signal_2 = generator(z_1), generator(z_2)

        with torch.no_grad():
            x = vgg(signal_1).view(signal_1.size(0), -1)
            y = vgg(signal_2).view(signal_2.size(0), -1)

        d.append((torch.norm(x - y, dim = 1).mean()).item())

    return np.mean(d)

#Precision and Recall
def precision_recall(features_r, features_g, k = 3):

    i = faiss.IndexFlatL2(features_r.shape[1])
    i.add(features_r)

    #Precision
    x, y = i.search(features_g, k)
    precision = np.mean([np.any(i in y for i in range(len(features_r))) for _ in x])

    #Recall
    i.reset()
    i.add(features_g)
    x, y = i.search(features_r, k)
    recall = np.mean([np.any(i in y for i in range(len(features_g))) for _ in x])

    return precision, recall

def evaluate(real_signals, generated_signals, m_inception, vgg, m_vgg = None, metrics = ['FID', 'IS', 'KID', 'PPL', 'Precision and Recall']):

    r = {}
    if 'FID' in metrics:
        r['FID'] = fid(real_signals, generated_signals)

    if 'IS' in metrics:
        r['IS'] = inception_s(generated_signals, m_inception)

    if 'KID' in metrics:
        r['KID'] = kid(real_signals, generated_signals)

    if 'PPL' in metrics:
        r['PPL'] = ppl(m_vgg, generated_signals, vgg)

    if 'Precision and Recall' in metrics:
        r['Precision and Recall'] = precision_recall(real_signals, generated_signals, k = 3)
    else:
        print('Metric not used')
        return

    return r