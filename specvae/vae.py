import numpy as np
from collections import OrderedDict
from numpy.lib import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

import train, utils


class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return self.lambd(x)


class SampleZ(nn.Module):
    def __init__(self, device=None):
        super(SampleZ, self).__init__()
        self.device = device

    def forward(self, x):
        mu, log_sigma = x
        std = torch.exp(0.5 * log_sigma).to(self.device)
        with torch.no_grad():
            epsilon = torch.randn_like(std).to(self.device)
        return mu + std * epsilon


class ReLUlimit(nn.Module):
    def __init__(self, limit):
        super(ReLUlimit, self).__init__()
        self.limit = limit
    
    def forward(self, x):
        return torch.clamp(x, min=0., max=self.limit)


class BaseVAE(nn.Module):
    def __init__(self, device=None):
        super(BaseVAE, self).__init__()
        self.layer_config = None
        self.device = device
        self.name = 'vae'

    def latent_dim(self):
        return self.latent_dim

    def loss(self, y_true, y_pred, mu, log_var):
        # E[log P(X|z)]
        recon = torch.sum(F.binary_cross_entropy(y_pred, y_true.data, reduction='none'), dim=1)
        # D_KL(Q(z|X) || P(z|X))
        kld = 0.5 * torch.sum(
            torch.exp(log_var) + torch.square(mu) - 1. - log_var, dim=1)
        return (kld + recon).mean()

    def encode(self, x):
        x = self.encoder_(x)
        mu = F.relu(self.en_mu(x))
        log_var = F.relu(self.en_log_var(x))
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x, _, __ = self.forward_(x)
        return x

    def forward_(self, x):
        mu, log_var = self.encode(x)
        z = self.sample([mu, log_var])
        x = self.decode(z)
        return x, mu, log_var

    def get_layer_string(self):
        layers = self.layer_config
        layer_string = '-'.join(str(x) for x in layers[0]) + '-' + '-'.join(str(x) for x in np.array(layers[1])[1:])
        return layer_string

    def get_name(self):
        return self.name

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path, device=None):
        if device:
            model = torch.load(path, map_location=device)
        else:
            model = torch.load(path)
        model.device = device
        model.eval()
        return model


class SpecVEA(BaseVAE):
    def __init__(self, config, device=None):
        super(SpecVEA, self).__init__(device)
        self.name = 'specvae'
        self.trainer = None
        
        # Extract model parameters:
        if 'resolution' in config:
            self.resolution = config['resolution']
        else:
            raise ValueError('resolution parameter missing')
        if 'max_mz' in config:
            self.max_mz = config['max_mz']
        else:
            raise ValueError('max_mz parameter missing')
        
        if 'layer_config' in config:
            self.layer_config = config['layer_config']
        else:
            raise ValueError('layer_config parameter missing')

        # self.modcossim = metrics.ModifiedCosine(self.resolution, self.max_mz)
        
        # Build model layers
        self.layer_config = config['layer_config']
        self.encoder_layer_config = self.layer_config[0]
        self.decoder_layer_config = self.layer_config[1]

        # Encoder layers:
        encoder_layers_num = len(self.encoder_layer_config)
        encoder_layers = []
        for i in range(1, encoder_layers_num - 1):
            in_dim, out_dim = self.encoder_layer_config[i - 1], self.encoder_layer_config[i]
            encoder_layers.append(('en_lin_%d' % i, nn.Linear(in_dim, out_dim)))
            encoder_layers.append(('en_lin_batchnorm_%d' % i, nn.BatchNorm1d(out_dim)))
            encoder_layers.append(('en_act_%d' % i, nn.ReLU()))
        self.encoder_ = nn.Sequential(OrderedDict(encoder_layers))

        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        self.latent_dim = out_dim
        self.en_mu = nn.Linear(in_dim, out_dim)
        self.en_mu_batchnorm = nn.BatchNorm1d(out_dim)
        self.en_log_var = nn.Linear(in_dim, out_dim)
        self.en_log_var_batchnorm = nn.BatchNorm1d(out_dim)

        # Sample from N(0., 1.)
        self.sample = SampleZ()

        # Decoder layers:
        decoder_layers_num = len(self.decoder_layer_config)
        decoder_layers = []
        for i in range(1, decoder_layers_num - 1):
            in_dim, out_dim = self.decoder_layer_config[i - 1], self.decoder_layer_config[i]
            decoder_layers.append(('de_lin_%d' % i, nn.Linear(in_dim, out_dim)))
            decoder_layers.append(('de_lin_batchnorm_%d' % i, nn.BatchNorm1d(out_dim)))
            decoder_layers.append(('de_act_%d' % i, nn.ReLU()))

        # Last layer of decoder:
        in_dim, out_dim = \
            self.decoder_layer_config[decoder_layers_num - 2], \
            self.decoder_layer_config[decoder_layers_num - 1]
        decoder_layers.append(('de_lin_%d' % (decoder_layers_num - 1), nn.Linear(in_dim, out_dim)))
        decoder_layers.append(('de_act_%d' % (decoder_layers_num - 1), ReLUlimit(100.)))
        
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))

        if self.device:
            self.to(self.device)

    def encode(self, x):
        x = self.encoder_(x)
        mu = self.en_mu(x)
        mu = self.en_mu_batchnorm(mu)
        log_var = self.en_log_var(x)
        log_var = self.en_log_var_batchnorm(log_var)
        return mu, log_var

    def loss(self, y_true, y_pred, mu, log_var):
        # E[log P(X|z)]
        recon = -gaussian_likelihood(y_true, y_pred, self.device)
        # D_KL(Q(z|X) || P(z|X))
        kld = 0.5 * torch.sum(
            torch.exp(log_var) + torch.square(mu) - 1. - log_var, dim=1)
        return (kld + recon).mean()


def gaussian_likelihood(x_true, x_pred, device):
    scale = torch.exp(torch.Tensor([0.0])).to(device)
    dist = torch.distributions.Normal(x_pred, scale)
    log_p = dist.log_prob(x_true).to(device)
    return log_p.sum(dim=1)
