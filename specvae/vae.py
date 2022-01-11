import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


from .model import BaseModel


class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    
    def forward(self, x):
        return self.lambd(x)

class SampleZ(nn.Module):
    def forward(self, x):
        mu, log_sigma = x
        std = torch.exp(0.5 * log_sigma).to(mu.device)
        with torch.no_grad():
            epsilon = torch.randn_like(std).to(mu.device)
        return mu + std * epsilon

class ReLUlimit(nn.Module):
    def __init__(self, limit):
        super(ReLUlimit, self).__init__()
        self.limit = limit
    
    def forward(self, x):
        return torch.clamp(x, min=0., max=self.limit)

class GaussianNLLLoss(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = torch.Tensor([scale])

    def forward(self, input, target):
        scale = self.scale.to(input.device)
        dist = torch.distributions.Normal(target, scale)
        log_p = dist.log_prob(input).to(input.device)
        return -log_p.sum(dim=1)

class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, input, target, latent_dist):
        loss, _, __ = self.forward_(input, target, latent_dist)
        return loss

    def forward_(self, input, target, latent_dist):
        mean, logvar = latent_dist
        if torch.isnan(input).any():
            raise ValueError("input for VAELoss contains nan elements")
        # E[log P(X|z)]
        recon = torch.sum(F.binary_cross_entropy(input, target, reduction='none'), dim=1).mean()
        # D_KL(Q(z|X) || P(z|X))
        kld = self.beta * 0.5 * torch.sum(torch.exp(logvar) + torch.square(mean) - 1. - logvar, dim=1).mean()
        loss = recon + kld
        return loss, recon, kld

class VAELossGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnll = nn.GaussianNLLLoss(reduction='mean')

    def forward(self, input, target, latent_dist):
        loss, _, __ = self.forward_(input, target, latent_dist)
        return loss

    def forward_(self, input, target, latent_dist):
        mean, logvar = latent_dist
        # E[log P(X|z)]
        var = torch.ones(input.shape[0], 1, requires_grad=True).to(input.device)
        recon = self.gnll(input, target, var)
        # D_KL(Q(z|X) || P(z|X))
        kld = (0.5 * torch.sum(torch.exp(logvar) + torch.square(mean) - 1. - logvar, dim=1)).mean()
        loss = recon + kld
        return loss, recon, kld

class BaseVAE(BaseModel):
    def __init__(self, config, device=None):
        super().__init__(config, device)
        self.name = self.get_attribute('name', required=False, default='vae')

    def build_layers(self):
        self.encoder = None
        self.fc_mean = None
        self.fc_logvar = None
        self.decoder = None
        self.loss = None
        raise NotImplementedError("Build model method 'build_layers' is not implemented")

    def reparameterize(self, latent_dist):
        mean, logvar = latent_dist
        if self.training:
            z = self.sample(latent_dist)
        else:
            z = mean
        return z

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        x, _, __ = self.forward_(x)
        return x

    def forward_(self, x):
        latent_dist = self.encode(x)
        z = self.reparameterize(latent_dist)
        x = self.decode(z)
        return x, z, latent_dist

    def get_layer_string(self):
        layers = self.layer_config
        layer_string = '-'.join(str(x) for x in layers[0]) + '-' + '-'.join(str(x) for x in np.array(layers[1])[1:])
        return layer_string

class SpecVEA(BaseVAE):
    def __init__(self, config, device=None):
        super(SpecVEA, self).__init__(config, device)
        self.name = self.get_attribute('name', required=False, default='specvae')
        self.beta = self.get_attribute('beta', required=False, default=1.0)
        self.build_layers()
        if self.device:
            self.to(self.device)

    def build_layers(self):
        # Build model layers
        self.layer_config = self.config['layer_config']
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
        self.encoder = nn.Sequential(OrderedDict(encoder_layers))

        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        self.latent_dim = out_dim
        self.fc_mean = nn.Linear(in_dim, out_dim)
        self.mean_batchnorm = nn.BatchNorm1d(out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)
        self.logvar_batchnorm = nn.BatchNorm1d(out_dim)

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
        decoder_layers.append(('de_act_%d' % (decoder_layers_num - 1), nn.Sigmoid()))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))

        # Loss:
        self.input_size = self.encoder_layer_config[0]
        self.loss = VAELoss(self.beta)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mean_batchnorm(self.fc_mean(x))
        log_var = self.logvar_batchnorm(self.fc_logvar(x))
        return mu, log_var

class SpecGaussianVAE(BaseVAE):
    def __init__(self, config, device=None):
        super(SpecGaussianVAE, self).__init__(config, device)
        self.name = self.get_attribute('name', required=False, default='specgvae')
        self.limit = self.get_attribute('limit')
        self.build_layers()
        if self.device:
            self.to(self.device)

    def build_layers(self):
        # Build model layers
        self.layer_config = self.config['layer_config']
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
        self.encoder = nn.Sequential(OrderedDict(encoder_layers))

        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        self.latent_dim = out_dim
        self.fc_mean = nn.Linear(in_dim, out_dim)
        self.mean_batchnorm = nn.BatchNorm1d(out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)
        self.logvar_batchnorm = nn.BatchNorm1d(out_dim)

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
        decoder_layers.append(('de_act_%d' % (decoder_layers_num - 1), ReLUlimit(limit=self.limit)))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))

        # Loss:
        self.loss = VAELossGaussian()

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mean_batchnorm(self.fc_mean(x))
        log_var = self.logvar_batchnorm(self.fc_logvar(x))
        return mu, log_var


from .classifier import BaseClassifier, BaseClassifierCriterium
from .utils import get_attribute

class GaussianVAEandClassifierCriterium(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.vaeg_loss = VAELossGaussian()
        self.clf_loss = BaseClassifierCriterium(n_classes=self.n_classes)

    def forward(self, input, target, latent_dist):
        loss, _, __ = self.forward_(input, target, latent_dist)
        return loss

    def forward_(self, input, target, latent_dist, y_logits_pred, y_true):
        vae_loss, recon, kld = self.vaeg_loss.forward_(input, target, latent_dist)
        clf_loss = self.clf_loss(y_logits_pred, y_true)
        loss = vae_loss + clf_loss
        return loss, recon, kld, clf_loss

class JointSpecGaussianVAEandClassifier(BaseVAE):
    def __init__(self, config, device=None):
        super(JointSpecGaussianVAEandClassifier, self).__init__(config, device)
        self.name = self.get_attribute('name', required=False, default='joint_gvae_classifier')
        self.limit = self.get_attribute('limit')
        self.clf_config = self.get_attribute('clf_config')
        self.build_layers()
        if self.device:
            self.to(self.device)

    def build_layers(self):
        # Build model layers
        self.layer_config = self.config['layer_config']
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
        self.encoder = nn.Sequential(OrderedDict(encoder_layers))

        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        self.latent_dim = out_dim
        self.fc_mean = nn.Linear(in_dim, out_dim)
        self.mean_batchnorm = nn.BatchNorm1d(out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)
        self.logvar_batchnorm = nn.BatchNorm1d(out_dim)

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
        decoder_layers.append(('de_act_%d' % (decoder_layers_num - 1), ReLUlimit(limit=self.limit)))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))

        # Classification model:
        from dataset import Identity
        self.clf_model = BaseClassifier(config={
            'name':                 get_attribute(self.clf_config, 'name', required=False, default='InnerVAEClf'),
            'n_classes':            get_attribute(self.clf_config, 'n_classes'),
            'layer_config':         get_attribute(self.clf_config, 'layer_config'),
            'transform':            get_attribute(self.clf_config, 'transform', 
                                        required=False, default=tv.transforms.Compose([Identity()])),
            'target_column':        get_attribute(self.clf_config, 'target_column'),
            'target_column_id':     get_attribute(self.clf_config, 'target_column'),
            'input_columns':        get_attribute(self.clf_config, 'input_columns'),
            'types':                get_attribute(self.clf_config, 'types'),
            'class_subset':         get_attribute(self.clf_config, 'class_subset', required=False, default=[]),
            'dataset':              self.get_attribute('dataset', required=False)
        })
        # Loss:
        self.loss = GaussianVAEandClassifierCriterium(n_classes=self.clf_model.config['n_classes'])

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mean_batchnorm(self.fc_mean(x))
        log_var = self.logvar_batchnorm(self.fc_logvar(x))
        return mu, log_var

    def forward(self, x):
        x, _, __, ___ = self.forward_(x)
        return x

    def forward_(self, x):
        latent_dist = self.encode(x)
        logits, labels = self.clf_model.forward_(latent_dist[0])
        z = self.reparameterize(latent_dist)
        x = self.decode(z)
        return x, z, latent_dist, (logits, labels)

class VAEandClassifierCriterium(nn.Module):
    def __init__(self, n_classes=2, beta=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.beta = beta
        self.vae_loss = VAELoss(beta=beta)
        self.clf_loss = BaseClassifierCriterium(n_classes=self.n_classes)

    def forward(self, input, target, latent_dist):
        loss, _, __ = self.forward_(input, target, latent_dist)
        return loss

    def forward_(self, input, target, latent_dist, y_logits_pred, y_true):
        vae_loss, recon, kld = self.vae_loss.forward_(input, target, latent_dist)
        clf_loss = self.clf_loss(y_logits_pred, y_true)
        loss = vae_loss + clf_loss
        return loss, recon, kld, clf_loss


class VAEandClassifier(SpecVEA):
    def __init__(self, config, device=None):
        super().__init__(config, device)

    def build_layers(self):
        self.clf_config = self.get_attribute('clf_config')
        self.name = self.get_attribute('name', required=False, default='vae_clf')
        super().build_layers()
        # Classification model:
        from .dataset import Identity
        self.clf_model = BaseClassifier(config={
            'name':                 get_attribute(self.clf_config, 'name', required=False, default='inner_clf'),
            'n_classes':            get_attribute(self.clf_config, 'n_classes'),
            'layer_config':         get_attribute(self.clf_config, 'layer_config'),
            'transform':            get_attribute(self.clf_config, 'transform', 
                                        required=False, default=tv.transforms.Compose([Identity()])),
            'target_column':        get_attribute(self.clf_config, 'target_column'),
            'target_column_id':     get_attribute(self.clf_config, 'target_column'),
            'input_columns':        get_attribute(self.clf_config, 'input_columns'),
            'input_sizes':          get_attribute(self.clf_config, 'input_sizes'),
            'types':                get_attribute(self.clf_config, 'types'),
            'class_subset':         get_attribute(self.clf_config, 'class_subset', required=False, default=[]),
            'class_weights':        get_attribute(self.clf_config, 'class_weights', required=False, default=[]),
            'dataset':              self.get_attribute('dataset', required=False)
        })
        # Loss:
        self.loss = VAEandClassifierCriterium(n_classes=self.clf_model.config['n_classes'], beta=self.beta)

    def forward(self, x, feature=torch.tensor([])):
        x, _1, _2, _3 = self.forward_(x, feature)
        return x

    def forward_(self, x, feature):
        latent_dist = self.encode(x)
        z = self.reparameterize(latent_dist)
        if torch.numel(feature) == 0 and self.clf_model.input_size > self.latent_dim:
            size = self.clf_model.input_size - self.latent_dim
            feature = torch.zeros((x.shape[0], size), dtype=x.dtype).to(x.device)
            logits, labels = self.clf_model.forward_(torch.cat([z, feature], dim=1))
        elif torch.numel(feature) == 0 and self.clf_model.input_size == self.latent_dim:
            logits, labels = self.clf_model.forward_(z)
        else:
            logits, labels = self.clf_model.forward_(torch.cat([z, feature], dim=1))
        x = self.decode(z)
        return x, z, latent_dist, (logits, labels)

    def latent_with_features_(self, x):
        v_size = self.encoder_layer_config[0]
        x = torch.from_numpy(x) if not torch.is_tensor(x) else x
        latent_dist = self.encode(x[:,:v_size])
        z = self.reparameterize(latent_dist)
        if self.clf_model.input_size > self.latent_dim:
            z = torch.cat([z, x[:,v_size:]], dim=1)
        return z

    def predict(self, x):
        z = self.latent_with_features_(x)
        return self.clf_model.predict(z)

    def predict_log_proba(self, x):
        z = self.latent_with_features_(x)
        return self.clf_model.predict_log_proba(z)

    def predict_proba(self, x):
        z = self.latent_with_features_(x)
        return self.clf_model.predict_proba(z)

    def fit(self, X, y):
        # mock for sklearn...
        ...

    def score(self, X, y, sample_weight=None):
        Z = self.latent_with_features_(X)
        return self.clf_model.score(Z, y, sample_weight)


from .regressor import BaseRegressor, BaseRegressorCriterium

class GaussianVAEandRegressorCriterium(nn.Module):
    def __init__(self):
        super().__init__()
        self.vaeg_loss = VAELossGaussian()
        self.reg_loss = BaseRegressorCriterium()

    def forward(self, input, target, latent_dist):
        loss, _, __ = self.forward_(input, target, latent_dist)
        return loss

    def forward_(self, input, target, latent_dist, y_pred, y_true):
        vae_loss, recon, kld = self.vaeg_loss.forward_(input, target, latent_dist)
        reg_loss = self.reg_loss(y_pred, y_true)
        loss = vae_loss + reg_loss
        return loss, recon, kld, reg_loss

class JointSpecGaussianVAEandRegressor(BaseVAE):
    def __init__(self, config, device=None):
        super(JointSpecGaussianVAEandRegressor, self).__init__(config, device)
        self.name = self.get_attribute('name', required=False, default='joint_gvae_regressor')
        self.limit = self.get_attribute('limit')
        self.regressor_config = self.get_attribute('regressor_config')
        self.build_layers()
        if self.device:
            self.to(self.device)

    def build_layers(self):
        # Build model layers
        self.layer_config = self.config['layer_config']
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
        self.encoder = nn.Sequential(OrderedDict(encoder_layers))

        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        self.latent_dim = out_dim
        self.fc_mean = nn.Linear(in_dim, out_dim)
        self.mean_batchnorm = nn.BatchNorm1d(out_dim)
        self.fc_logvar = nn.Linear(in_dim, out_dim)
        self.logvar_batchnorm = nn.BatchNorm1d(out_dim)

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
        decoder_layers.append(('de_act_%d' % (decoder_layers_num - 1), ReLUlimit(limit=self.limit)))
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))

        # Classification model:
        from .dataset import Identity
        self.regressor_model = BaseRegressor(config={
            'name':                 get_attribute(self.regressor_config, 'name', required=False, default='InnerVAERegressor'),
            'layer_config':         get_attribute(self.regressor_config, 'layer_config'),
            'transform':            get_attribute(self.regressor_config, 'transform', 
                                        required=False, default=tv.transforms.Compose([Identity()])),
            'target_column':        get_attribute(self.regressor_config, 'target_column'),
            'input_columns':        get_attribute(self.regressor_config, 'input_columns'),
            'types':                get_attribute(self.regressor_config, 'types'),
            'dataset':              self.get_attribute('dataset', required=False)
        })
        # Loss:
        self.loss = GaussianVAEandRegressorCriterium()

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mean_batchnorm(self.fc_mean(x))
        log_var = self.logvar_batchnorm(self.fc_logvar(x))
        return mu, log_var

    def forward(self, x):
        x, _, __, ___ = self.forward_(x)
        return x

    def forward_(self, x):
        latent_dist = self.encode(x)
        y_pred = self.regressor_model(latent_dist[0])
        z = self.reparameterize(latent_dist)
        x = self.decode(z)
        return x, z, latent_dist, y_pred


class VAEandRegressorCriterium(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.vae_loss = VAELoss(beta=beta)
        self.reg_loss = BaseRegressorCriterium()

    def forward(self, input, target, latent_dist):
        loss, _1, _2, _3 = self.forward_(input, target, latent_dist)
        return loss

    def forward_(self, input, target, latent_dist, y_pred, y_true):
        vae_loss, recon, kld = self.vae_loss.forward_(input, target, latent_dist)
        reg_loss = self.reg_loss(y_pred, y_true)
        loss = vae_loss + reg_loss
        return loss, recon, kld, reg_loss
        

class VAEandRegressor(SpecVEA):
    def __init__(self, config, device=None):
        super().__init__(config, device)

    def build_layers(self):
        # Build model layers
        self.regressor_config = self.get_attribute('regressor_config')
        self.name = self.get_attribute('name', required=False, default='vae_reg')
        super().build_layers()
        # Classification model:
        from .dataset import Identity
        self.regressor_model = BaseRegressor(config={
            'name':                 get_attribute(self.regressor_config, 'name', required=False, default='inner_reg'),
            'layer_config':         get_attribute(self.regressor_config, 'layer_config'),
            'transform':            get_attribute(self.regressor_config, 'transform', 
                                        required=False, default=tv.transforms.Compose([Identity()])),
            'target_column':        get_attribute(self.regressor_config, 'target_column'),
            'input_columns':        get_attribute(self.regressor_config, 'input_columns'),
            'input_sizes':          get_attribute(self.regressor_config, 'input_sizes'),
            'types':                get_attribute(self.regressor_config, 'types'),
            'dataset':              self.get_attribute('dataset', required=False)
        })
        # Loss:
        self.loss = VAEandRegressorCriterium(beta=self.beta)

    def forward(self, x, feature=torch.tensor([])):
        x, _1, _2, _3 = self.forward_(x, feature)
        return x

    def forward_(self, x, feature):
        latent_dist = self.encode(x)
        z = self.reparameterize(latent_dist)
        if torch.numel(feature) == 0 and self.regressor_model.input_size > self.latent_dim:
            size = self.regressor_model.input_size - self.latent_dim
            feature = torch.zeros((x.shape[0], size), dtype=x.dtype).to(x.device)
            pred = self.regressor_model.forward(torch.cat([z, feature], dim=1))
        elif torch.numel(feature) == 0 and self.regressor_model.input_size == self.latent_dim:
            pred = self.regressor_model.forward(z)
        else:
            pred = self.regressor_model.forward(torch.cat([z, feature], dim=1))
        x = self.decode(z)
        return x, z, latent_dist, pred

    def latent_with_features_(self, x):
        v_size = self.encoder_layer_config[0]
        x = torch.from_numpy(x) if not torch.is_tensor(x) else x
        latent_dist = self.encode(x[:,:v_size])
        z = self.reparameterize(latent_dist)
        if self.regressor_model.input_size > self.latent_dim:
            z = torch.cat([z, x[:,v_size:]], dim=1)
        return z

    def fit(self, X, y):
        # mock for sklearn...
        ...

    def predict(self, x):
        z = self.latent_with_features_(x)
        return self.regressor_model.predict(z)

    def score(self, X, y, sample_weight=None):
        Z = self.latent_with_features_(X)
        return self.regressor_model.score(Z, y, sample_weight)
