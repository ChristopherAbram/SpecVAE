import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .vae import BaseVAE, ReLUlimit

EPS = 1e-12


class JointVAECriterium(nn.Module):
    def __init__(self, latent_spec, cont_capacity=None, disc_capacity=None):
        super().__init__()
        self.latent_spec = latent_spec
        self.is_continuous = 'cont' in self.latent_spec
        self.is_discrete = 'disc' in self.latent_spec
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.gnll = nn.GaussianNLLLoss(reduction='mean')
        self.num_steps = 0

    def step(self):
        self.num_steps += 1

    def forward(self, input, target, latent_dist):
        loss, _1, _2, _3, _4, _5, _6 = self.forward__(input, target, latent_dist)
        return loss

    def forward_(self, input, target, latent_dist):
        loss, recon, kld, _3, _4, _5, _6 = self.forward__(input, target, latent_dist)
        return loss, recon, kld

    def forward__(self, y_pred, y_true, latent_dist):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor,
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Compute reconstruction loss:
        var = torch.ones(y_pred.shape[0], 1, requires_grad=True).to(y_pred.device)
        recon_loss = self.gnll(y_pred, y_true, var)

        kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        cont_capacity_loss = 0
        disc_capacity_loss = 0
        kl_cont_means = 0
        kl_disc_losses = []

        if self.is_continuous:
            # Calculate KL divergence
            mean, logvar = latent_dist['cont']
            kl_cont_loss, kl_cont_means = self._kl_normal_loss(mean, logvar)
            # Linearly increase capacity of continuous channels
            cont_min, cont_max, cont_num_iters, cont_gamma = self.cont_capacity
            # Increase continuous capacity without exceeding cont_max
            cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
            cont_cap_current = min(cont_cap_current, cont_max)
            # Calculate continuous capacity loss
            cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

        if self.is_discrete:
            # Calculate KL divergence
            kl_disc_loss, kl_disc_losses = self._kl_multiple_discrete_loss(latent_dist['disc'])
            # Linearly increase capacity of discrete channels
            disc_min, disc_max, disc_num_iters, disc_gamma = self.disc_capacity
            # Increase discrete capacity without exceeding disc_max or theoretical
            # maximum (i.e. sum of log of dimension of each discrete variable)
            disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            # Require float conversion here to not end up with numpy float
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.latent_spec['disc']])
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)
            # Calculate discrete capacity loss
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss
        # Calculate total loss
        loss = (recon_loss + cont_capacity_loss + disc_capacity_loss).mean()
        return loss, recon_loss, kl_loss, kl_cont_means, kl_disc_losses, cont_capacity_loss, disc_capacity_loss

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)
        return kl_loss, kl_means

    def _kl_multiple_discrete_loss(self, alphas):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]
        # Total loss is sum of kl loss for each discrete latent
        kl_loss = torch.sum(torch.cat(kl_losses))
        return kl_loss, kl_losses

    def _kl_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)]).to(alpha.device)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss


class JointVAESigmoidCriterium(JointVAECriterium):
    def __init__(self, latent_spec, cont_capacity=None, disc_capacity=None):
        super(JointVAESigmoidCriterium, self).__init__(latent_spec, cont_capacity, disc_capacity)

    def forward_(self, input, target, latent_dist):
        loss, recon, kld, _3, _4, _5, _6 = self.forward__(input, target, latent_dist)
        return loss, recon, kld

    def forward__(self, y_pred, y_true, latent_dist):
        recon_loss = torch.sum(F.binary_cross_entropy(y_pred, y_true, reduction='none'), dim=1)

        kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        cont_capacity_loss = 0
        disc_capacity_loss = 0
        kl_cont_means = 0
        kl_disc_losses = []

        if self.is_continuous:
            # Calculate KL divergence
            mean, logvar = latent_dist['cont']
            kl_cont_loss, kl_cont_means = self._kl_normal_loss(mean, logvar)
            # Linearly increase capacity of continuous channels
            cont_min, cont_max, cont_num_iters, cont_gamma = self.cont_capacity
            # Increase continuous capacity without exceeding cont_max
            cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
            cont_cap_current = min(cont_cap_current, cont_max)
            # Calculate continuous capacity loss
            cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)

        if self.is_discrete:
            # Calculate KL divergence
            kl_disc_loss, kl_disc_losses = self._kl_multiple_discrete_loss(latent_dist['disc'])
            # Linearly increase capacity of discrete channels
            disc_min, disc_max, disc_num_iters, disc_gamma = self.disc_capacity
            # Increase discrete capacity without exceeding disc_max or theoretical
            # maximum (i.e. sum of log of dimension of each discrete variable)
            disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            # Require float conversion here to not end up with numpy float
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.latent_spec['disc']])
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)
            # Calculate discrete capacity loss
            disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss
        # Calculate total loss
        loss = (recon_loss + cont_capacity_loss + disc_capacity_loss).mean()
        return loss, recon_loss.mean(), kl_loss, kl_cont_means, kl_disc_losses, cont_capacity_loss, disc_capacity_loss





class JointVAE(BaseVAE):
    def __init__(self, config, device=None):
        """
        The impl of VAE from "Learning Disentangled Joint Continuous and Discrete Representations".

        Parameters
        ----------
        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        temperature : float
            Temperature for gumbel softmax distribution.

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.

        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.
        """
        super(JointVAE, self).__init__(config, device)
        # Parameters:
        self.name = self.get_attribute('name', required=False, default='joint_vae')
        self.limit = self.get_attribute('limit')
        self.latent_spec = self.get_attribute('latent_spec')
        self.temperature = self.get_attribute('temperature')
        self.cont_capacity = self.get_attribute('cont_capacity', required=False)
        self.disc_capacity = self.get_attribute('disc_capacity', required=False)
        self.is_continuous = 'cont' in self.latent_spec
        self.is_discrete = 'disc' in self.latent_spec
        self.loss = None

        # Calculate dimensions of latent distribution:
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

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
        self.encoder_ = nn.Sequential(OrderedDict(encoder_layers))

        # Encode parameters of latent distribution
        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        if self.is_continuous:
            self.en_mu = nn.Linear(in_dim, self.latent_cont_dim)
            self.en_mu_batchnorm = nn.BatchNorm1d(self.latent_cont_dim)
            self.en_log_var = nn.Linear(in_dim, self.latent_cont_dim)
            self.en_log_var_batchnorm = nn.BatchNorm1d(self.latent_cont_dim)
        
        # Linear layer for each of the categorical distributions
        if self.is_discrete:
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(in_dim, disc_dim))
            self.en_alphas = nn.ModuleList(fc_alphas)

        # First layer of decoder:
        decoder_layers_num = len(self.decoder_layer_config)
        decoder_layers = []
        in_dim, out_dim = self.latent_dim, self.decoder_layer_config[1]
        decoder_layers.append(('de_lin_%d' % 1, nn.Linear(in_dim, out_dim)))
        decoder_layers.append(('de_lin_batchnorm_%d' % 1, nn.BatchNorm1d(out_dim)))
        decoder_layers.append(('de_act_%d' % 1, nn.ReLU()))

        # Decoder layers:
        if encoder_layers_num >= 3:
            for i in range(2, decoder_layers_num - 1):
                in_dim, out_dim = self.decoder_layer_config[i - 1], self.decoder_layer_config[i]
                decoder_layers.append(('de_lin_%d' % i, nn.Linear(in_dim, out_dim)))
                decoder_layers.append(('de_lin_batchnorm_%d' % i, nn.BatchNorm1d(out_dim)))
                decoder_layers.append(('de_act_%d' % i, nn.ReLU()))

        # Last layer of decoder:
        in_dim, out_dim = \
            self.decoder_layer_config[decoder_layers_num - 2], \
            self.decoder_layer_config[decoder_layers_num - 1]
        decoder_layers.append(('de_lin_%d' % (decoder_layers_num - 1), nn.Linear(in_dim, out_dim)))
        decoder_layers.append(('de_act_%d' % (decoder_layers_num - 1), ReLUlimit(self.limit)))
        
        self.decoder = nn.Sequential(OrderedDict(decoder_layers))
        self.loss = JointVAECriterium(self.latent_spec, self.cont_capacity, self.disc_capacity)

    def encode(self, x):
        h = self.encoder_(x)
        latent_dist = {}
        if self.is_continuous:
            latent_dist['cont'] = [self.en_mu(h), self.en_log_var(h)]
        if self.is_discrete:
            latent_dist['disc'] = []
            for alpha in self.en_alphas:
                latent_dist['disc'].append(F.softmax(alpha(h), dim=1))
        return latent_dist

    def reparameterize(self, latent_dist):
        """
        Samples from latent distribution using the reparameterization trick.

        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []
        if self.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_sample = self.sample_normal(mean, logvar)
            latent_sample.append(cont_sample)
        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)
        return torch.cat(latent_sample, dim=1)

    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_().to(std.device)
            return mean + std * eps
        else:
            return mean

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size()).to(alpha.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            return one_hot_samples.to(alpha.device)

    def decode(self, latent_sample):
        return self.decoder(latent_sample)

    def forward(self, x):
        x, _, __ = self.forward_(x)
        return x

    def forward_(self, x):
        latent_dist = self.encode(x)
        latent_sample = self.reparameterize(latent_dist)
        return self.decode(latent_sample), latent_sample, latent_dist
    



class JointVAESigmoid(JointVAE):
    def __init__(self, config, device=None):
        super(JointVAESigmoid, self).__init__(config, device)

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
        self.encoder_ = nn.Sequential(OrderedDict(encoder_layers))

        # Encode parameters of latent distribution
        # Latent space layer (mu & log_var):
        in_dim, out_dim = \
            self.encoder_layer_config[encoder_layers_num - 2], \
            self.encoder_layer_config[encoder_layers_num - 1]
        if self.is_continuous:
            self.en_mu = nn.Linear(in_dim, self.latent_cont_dim)
            self.en_mu_batchnorm = nn.BatchNorm1d(self.latent_cont_dim)
            self.en_log_var = nn.Linear(in_dim, self.latent_cont_dim)
            self.en_log_var_batchnorm = nn.BatchNorm1d(self.latent_cont_dim)
        
        # Linear layer for each of the categorical distributions
        if self.is_discrete:
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(in_dim, disc_dim))
            self.en_alphas = nn.ModuleList(fc_alphas)

        # First layer of decoder:
        decoder_layers_num = len(self.decoder_layer_config)
        decoder_layers = []
        in_dim, out_dim = self.latent_dim, self.decoder_layer_config[1]
        decoder_layers.append(('de_lin_%d' % 1, nn.Linear(in_dim, out_dim)))
        decoder_layers.append(('de_lin_batchnorm_%d' % 1, nn.BatchNorm1d(out_dim)))
        decoder_layers.append(('de_act_%d' % 1, nn.ReLU()))

        # Decoder layers:
        if encoder_layers_num >= 3:
            for i in range(2, decoder_layers_num - 1):
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
        self.loss = JointVAESigmoidCriterium(self.latent_spec, self.cont_capacity, self.disc_capacity)

    def encode(self, x):
        h = self.encoder_(x)
        latent_dist = {}
        if self.is_continuous:
            latent_dist['cont'] = [
                self.en_mu_batchnorm(self.en_mu(h)), 
                self.en_log_var_batchnorm(self.en_log_var(h))]
        if self.is_discrete:
            latent_dist['disc'] = []
            for alpha in self.en_alphas:
                latent_dist['disc'].append(F.softmax(alpha(h), dim=1))
        return latent_dist
