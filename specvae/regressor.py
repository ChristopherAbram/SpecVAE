import torch
import torch.nn as nn
from collections import OrderedDict
from .model import BaseModel


class BaseRegressorCriterium(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(input, target)

class BaseRegressor(BaseModel):
    def __init__(self, config, device=None):
        super(BaseRegressor, self).__init__(config, device)
        self.name = 'regressor'
        self.build_layers()
        if self.device:
            self.to(self.device)

    def build_layers(self):
        layers_num = len(self.layer_config)
        layers = []
        for i in range(1, layers_num):
            in_dim, out_dim = self.layer_config[i - 1], self.layer_config[i]
            layers.append(('en_lin_%d' % i, nn.Linear(in_dim, out_dim)))
            layers.append(('en_lin_batchnorm_%d' % i, nn.BatchNorm1d(out_dim)))
            layers.append(('en_act_%d' % i, nn.ReLU()))
        in_dim, out_dim = self.layer_config[layers_num - 1], 1
        layers.append(('en_lin_%d' % layers_num, nn.Linear(in_dim, out_dim)))
        self.input_size = self.layer_config[0]
        self.layers = nn.Sequential(OrderedDict(layers))
        self.loss = BaseRegressorCriterium()

    def forward(self, x):
        return self.layers(x)

    def fit(self, X, y):
        # mock for sklearn...
        ...

    def predict(self, x):
        return self.forward(x)

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics._scorer import neg_mean_squared_error_scorer
        with torch.no_grad():
            if not torch.is_tensor(X):
                X = torch.from_numpy(X)
            return neg_mean_squared_error_scorer(self, X, y, sample_weight=sample_weight)

    def get_layer_string(self):
        return '-'.join(str(x) for x in self.layer_config) + '-1'
