import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .model import BaseModel


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        return super().forward(input, target.float())

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return super().forward(input, target.squeeze())

class BaseClassifierCriterium(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        if self.n_classes == 2:
            self.out = nn.Sigmoid()
            self.loss = BCEWithLogitsLoss(reduction='mean')
        elif self.n_classes > 2:
            self.out = nn.LogSoftmax()
            self.loss = CrossEntropyLoss(reduction='mean')

    def forward(self, input, target):
        return self.loss(input, target)

class BaseClassifier(BaseModel):
    def __init__(self, config, device=None):
        super(BaseClassifier, self).__init__(config, device)
        self.name = self.get_attribute('name', required=False, default='classifier')
        self.n_classes = self.get_attribute('n_classes')
        self.class_weights = self.get_attribute('class_weights', required=False)
        self.dropout = self.get_attribute('dropout', required=False)
        if self.dropout == 0.0:
            self.dropout = None

        self.build_layers()
        if self.device:
            self.to(self.device)

    def build_layers(self):
        layers_num = len(self.layer_config)
        layers = []
        for i in range(1, layers_num):
            in_dim, out_dim = self.layer_config[i - 1], self.layer_config[i]
            layers.append(('lin_%d' % i, nn.Linear(in_dim, out_dim)))
            layers.append(('lin_batchnorm_%d' % i, nn.BatchNorm1d(out_dim)))
            # Add dropout before activation func. of first and last layer:
            if self.dropout and (i == layers_num - 1 or i == 1):
                layers.append(('drp_%d' % i, nn.Dropout(p=self.dropout)))
            layers.append(('act_%d' % i, nn.ReLU()))

        in_dim, out_dim = self.layer_config[layers_num - 1], self.n_classes if self.n_classes > 2 else 1
        layers.append(('lin_%d' % layers_num, nn.Linear(in_dim, out_dim)))
        # layers.append(('lin_batchnorm_%d' % layers_num, nn.BatchNorm1d(out_dim)))

        self.layers = nn.Sequential(OrderedDict(layers))
        self.loss = BaseClassifierCriterium(n_classes=self.n_classes)
        self.out = self.loss.out

    def forward_(self, x):
        x = self.layers(x)
        out = self.out(x)
        out = torch.argmax(out, dim=-1).unsqueeze(1) if self.n_classes > 2 else torch.round(out).long()
        return x, out

    def forward(self, x):
        _, x = self.forward_(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def predict_log_proba(self, x):
        if self.n_classes == 2:
            return F.logsigmoid(self.out(self.layers(x)))
        else:
            return self.out(self.layers(x))

    def predict_proba(self, x):
        if self.n_classes == 2:
            return self.out(self.layers(x))
        else:
            return F.softmax(self.layers(x))

    def fit(self, X, y):
        # mock for sklearn...
        ...

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        if torch.is_tensor(X):
            y_pred = self.predict(X)
        else:
            X_ = torch.from_numpy(X)
            y_pred = self.predict(X_)
        y, y_pred = y.data.cpu().numpy(), y_pred.data.cpu().numpy()
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    def get_layer_string(self):
        return '-'.join(str(x) for x in self.layer_config) + '-' + (str(self.n_classes) if self.n_classes > 2 else '1')
