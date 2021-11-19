import torch
import torch.nn as nn
import utils


class BaseModel(nn.Module):
    def __init__(self, config, device=None):
        super(BaseModel, self).__init__()
        self.config = config
        self.name = 'model'
        self.device = device
        self.trainer = None
        self.layer_config = self.get_attribute('layer_config')
        self.transform = self.get_attribute('transform')

    def get_attribute(self, name, default=None, required=True):
        return utils.get_attribute(self.config, name, default, required)

    def set_attribute(self, name, value):
        self.config[name] = value

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
