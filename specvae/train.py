from datetime import datetime
import torch, os
import numpy as np
# import torch.profiler
import utils
import metrics as mcs


class MetricCounter:
    def __init__(self, value: dict):
        self.value = value

    def select_keys(self, keys):
        return dict(zip(keys, [self.value[key] for key in keys]))

    def copy(self):
        return MetricCounter(self.value.copy())

    def __add__(self, other):
        if isinstance(other, MetricCounter):
            value = self.value.copy()
            for other_key, other_value in other.value.items():
                if other_key in value:
                    if isinstance(other_value, dict) and isinstance(self.value[other_key], dict):
                        val = MetricCounter(value[other_key]) + MetricCounter(other_value)
                        value[other_key] = val.value
                    else:
                        value[other_key] += other_value
                else:
                    value[other_key] = other_value
            return MetricCounter(value)
        
        elif isinstance(other, dict):
            return self + MetricCounter(other)
        else:
            raise ValueError('Unsupported type')

    def __mul__(self, other):
        value = self.value.copy()
        for key, val in value.items():
            if isinstance(val, dict):
                v = MetricCounter(val)
                v = v * other
                value[key] = v.value
            else:
                value[key] = val * other
        return MetricCounter(value)

    def __truediv__(self, other):
        value = self.value.copy()
        for key, val in value.items():
            if isinstance(val, dict):
                v = MetricCounter(val)
                v = v / other
                value[key] = v.value
            else:
                value[key] = val / other
        return MetricCounter(value)

    def __getitem__(self, key):
        return self.value[key]

    def __setitem__(self, key, value):
        self.value[key] = value

    def __delitem__(self, key):
        del self.value[key]

    def __contains__(self, key):
        return key in self.value
    

class Trainer:
    def __init__(self, model, writer=None):
        self.model = model
        self.writer = writer
        self.optimizer = None
        self.epochs = 0
        self.history = {}
        self.internal_metric_names = []
        self.metric_names = []
        self.metric_funcs = {}

    def compile(self, optimizer, metrics=['loss'], evaluation_metrics=[]):
        self.optimizer = optimizer
        self.internal_metric_names = metrics
        for m in self.internal_metric_names:
            self.history[m] = []
            self.history['epoch_' + m] = []
            self.history['val_' + m] = []
            self.history['epoch_val_' + m] = []
        self.metric_names = evaluation_metrics
        for m in self.metric_names:
            self.metric_funcs[m] = getattr(mcs, m)
    
    def train_step(self, epoch, step, data_batch):
        loss, y_true, y_pred, values = self.model_forward(data_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if 'loss' in self.internal_metric_names:
            values['loss'] = loss.item()
        return values

    def valid_step(self, epoch, step, data_batch):
        loss, y_true, y_pred, values = self.model_forward(data_batch)
        if 'loss' in self.internal_metric_names:
            values['loss'] = loss.item()
        return values

    def test_step(self, epoch, step, data_batch, metrics=[], metric_funcs={}, internal_metrics=[]):
        values = {}
        loss, y_true, y_pred, mvalues = self.model_forward(data_batch)
        values['loss'] = loss.item()
        # y_true = y_true.squeeze().data.cpu().numpy()
        # y_pred = y_pred.squeeze().data.cpu().numpy()
        for metric_name in internal_metrics:
            if metric_name in mvalues:
                values[metric_name] = mvalues[metric_name]
        for metric_name in metrics:
            values[metric_name] = metric_funcs[metric_name](y_true, y_pred)
        return values

    def model_forward(self, data_batch):
        return 0., None, None, None # loss, y_true, y_pred, values

    def metrics_eval_(self, train_data, validation_data):
        e_train, e_valid = None, None
        if train_data is not None:
            e_train = self.evaluate(train_data, evaluation_metrics=self.metric_names)
        if validation_data is not None:
            e_valid = self.evaluate(validation_data, evaluation_metrics=self.metric_names)
        return e_train, e_valid

    def metrics(self, 
        epoch, n_epochs, 
        train_step, n_train_steps, 
        train_avg_values, valid_avg_values, 
        train_data, validation_data, log_freq=10):

        m_list, f_list = [], []
        if 'loss' in train_avg_values:
            m_list.append(train_avg_values['loss'])
            f_list.append('loss= {:.4f}')
        if 'loss' in valid_avg_values:
            m_list.append(valid_avg_values['loss'])
            f_list.append('val/loss= {:.4f}')

        e_train, e_valid = self.metrics_eval_(train_data, validation_data)
        for name in e_train.keys():
            if name == 'loss':
                continue
            if e_train:
                self.log('train/' + name, e_train[name], epoch)
                m_list.append(e_train[name])
                f_list.append(name + '= {:.2f}')
            if e_valid:
                self.log('valid/' + name, e_valid[name], epoch)
                m_list.append(e_valid[name])
                f_list.append('val/' + name + '= {:.2f}')

        self.update_progress(epoch, n_epochs, train_step, n_train_steps, 
            ', '.join(f_list).format(*m_list))

    def log(self, name, value, step):
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def log_history(self, epoch, step, values, log_prefix='', history_prefix=''):
        for val_name, value in values.select_keys(self.internal_metric_names).items():
            if isinstance(value, dict):
                if self.writer is not None:
                    self.writer.add_scalars(log_prefix + val_name, value, step)
                for k, v in value.items():
                    if history_prefix + k in self.history:
                        self.history[history_prefix + k].append([step, v])
                    else:
                        self.history[history_prefix + k] = [[step, v]]
            else:
                if history_prefix + val_name in self.history:
                    self.history[history_prefix + val_name].append([step, value])
                else:
                    self.history[history_prefix + val_name] = [[step, value]]
                self.log(log_prefix + val_name, value, step)

    def fit(self, X, epochs=100, 
        batch_size=64, 
        initial_epoch=1, 
        steps_per_epoch=None, 
        validation_data=None, 
        validation_freq=1,
        log_freq=10,
        visualization=False,
        dirpath='.',
        profiler=None):

        self.epochs = epochs
        total_train_step = 0
        total_validation_step = 0

        for epoch in range(initial_epoch, epochs + 1):
            self.model.train()
            acc_values, avg_values = MetricCounter({}), MetricCounter({})

            # Display a progress bar
            num_steps_ = len(X)
            printProgressBar(0, num_steps_, 
                prefix=f'Epochs {epoch}/{epochs}', suffix='', length=50)

            for train_step, data_batch in enumerate(X):
                values = self.train_step(epoch, train_step, data_batch)
                if profiler is not None:
                    profiler.step()

                # Accumulate and average train metrics:
                acc_values = acc_values + values
                avg_values = acc_values / (train_step + 1)

                printProgressBar(train_step + 1, num_steps_, 
                    prefix=f'Epochs {epoch}/{epochs}', 
                    suffix='loss= {:.4f}'.format(avg_values['loss']) if 'loss' in avg_values else '', length=50, end=False)

                if log_freq > 0 and total_train_step % log_freq == 0:
                    self.log_history(epoch, total_train_step, 
                        MetricCounter(values), log_prefix='train/', history_prefix='')

                total_train_step += 1

            train_avg_values = avg_values.copy()
            self.log_history(epoch, epoch, avg_values, 
                log_prefix='train/epoch_', history_prefix='epoch_')

            # Validate:
            if validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    vis_data_batch = None
                    acc_values, avg_values = MetricCounter({}), MetricCounter({})

                    for valid_step, data_batch in enumerate(validation_data):
                        if valid_step == 0:
                            vis_data_batch = data_batch
                        values = self.valid_step(epoch, valid_step, data_batch)

                        # Accumulate and average train metrics:
                        acc_values = acc_values + values
                        avg_values = acc_values / (valid_step + 1)

                        if log_freq > 0 and total_validation_step % log_freq == 0:
                            self.log_history(epoch, total_validation_step, 
                                MetricCounter(values), log_prefix='valid/', history_prefix='val_')
                    
                        total_validation_step += 1

                    valid_avg_values = avg_values.copy()
                    self.log_history(epoch, epoch, avg_values, 
                        log_prefix='valid/epoch_', history_prefix='epoch_val_')

                    # Log train and validation metrics together:
                    if self.writer is not None:
                        for val_name, avg_value in avg_values.select_keys(self.internal_metric_names).items():
                            if not isinstance(avg_value, dict) and not isinstance(train_avg_values[val_name], dict):
                                self.writer.add_scalars(val_name, {'train': train_avg_values[val_name], 'valid': avg_value}, epoch)

                    # Visualize the output and input at each epoch:
                    if visualization:
                        visualization(self.model, vis_data_batch, dirpath, epoch)

            # Compute metrics and insert to the log:
            with torch.no_grad():
                if validation_freq > 0 and (epoch % validation_freq == 0 or epoch == 1):
                    self.metrics(epoch, epochs, train_step, num_steps_, 
                        train_avg_values, valid_avg_values, X, validation_data)
            # else:
            #     print() # new line for progress bar
        return self.history

    def evaluate(self, X, metrics=['loss'], evaluation_metrics=[]):
        # Compile list of metric functions:
        metric_funcs = self.metric_funcs
        if self.metric_names != evaluation_metrics:
            metric_funcs = {}
            for m in evaluation_metrics:
                metric_funcs[m] = getattr(mcs, m)
        # Compute values:
        # metrics_acc = {}
        # metrics_acc['loss'] = 0.
        # for metric_name in metrics:
        #     metrics_acc[metric_name] = 0.
        acc_values = MetricCounter({})
        for test_step, data_batch in enumerate(X):
            values = self.test_step(-1, test_step, data_batch, evaluation_metrics, metric_funcs, metrics)
            acc_values = acc_values + values
            # for metric_name, value in values.items():
            #     metrics_acc[metric_name] += value
        avg_values = acc_values / (test_step + 1)
        # Average:
        # metrics_value = {}
        # for metric_name, metric_acc in metrics_acc.items():
        #     metrics_value[metric_name] = metric_acc / (test_step + 1)
        return avg_values.value

    def update_progress(self, 
        epoch, n_epochs, train_step, n_train_steps, text):
        printProgressBar(train_step + 1, n_train_steps, 
            prefix=f'Epochs {epoch}/{n_epochs}', 
            suffix=('%s' % text), length=50, end=False)
        print()




class ClassifierTrainer(Trainer):
    def __init__(self, model, writer=None):
        super().__init__(model, writer)

    def model_forward(self, data_batch):
        x_batch, y_batch, ids_batch = data_batch
        y_logits_batch, y_pred_batch = self.model.forward_(x_batch)
        loss = self.model.loss(y_logits_batch, y_batch)
        return loss, y_batch, y_pred_batch, {}


class RegressorTrainer(Trainer):
    def __init__(self, model, writer=None):
        super().__init__(model, writer)

    def model_forward(self, data_batch):
        x_batch, y_batch, ids_batch = data_batch
        y_pred_batch = self.model(x_batch)
        loss = self.model.loss(y_pred_batch, y_batch)
        return loss, y_batch, y_pred_batch, {}


class VAETrainer(Trainer):
    def __init__(self, model, writer=None):
        super().__init__(model, writer=writer)

    def model_forward(self, data_batch):
        x_batch, ids_batch = data_batch
        x_recon_batch, latent_sample, latent_dist = self.model.forward_(x_batch)
        loss, recon, kldiv = self.model.loss.forward_(x_recon_batch, x_batch, latent_dist)
        return loss, x_batch, x_recon_batch, {'kldiv': kldiv.item(), 'recon': recon.item()}


class ComposedTrainer(Trainer):
    def __init__(self, model, writer=None):
        super().__init__(model, writer=writer)
        self.metric_funcs_sub = {}

    def compile(self, optimizer, metrics=['loss'], evaluation_metrics=[], evaluation_metrics_sub=[]):
        super().compile(optimizer, metrics, evaluation_metrics)
        self.metric_names_sub = evaluation_metrics_sub
        for m in self.metric_names_sub:
            self.metric_funcs_sub[m] = getattr(mcs, m)

    def metrics_eval_(self, train_data, validation_data):
        e_train, e_valid = None, None
        if train_data is not None:
            e_train = self.evaluate(train_data, self.metric_names, self.metric_names_sub)
        if validation_data is not None:
            e_valid = self.evaluate(validation_data, self.metric_names, self.metric_names_sub)
        return e_train, e_valid

    def test_step_sub(self, epoch, step, data_batch, metrics=[], metric_funcs={}):
        values = {}
        x_batch, y_true, ids_batch = data_batch
        x_recon_batch, latent_sample, latent_dist, (y_logits_batch, y_pred) = self.model.forward_(x_batch)
        # y_true = y_batch.squeeze().data.cpu().numpy()
        # y_pred = y_pred_batch.squeeze().data.cpu().numpy()
        for metric_name in metrics:
            values[metric_name] = metric_funcs[metric_name](y_true, y_pred)
        return values

    def evaluate(self, X, metrics=[], metrics_sub=[]):
        metrics_value = super().evaluate(X, evaluation_metrics=metrics)
        # Compile list of metric functions:
        metric_funcs = self.metric_funcs_sub
        if self.metric_names_sub != metrics_sub:
            metric_funcs = {}
            for m in metrics_sub:
                metric_funcs[m] = getattr(mcs, m)
        # Compute values:
        metrics_acc = {}
        for metric_name in metrics_sub:
            metrics_acc[metric_name] = 0.
        for test_step, data_batch in enumerate(X):
            values = self.test_step_sub(-1, test_step, data_batch, metrics_sub, metric_funcs)
            for metric_name, value in values.items():
                metrics_acc[metric_name] += value
        # Average:
        for metric_name, metric_acc in metrics_acc.items():
            metrics_value[metric_name] = metric_acc / (test_step + 1)
        return metrics_value


class JointVAEandClassifierTrainer(ComposedTrainer):
    def __init__(self, model, writer=None):
        super().__init__(model, writer=writer)

    def model_forward(self, data_batch):
        x_batch, y_batch, ids_batch = data_batch
        x_recon_batch, latent_sample, latent_dist, (y_logits_batch, y_pred_batch) = self.model.forward_(x_batch)
        loss, recon, kldiv, clf_loss = self.model.loss.forward_(
            x_batch, x_recon_batch, latent_dist, y_logits_batch, y_batch)
        return loss, x_batch, x_recon_batch, {'kldiv': kldiv.item(), 'recon': recon.item(), 'clf_loss': clf_loss.item()}

    def test_step_sub(self, epoch, step, data_batch, metrics=[], metric_funcs={}):
        values = {}
        x_batch, y_true, ids_batch = data_batch
        x_recon_batch, latent_sample, latent_dist, y_pred = self.model.forward_(x_batch)
        # y_true = y_batch.squeeze().data.cpu().numpy()
        # y_pred = y_pred_batch.squeeze().data.cpu().numpy()
        for metric_name in metrics:
            values[metric_name] = metric_funcs[metric_name](y_true, y_pred)
        return values


class JointVAEandRegressorTrainer(ComposedTrainer):
    def __init__(self, model, writer=None):
        super().__init__(model, writer=writer)

    def model_forward(self, data_batch):
        x_batch, y_batch, ids_batch = data_batch
        x_recon_batch, latent_sample, latent_dist, y_pred_batch = self.model.forward_(x_batch)
        loss, recon, kldiv, reg_loss = self.model.loss.forward_(
            x_batch, x_recon_batch, latent_dist, y_pred_batch, y_batch)
        return loss, x_batch, x_recon_batch, {'kldiv': kldiv.item(), 'recon': recon.item(), 'reg_loss': reg_loss.item()}

    def test_step_sub(self, epoch, step, data_batch, metrics=[], metric_funcs={}):
        values = {}
        x_batch, y_true, ids_batch = data_batch
        x_recon_batch, latent_sample, latent_dist, y_pred = self.model.forward_(x_batch)
        # y_true = y_batch.squeeze().data.cpu().numpy()
        # y_pred = y_pred_batch.squeeze().data.cpu().numpy()
        for metric_name in metrics:
            values[metric_name] = metric_funcs[metric_name](y_true, y_pred)
        return values


class JointVAETrainer(Trainer):
    def __init__(self, model, writer=None):
        super().__init__(model, writer=writer)

    def train_step(self, epoch, step, data_batch):
        loss, y_true, y_pred, values = self.model_forward(data_batch)
        self.model.loss.step()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if 'loss' in self.internal_metric_names:
            values['loss'] = loss.item()
        return values

    def model_forward(self, data_batch):
        x_batch, ids_batch = data_batch
        x_recon_batch, latent_sample, latent_dist = self.model.forward_(x_batch)
        loss, recon, kldiv, kldiv_cont, kldiv_disc, cont_capacity_loss, disc_capacity_loss = \
            self.model.loss.forward__(x_recon_batch, x_batch, latent_dist)
        return loss, x_batch, x_recon_batch, \
            {
                'kldiv': kldiv.item(), 
                'recon': recon.item(),
                'kldiv_cont': dict(zip(
                    ['kldiv_cont_%d' % dim for dim in range(kldiv_cont.shape[0])], 
                    kldiv_cont.data.cpu().numpy().tolist())), 
                'kldiv_disc': dict(zip(
                    ['kldiv_disc_%d' % dim for dim in range(len(kldiv_disc))], 
                    [kl.item() for kl in kldiv_disc])),
                'cont_capacity_loss': cont_capacity_loss.item(),
                'disc_capacity_loss': 0 if disc_capacity_loss == 0 else disc_capacity_loss.item(),
            }

    



def get_training_path(base, model):
    layer_string = model.get_layer_string()
    model_name = model.get_name()
    now = datetime.now()
    dt = now.strftime("%d-%m-%Y_%H-%M-%S")
    full_model_name = '%s_%s (%s)' % (model_name, layer_string, dt)
    return os.path.join(base, full_model_name), full_model_name

def prepare_training_session(model, subdirectory=None, session_name=None):
    # Create paths and files:
    paths = {}
    paths['model_dir'] = os.path.join(utils.get_project_path(), '.model')
    if subdirectory:
        paths['model_dir'] = os.path.join(paths['model_dir'], subdirectory)
    if session_name:
        paths['model_dir'] = os.path.join(paths['model_dir'], session_name)

    if not os.path.exists(paths['model_dir']):
        os.makedirs(paths['model_dir'])
    training_path, full_model_name = get_training_path(paths['model_dir'], model)
    paths['full_model_name'] = full_model_name
    paths['training_path'] = training_path
    paths['img_path'] = os.path.join(paths['training_path'], 'img')
    # os.makedirs(paths['training_path'])
    os.makedirs(paths['img_path'])

    paths['model_filename'] = 'model.pth'
    paths['plot_train_loss_filename'] = 'train_loss.png'
    paths['plot_train_recon_loss_filename'] = 'train_recon_loss.png'
    paths['plot_train_kl_loss_filename'] = 'train_kl_loss.png'
    paths['plot_train_klcont_loss_filename'] = 'train_klcont_loss.png'
    paths['plot_train_klcont_dim_loss_filename'] = 'train_klcont_dim_loss.png'
    paths['plot_train_kldisc_loss_filename'] = 'train_kldisc_loss.png'
    paths['plot_train_kldisc_dim_loss_filename'] = 'train_kldisc_dim_loss.png'
    paths['plot_val_loss_filename'] = 'validation_loss.png'
    paths['plot_train_val_loss_filename'] = 'train_validation_loss.png'
    paths['plot_cossim_filename'] = 'cossim.png'
    paths['plot_modcossim_filename'] = 'modcossim.png'
    paths['history_filename'] = 'history.json'
    paths['model_path'] = os.path.join(paths['training_path'], paths['model_filename'])
    paths['plot_train_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_loss_filename'])
    paths['plot_train_recon_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_recon_loss_filename'])
    paths['plot_train_kl_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_kl_loss_filename'])
    paths['plot_train_klcont_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_klcont_loss_filename'])
    paths['plot_train_klcont_dim_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_klcont_dim_loss_filename'])
    paths['plot_train_kldisc_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_kldisc_loss_filename'])
    paths['plot_train_kldisc_dim_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_kldisc_dim_loss_filename'])
    paths['plot_val_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_val_loss_filename'])
    paths['plot_train_val_loss_filepath'] = os.path.join(paths['training_path'], paths['plot_train_val_loss_filename'])
    paths['plot_cossim_filepath'] = os.path.join(paths['training_path'], paths['plot_cossim_filename'])
    paths['plot_modcossim_filepath'] = os.path.join(paths['training_path'], paths['plot_modcossim_filename'])
    paths['history_filepath'] = os.path.join(paths['training_path'], paths['history_filename'])
    return paths


def export_training_session(trainer, paths, 
    train_loader=None, valid_loader=None, test_loader=None, 
        n_mol=100, metrics=[], evaluation_metrics=[]):

    import visualize as vis, json
    model = trainer.model
    # Save model:
    if hasattr(model, 'save'):
        model.save(paths['model_path'])
    # Evaluation mode:
    if hasattr(model, 'eval'):
        model.eval()
    
    # Collect numeric and remaining parameters:
    hparams = {}
    rhparams = {}
    for key, item in model.config.items():
        if type(item) is str:
            rhparams[key] = item
        if type(item) in (int, float, bool):
            hparams[key] = item

    # Collect remaining parameters:
    if 'full_model_name' in paths:
        rhparams['full_model_name'] = paths['full_model_name']
    if 'layer_config' in model.config:
        rhparams['layer_config'] = model.config['layer_config'].tolist()
    if 'input_columns' in model.config:
        rhparams['input_columns'] = model.config['input_columns']

    # Compute metrics against average sample:
    def compute_metrics_for_average_sample(loader, metrics):
        vals = {}
        try:
            spectra = loader.dataset.dataset.data['spectrum'] #.data.cpu().numpy()
            mean_spectrum = torch.unsqueeze(spectra.mean(dim=0), dim=0)
            for m in metrics:
                f = getattr(mcs, m)
                vals[m] = f(spectra, mean_spectrum)
            return vals
        except:
            print("Unable to compute metrics for average sample...")
            return None
    
    # Evaluate model:
    metric = {}
    fmetric = {}
    from .vae import BaseVAE
    if test_loader is not None:
        train_e = trainer.evaluate(train_loader, metrics, evaluation_metrics)
        for key, item in train_e.items():
            metric['model/train/' + key] = item
            fmetric['train_' + key] = item
        if isinstance(model, BaseVAE):
            train_avg = compute_metrics_for_average_sample(train_loader, evaluation_metrics)
            if train_avg is not None:
                for key, item in train_avg.items():
                    fmetric['train_avg_' + key] = item
    if valid_loader is not None:
        valid_e = trainer.evaluate(valid_loader, metrics, evaluation_metrics)
        for key, item in valid_e.items():
            metric['model/valid/' + key] = item
            fmetric['valid_' + key] = item
        if isinstance(model, BaseVAE):
            valid_avg = compute_metrics_for_average_sample(valid_loader, evaluation_metrics)
            if valid_avg is not None:
                for key, item in valid_avg.items():
                    fmetric['valid_avg_' + key] = item
    if test_loader is not None:
        test_e = trainer.evaluate(test_loader, metrics, evaluation_metrics)
        for key, item in test_e.items():
            metric['model/test/' + key] = item
            fmetric['test_' + key] = item
        if isinstance(model, BaseVAE):
            test_avg = compute_metrics_for_average_sample(test_loader, evaluation_metrics)
            if test_avg is not None:
                for key, item in test_avg.items():
                    fmetric['test_avg_' + key] = item

    # Write the summary:
    with open(os.path.join(paths['training_path'], 'summary.txt'), 'w+') as summary:
        if hasattr(model, 'get_layer_string'):
            summary.write('Model\n')
            summary.write('\tLayers: %s\n' % model.get_layer_string())
            if hasattr(model, 'latent_dim'):
                summary.write('\tLatent dimension: %d\n' % model.latent_dim)
            if hasattr(model, 'resolution'):
                summary.write('\tResolution: %f\n' % model.resolution)
            summary.write('\tTrain samples: %s\n' % str(n_mol))
        if hasattr(model, 'config'):
            summary.write('Config\n')
            summary.write('\t' + str(model.config) + '\n')
        # Print model's state_dict:
        if hasattr(model, 'state_dict'):
            summary.write("\tModel's state_dict:\n")
            for param_tensor in model.state_dict():
                summary.write('\t\t' + param_tensor + ': ' + str(model.state_dict()[param_tensor].size()) + '\n')
        # Print optimizer's param_groups:
        if hasattr(trainer, 'optimizer'):
            # hparams['lr'] = trainer.optimizer.param_groups[0]['lr']
            summary.write("\tOptimizers's param_groups:\n")
            summary.write('\t\t' + str(trainer.optimizer.state_dict()['param_groups']) + '\n')
        # Model results:
        summary.write('Results\n')
        if test_loader is not None:
            summary.write('\t TRAIN:' + str(train_e) + '\n')
        if valid_loader is not None:
            summary.write('\t VALID:' + str(valid_e) + '\n')
        if test_loader is not None:
            summary.write('\t  TEST:' + str(test_e) + '\n')

    # Visualize reconstruction results:
    # Plot metrics:
    if hasattr(trainer, 'history'):
        # Save history to json file:
        with open(paths['history_filepath'], 'w') as history_file:
            json.dump(trainer.history, history_file)
        vis.plot_history(trainer.history, 'epoch_loss', paths['plot_train_loss_filepath'])
        vis.plot_history(trainer.history, 'epoch_val_loss', paths['plot_val_loss_filepath'])
        vis.plot_history_2combined(trainer.history, 'epoch_loss', 'epoch_val_loss', paths['plot_train_val_loss_filepath'])

    # Save CSV file:
    try:
        import pandas as pd
        stats_file = os.path.join(paths['model_dir'], 'experiment.csv')
        cols = list(rhparams.keys()) + ['param_' + name for name in hparams.keys()] + ['m_' + name for name in fmetric.keys()]
        vals = list(rhparams.values()) + list(hparams.values()) + list(fmetric.values())
        df2 = pd.DataFrame([vals], columns=cols)
        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file, index_col=0)
            df = pd.concat([df, df2], ignore_index=True)
            df.to_csv(stats_file)
        else:
            df2.to_csv(stats_file)
    except Exception as e:
        print("Error while adding record to experiment.csv:", e)

    if trainer.writer is not None:
        trainer.writer.add_hparams(hparams, metric)
        trainer.writer.close()


# Method for printing a pretty pregressbar when training the network
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", end=True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {iteration}/{total} {suffix}', end = printEnd)
    if iteration == total and end: 
        print()
