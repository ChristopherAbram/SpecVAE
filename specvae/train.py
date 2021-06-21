from datetime import datetime
import torch, os
import utils


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.history = { 
            'loss':             [], 
            'epoch_loss':       [], 
            'val_loss':         [], 
            'epoch_val_loss':   [],
        }

    def compile(self, optimizer):
        self.optimizer = optimizer

    def train_step(self, epoch, step, data_batch):
        return 0.

    def valid_step(self, epoch, step, data_batch):
        return 0.

    def train_metrics(self,
        epoch, n_epochs, 
        train_step, n_train_steps, 
        train_loss, train_data, log_freq=10):
        pass

    def metrics(self, 
        epoch, n_epochs, 
        train_step, n_train_steps, 
        train_loss, val_loss, validation_data, log_freq=10):

        self.update_progress(epoch, n_epochs, train_step, n_train_steps, 
            'loss= {:.4f}, val_loss= {:.4f}'.format(
                train_loss, val_loss))


    def fit(self, X, epochs=100, 
        batch_size=64, 
        initial_epoch=1, 
        steps_per_epoch=None, 
        validation_data=None, 
        train_freq=1,
        validation_freq=1,
        log_freq=10,
        visualization=False,
        dirpath='.'):

        total_train_step = 0
        total_validation_step = 0
        for epoch in range(initial_epoch, epochs + 1):
            self.model.train()
            total_training_loss = 0.
            avg_training_loss = 0.
            total_validation_loss = 0.
            avg_validation_loss = 0.

            # Display a progress bar
            num_steps_ = len(X)
            printProgressBar(0, num_steps_, 
                prefix=f'Epochs {epoch}/{epochs}', suffix='', length=50)

            for train_step, data_batch in enumerate(X):
                loss = self.train_step(epoch, train_step, data_batch)

                total_training_loss += loss.item()
                avg_training_loss = total_training_loss / (train_step + 1)
 
                printProgressBar(train_step + 1, num_steps_, 
                    prefix=f'Epochs {epoch}/{epochs}', 
                    suffix='loss= {:.4f}'.format(avg_training_loss), length=50, end=False)

                if total_train_step % log_freq == 0:
                    # TODO: write to file...
                    self.history['loss'].append([total_train_step, avg_training_loss])

                total_train_step += 1

            self.history['epoch_loss'].append([epoch, avg_training_loss])
            # Compute metrics and insert to the log:
            if train_freq > 0 and (epoch % train_freq == 0 or epoch == 1):
                self.train_metrics(epoch, epochs, train_step, num_steps_, avg_training_loss, X, log_freq)

            # Validate:
            if validation_data is not None:
                self.model.eval()
                with torch.no_grad():
                    vis_data_batch = None
                    for valid_step, data_batch in enumerate(validation_data):
                        if valid_step == 0:
                            vis_data_batch = data_batch
                        loss = self.valid_step(epoch, valid_step, data_batch)

                        total_validation_loss += loss.item()
                        avg_validation_loss = total_validation_loss / (valid_step + 1)

                        if total_validation_step % log_freq == 0:
                            self.history['val_loss'].append([total_validation_step, avg_validation_loss])
                    
                        total_validation_step += 1

                    self.history['epoch_val_loss'].append([epoch, avg_validation_loss])

                    # Compute metrics and insert to the log:
                    if validation_freq > 0 and (epoch % validation_freq == 0 or epoch == 1):
                        self.metrics(epoch, epochs, train_step, num_steps_, 
                            avg_training_loss, avg_validation_loss, validation_data)

                    # Visualize the output and input at each epoch:
                    if visualization:
                        visualization(self.model, vis_data_batch, dirpath, epoch)
            else:
                print() # new line for progress bar

        return self.history

    def evaluate(self, X):
        return dict()

    def update_progress(self, 
        epoch, n_epochs, train_step, n_train_steps, text):
        printProgressBar(train_step + 1, n_train_steps, 
            prefix=f'Epochs {epoch}/{n_epochs}', 
            suffix=('%s' % text), length=50, end=False)
        print()


class SpecVAETrainer(Trainer):
    def __init__(self, model):
        super().__init__(model)
        self.modcossim = utils.ModifiedCosine(model.resolution, model.max_mz)
        self.history['cossim'] = []
        self.history['modcossim'] = []

    def train_step(self, epoch, step, data_batch):
        x_batch, y_batch = data_batch
        x_pred_batch, mu, log_var = self.model.forward_(x_batch)
        loss = self.model.loss(x_batch, x_pred_batch, mu, log_var)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def valid_step(self, epoch, step, data_batch):
        x_batch, y_batch = data_batch
        x_pred_batch, mu, log_var = self.model.forward_(x_batch)
        return self.model.loss(x_batch, x_pred_batch, mu, log_var)

    def evaluate(self, X):
        total_loss, total_cossim, total_modcossim = 0., 0., 0.
        for test_step, (x_batch, y_batch) in enumerate(X):
            x_pred_batch, mu, log_var = self.model.forward_(x_batch)
            total_loss += self.model.loss(x_batch, x_pred_batch, mu, log_var)
            x_, xp_ = x_batch.data.cpu().numpy(), x_pred_batch.data.cpu().numpy()
            total_cossim += utils.cosine_similarity(x_, xp_)
            total_modcossim += self.modcossim(x_, xp_)
            
        avg_loss = total_loss / (test_step + 1)
        avg_cossim = total_cossim / (test_step + 1)
        avg_modcossim = total_modcossim / (test_step + 1)
        # TODO: add KL divergence and reconstruction...
        return {
            'loss': avg_loss,
            'cossim': avg_cossim,
            'modcossim': avg_modcossim,
        }

    def train_metrics(self,
        epoch, n_epochs, 
        train_step, n_train_steps, 
        train_loss, train_data, log_freq=10):
        pass

    def metrics(self, epoch, n_epochs, train_step, n_train_steps, 
            train_loss, val_loss, validation_data, log_freq=10):
        e = self.evaluate(validation_data)
        self.history['cossim'].append([epoch, e['cossim']])
        self.history['modcossim'].append([epoch, e['modcossim']])

        self.update_progress(epoch, n_epochs, train_step, n_train_steps, 
            'loss= {:.4f}, val_loss= {:.4f}, cos.sim= {:.3f}, mod.cos.sim= {:.3f}'.format(
                train_loss, val_loss, e['cossim'], e['modcossim']))


def get_training_path(base, model):
    layer_string = model.get_layer_string()
    model_name = model.get_name()
    now = datetime.now()
    dt = now.strftime("%d-%m-%Y_%H-%M-%S")
    return os.path.join(base, '%s_%s (%s)' % (model_name, layer_string, dt))

def prepare_training_session(trainer, subdirectory=None):
    # Create paths and files:
    model = trainer.model
    paths = {}
    paths['model_dir'] = os.path.join(utils.get_project_path(), '.model')
    if subdirectory:
        paths['model_dir'] = os.path.join(paths['model_dir'], subdirectory)
    if not os.path.exists(paths['model_dir']):
        os.makedirs(paths['model_dir'])
    paths['training_path'] = get_training_path(paths['model_dir'], model)
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


def export_training_session(trainer, paths, test_loader=None, n_mol=100):
    import visualize as vis, json
    model = trainer.model
    spec_resolution = model.resolution
    # Save model:
    if hasattr(model, 'save'):
        model.save(paths['model_path'])

    # Write the summary:
    with open(os.path.join(paths['training_path'], 'summary.txt'), 'w+') as summary:
        if hasattr(model, 'get_layer_string'):
            summary.write('Model\n')
            summary.write('\tLayers: %s\n' % model.get_layer_string())
            if hasattr(model, 'latent_dim'):
                summary.write('\tLatent dimension: %d\n' % model.latent_dim)
            summary.write('\tResolution: %f\n' % spec_resolution)
            summary.write('\tTrain samples: %s\n' % str(n_mol))
        # Print model's state_dict:
        if hasattr(model, 'state_dict'):
            summary.write("\tModel's state_dict:\n")
            for param_tensor in model.state_dict():
                summary.write('\t\t' + param_tensor + ': ' + str(model.state_dict()[param_tensor].size()) + '\n')
        # Print optimizer's param_groups:
        if hasattr(trainer, 'optimizer'):
            summary.write("\tOptimizers's param_groups:\n")
            summary.write('\t\t' + str(trainer.optimizer.state_dict()['param_groups']) + '\n')
        # Model results:
        if test_loader is not None:
            train_loss = 0.
            if hasattr(trainer, 'history'):
                train_loss = trainer.history['epoch_loss']
                train_loss = train_loss[len(train_loss) - 1][1]
            e = trainer.evaluate(test_loader)
            summary.write('Results\n')
            summary.write('\t TRAIN LOSS: %f\n' % train_loss)
            summary.write('\t  TEST LOSS: %f\n' % e['loss'])
            if 'cossim' in e:
                summary.write('\t    COS SIM: %f\n' % e['cossim'])
            if 'modcossim' in e:
                summary.write('\tMOD COS SIM: %f\n' % e['modcossim'])

    # Visualize reconstruction results:
    if hasattr(model, 'eval'):
        model.eval()

    # Plot metrics:
    if hasattr(trainer, 'history'):
        vis.plot_history(trainer.history, 'epoch_loss', paths['plot_train_loss_filepath'])
        vis.plot_history(trainer.history, 'epoch_val_loss', paths['plot_val_loss_filepath'])
        vis.plot_history_2combined(trainer.history, 'epoch_loss', 'epoch_val_loss', paths['plot_train_val_loss_filepath'])
        if 'recon_loss' in trainer.history:
            vis.plot_history(trainer.history, 'recon_loss', paths['plot_train_recon_loss_filepath'])
        if 'kl_loss' in trainer.history:
            vis.plot_history(trainer.history, 'kl_loss', paths['plot_train_kl_loss_filepath'])
        if 'cossim' in trainer.history:
            vis.plot_history(trainer.history, 'cossim', paths['plot_cossim_filepath'])
        if 'modcossim' in trainer.history:
            vis.plot_history(trainer.history, 'modcossim', paths['plot_modcossim_filepath'])

        # Save history to json file:
        with open(paths['history_filepath'], 'w') as history_file:
            json.dump(trainer.history, history_file)


# Method for printing a pretty pregressbar when training the network
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", end=True):
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