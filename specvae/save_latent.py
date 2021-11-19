import numpy as np
import sys, os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

import torchvision as tv
from specvae.dataset import MoNA
import specvae.vae as vae, specvae.utils as utils
import specvae.dataset as dt

from io import BytesIO
from PIL import Image
import base64, io
import specvae.visualize as vis

use_cuda = False

cpu_device = torch.device('cpu')
if torch.cuda.is_available() and use_cuda:
    device = torch.device('cuda:0')
    print('GPU device count:', torch.cuda.device_count())
else:
    device = torch.device('cpu')

print('Device in use: ', device)


def load_spectra_data(dataset, transform, n_samples=-1, device=None, cols=['spectrum']):
    test_data = None
    if dataset == 'HMDB':
        # HMDB:
        test_data_path = utils.get_project_path() / '.data' / 'HMDB' / 'hmdb_cfmid_dataset_test.csv'
        data_frame = datasets.HMDB.open(test_data_path, cols)

        print("Load test data")
        test_data = datasets.HMDB.preload_tensor(
            device=device, filepath=test_data_path, columns=cols,
            transform=transform, limit=n_samples)
        return test_data

    elif dataset == 'MoNA':
        # MoNA:
        data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA.csv'
        df_train, df_valid, df_test = datasets.MoNA.get_by_split(train_data_path, columns=cols)

        print("Load test data")
        test_data = datasets.MoNA.preload_tensor(
            device=device, data_frame=df_test,
            transform=transform, limit=n_samples)
        return test_data

    return test_data


def load_taxonomy_data(dataset, transform, n_samples=-1, device=None, cols=['spectrum'], field_name='class'):
    data, labels = None, None
    tax_fields = ('kingdom', 'superclass', 'class', 'subclass')
    field = tax_fields.index(field_name)
    def retrieve_filed(row):
        tax = row['taxonomy']
        arr = [f.split(':')[0] for f in tax.split(';')]
        row[field_name] = arr[field] if len(arr) > field else ''
        return row

    if dataset == 'HMDB':
        # HMDB:
        test_data_path = utils.get_project_path() / '.data' / 'HMDB' / 'hmdb_cfmid_dataset_test.csv'
        data_frame = datasets.HMDB.open(test_data_path, cols)
        df = data_frame
        df = MoNA.get_n_molecules(1000, df=df)
        df['HMDB'] = df['id']
        data_path = utils.get_project_path() / '.data' / 'HMDB' / 'HMDB_test_taxonomy.csv'
        tax_df = pd.read_csv(data_path)[['HMDB', 'taxonomy']]
        # data_frame = datasets.HMDB.open(data_path, cols)

        print("Get taxonomy by matching on HMDB (id) column...")
        tax_df = tax_df[tax_df['HMDB'].isin(df['id'].unique())]
        tax_df = tax_df.apply(retrieve_filed, axis=1)
        tax_df = tax_df.drop(['taxonomy'], axis=1)

        print("Derive distinct classes...")
        labels = np.unique(tax_df[field_name].to_numpy())
        labels = labels[labels != ''].tolist()
        print("Number of distinct %s: %d" % (field_name, len(labels)))

        print("Classify each row in dataset and assign label (index to class)...")
        df_merged = df.merge(tax_df, on='HMDB', suffixes=('','_y')).drop(['HMDB'], axis=1)
        def assign_label(row, labels):
            tax = row[field_name]
            try:
                row[field_name + '_id'] = labels.index(tax)
            except:
                row[field_name + '_id'] = -1
            return row
        df = df_merged.apply(lambda row: assign_label(row, labels), axis=1).drop([field_name], axis=1)

        print("Load test data")
        data = datasets.MoNA.preload_tensor(
            device=device, data_frame=df,
            transform=transform, limit=n_samples)

    elif dataset == 'MoNA':
        # MoNA:
        data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA.csv'
        # df = pd.read_csv(data_path, index_col=0)[cols + ['InChIKey', 'InChI']]
        df = MoNA.open(data_path, cols + ['InChI'])
        # df['spectrum_str'] = df['spectrum']
        spectrum_str = df[['spectrum']].to_numpy()
        df = MoNA.get_unique(2000, df=df)
        tax_data_path = utils.get_project_path() / '.data' / 'MoNA' / 'MoNA_taxonomy.csv'
        tax_df = pd.read_csv(tax_data_path)[['InChI', 'taxonomy']]
        
        print("Get taxonomy by matching on inchi column...")
        tax_df = tax_df[tax_df['InChI'].isin(df['InChI'].unique())]
        tax_df = tax_df.apply(retrieve_filed, axis=1)
        tax_df = tax_df.drop(['taxonomy'], axis=1)

        print("Derive distinct classes...")
        labels = np.unique(tax_df[field_name].to_numpy())
        labels = labels[labels != ''].tolist()
        print("Number of distinct %s: %d" % (field_name, len(labels)))

        print("Classify each row in dataset and assign label (index to class)...")
        df_merged = df.merge(tax_df, on='InChI', suffixes=('','_y')).drop(['InChI'], axis=1)
        def assign_label(row, labels):
            tax = row[field_name]
            try:
                row[field_name + '_id'] = labels.index(tax)
            except:
                row[field_name + '_id'] = -1
            return row
        
        df = df_merged.apply(lambda row: assign_label(row, labels), axis=1).drop([field_name], axis=1)

        print("Load test data")
        data = dt.MoNA.preload_tensor(
            device=device, data_frame=df,
            transform=transform, limit=n_samples)

    return data, labels, spectrum_str


def generate_image(spectrum, mode, energy, id, scale=0.5):
    try:
        fig, ax = plt.subplots()
        meta = {
            'collision energy': energy,
            'ionization mode': mode
        }
        vis.plot_spectrum(spectrum, name=id, meta=meta, ax=ax, resolution=0.5, max_mz=2500, figsize=(5, 5))

        with io.BytesIO() as io_buf:
            fig.savefig(io_buf, format='raw')
            io_buf.seek(0)
            img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        plt.close()
        buffer = BytesIO()
        image = Image.fromarray(img_arr)
        width, height = image.size
        image = image.resize((int(scale * width), int(scale * height)))
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
    except:
        print('Unable to vis spectrum')
        return None


def main(argc, argv):
    # Processing parameters:
    # method = UMAP # PCA or TSNE
    dataset = 'MoNA' # HMDB and MoNA
    tax_name = 'superclass' # 'kingdom', 'superclass', 'class', 'subclass'
    model_name = 'alt_specvae_2000-1538-30-1538-2000 (28-06-2021_14-05-29)'
    max_num_peaks = 1000
    min_intensity = 0.1
    spec_max_mz = 2500
    batch_size = 25000
    generate_imgs = True

    # Data processing:
    n_samples = batch_size # -1 if all
    savefig = True
    cols = ['spectrum', 'ionization mode', 'collision_energy_new']

    transform = tv.transforms.Compose([
        dt.SplitSpectrum(),
        dt.FilterPeaks(max_mz=spec_max_mz, min_intensity=min_intensity),
        dt.Normalize(intensity=True, mass=True, max_mz=spec_max_mz),
        dt.ToMZIntConcatAlt(max_num_peaks=max_num_peaks),
        dt.Ion2Int(one_hot=True)
    ])
    revtv = tv.transforms.Compose([])

    # Load and transform dataset:
    data, labels, spectra_str = load_taxonomy_data(dataset, transform, n_samples, device, cols, tax_name)
    if data is None:
        print("No dataset specified, script terminates.")

    # Set data loaders:
    test_loader = DataLoader(
        dt.Spectra(data=data, device=device, columns=cols + [tax_name + '_id']),
        batch_size=batch_size, shuffle=False)

    spectra_str = np.array(spectra_str).flatten()[:batch_size]

    print("Load model: %s..." % model_name)
    model_path = utils.get_project_path() / '.model' / dataset / model_name / 'model.pth'
    model = vae.BaseVAE.load(model_path, device)
    model.eval()

    print("Encode N=%d spectra from %s dataset..." % (n_samples, dataset))
    spectrum_batch, mode_batch, energy_batch, class_batch, id_batch = next(iter(test_loader))
    mu, logvar = model.encode(spectrum_batch)

    latent_batch = mu
    X = latent_batch.data.cpu().numpy()
    Xenergy = energy_batch.data.cpu().numpy().reshape(-1)
    y = np.argmax(mode_batch.data.cpu().numpy(), axis=1)
    ids=np.array(id_batch)
    class_batch=class_batch.data.cpu().numpy().reshape(-1)

    print("Export image data")
    filename = "%s-%s.npz" % (dataset, model_name)
    filepath = utils.get_project_path() / '.data' / 'latent' / filename
    np.savez(filepath, X=X, mode=y, energy=Xenergy, ids=ids, tax=class_batch, classes=np.array(labels))

    if generate_imgs:
        ssize = spectra_str.shape[0]
        last = 0
        images = []
        for i, spectrum in enumerate(spectra_str):
            try:
                images.append(generate_image(spectrum, y[i], Xenergy[i], ids[i], scale=0.4))
                if (i / ssize - last) > 0.05:
                    last = i / ssize
                    print("Progress {}%".format(int(last * 100)))
            except IndexError as ie:
                print("Error: {0}".format(ie))
            except:
                print("Unknown error has occurred")

        print("Save file ", filepath)
        np.savez(filepath, X=X, mode=y, energy=Xenergy, ids=ids, tax=class_batch, classes=np.array(labels), spectra=spectra_str, imgs=images)

    return 0


if __name__ == '__main__':
    sys.exit(main(len(sys.argv), sys.argv))
