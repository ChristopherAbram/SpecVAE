import os
import numpy as np
# import config
import matplotlib
# if config.use_agg:
    # matplotlib.use('Agg')
import matplotlib.pyplot as plt

import specvae


def plot_spectra_grid(model, data_batch, dirpath='.', epoch=0, device=None, transform=None, dpi=100, name=None, grid=(3, 3), figsize=(20, 20)):
    name = "%s_spec_%s" % (model.get_name(), epoch) if name is None else name
    x_batch = data_batch[0]
    data_batch2 = model(x_batch)
    resolution, max_mz = 0.05, 2500.
    if hasattr(model, 'resolution') and hasattr(model, 'max_mz'):
        resolution, max_mz = model.resolution, model.max_mz

    if transform:
        db, db2 = [], []
        for i, x in enumerate(x_batch):
            db.append(transform(x_batch[i].data.cpu().numpy()))
            db2.append(transform(data_batch2[i].data.cpu().numpy()))
        data_batch = np.vstack(db)
        data_batch2 = np.vstack(db2)

    if hasattr(model, 'lattice') and epoch % 5 == 0:
        plot_spectra_(model.lattice(grid=(17, 17), zrange=(-75, 75), device=device), grid=(17, 17), figsize=(34, 18),
            filepath=os.path.join(dirpath, 'lattice_%s.png' % epoch), 
            resolution=resolution, max_mz=max_mz, dpi=dpi)

    return plot_spectra_compare(data_batch, data_batch2, grid=grid, figsize=figsize, 
        filepath=os.path.join(dirpath, name), resolution=resolution, max_mz=max_mz)


def plot_spectra_(spectra, grid=(4, 3), figsize=(17, 9), filepath=None, 
    resolution=0.05, max_mz=2500, transform=None, color='blue', linewidth=0.5, dpi=100):
    fig, axs = plt.subplots(grid[0], grid[1], figsize=figsize, dpi=dpi)
    for i, ax in enumerate(axs.flat):
        if i >= spectra.shape[0]:
            break
        if transform:
            import torch
            # db = []
            # for i, x in enumerate(spectra):
                # db.append(transform(spectra[i].data.cpu().numpy()))
            # data_batch = np.vstack(db)
            if torch.is_tensor(spectra):
                spectrum = transform(spectra[i].data.cpu().numpy())
            else:
                spectrum = transform(spectra[i])
        else:
            spectrum = spectra[i]

        mz = np.arange(0, max_mz, step=resolution)
        ax.plot(mz, spectrum.tolist(), color=color, linewidth=linewidth)
        ax.set_ylim([0, 100])
        if i % grid[1] == 0:
            ax.set_ylabel('Intensity [%]')
        if grid[1] * (grid[0] - 1) <= i:
            ax.set_xlabel('m/z')

        if i % grid[1] != 0:
            ax.set_yticks([])
        if grid[1] * (grid[0] - 1) > i:
            ax.set_xticks([])
    
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    # plt.close(fig)
    return fig, axs


def plot_spectra_compare(spectra1, spectra2, grid=(4, 3), figsize=(17, 9), 
    filepath=None, resolution=0.05, max_mz=2500, dpi=100):
    fig, axs = plt.subplots(grid[0], grid[1], figsize=figsize, dpi=dpi)
    for i, ax in enumerate(axs.flat):
        if i >= spectra1.shape[0] or i >= spectra2.shape[0]:
            break
        mz = np.arange(0, max_mz, step=resolution)
        ax.plot(mz, (-spectra2[i]).tolist(), color='blue')
        ax.plot(mz, spectra1[i].tolist(), color='red')
        # ax.set_title(spectra1['id'][i])
        ax.set_ylim([-100, 100])
        if i % grid[1] == 0:
            ax.set_ylabel('Intensity [%]')
        if grid[1] * (grid[0] - 1) <= i:
            ax.set_xlabel('m/z')

        if i % grid[1] != 0:
            ax.set_yticks([])
        if grid[1] * (grid[0] - 1) > i:
            ax.set_xticks([])
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    # plt.close(fig)
    return fig, axs


def plot_spectrum(spectrum, name='', filepath=None, resolution=0.05, max_mz=2500, meta=None, 
    ax=None, figsize=(5, 5), config=None, color='blue', transformed=False, linewidth=0.5):
    from . import dataset as dt
    import torchvision as tv
    if 'transform' in config:
        trans = config['transform']
        revtrans = tv.transforms.Compose([
            dt.ToMZIntDeConcatAlt(max_num_peaks=config['max_num_peaks']),
            dt.Denormalize(intensity=config['normalize_intensity'], mass=config['normalize_mass'], max_mz=config['max_mz']),
            dt.ToDenseSpectrum(resolution=resolution, max_mz=config['max_mz'])
        ])
        if not transformed:
            spectrum = trans(spectrum)
        spectrum = revtrans(spectrum)
    elif isinstance(spectrum, str):
        spectrum = dt.ToDenseSpectrum(resolution, max_mz)(dt.SplitSpectrum()(spectrum))
    return plot_spectrum_(spectrum, name, resolution, max_mz, meta, ax, figsize, color, linewidth)

def plot_spectrum_(spectrum, name='', resolution=0.05, max_mz=2500, meta=None, ax=None, figsize=(5, 5), color='blue', linewidth=0.5):
    fig = None
    title = name
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if meta:
        if 'idvis' in meta:
            ax.text(max_mz - 200., 85., meta['idvis'], fontsize=30)
            # title = ', '.join([title, str(meta['idvis'])])
        if 'collision energy' in meta:
            title = ', '.join([title, 'E=%s' % str(meta['collision energy'])])
        if 'ionization mode' in meta:
            title = ', '.join([title, '(%s)' % ('+' if (meta['ionization mode'] == 'positive' or meta['ionization mode'] == 1) else '-')])
    mz = np.arange(0, max_mz, step=resolution)
    ax.set_title(title)
    ax.plot(mz, spectrum.tolist(), color=color, linewidth=linewidth)
    ax.set_ylim([0, 100])
    ax.set_xlabel('m/z')
    ax.set_ylabel('Intensity [%]')
    return fig, ax

def plot_history(history, metric_name, filepath=None):
    values = np.array(history[metric_name])
    if len(values) > 0:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(values[:,0], values[:,1])
        plt.xlabel('epochs')
        plt.ylabel(metric_name)
        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()
        plt.close(fig)


def plot_history_2combined(history, metric_name_1, metric_name_2, filepath=None):
    values1 = np.array(history[metric_name_1])
    values2 = np.array(history[metric_name_2])
    if len(values1) > 0 and len(values2):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(values1[:,0], values1[:,1], values2[:,0], values2[:,1])
        plt.xlabel('epochs')
        plt.ylabel(metric_name_1)
        plt.legend([metric_name_1, metric_name_2])
        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()
        plt.close(fig)


def plot_history_n(history, name, names, filepath=None):
    values = [np.array(history[n]) for n in names]
    fig, ax = plt.subplots(figsize=(15, 10))
    for val in values:
        if len(val) > 0:
            ax.plot(val[:,0], val[:,1])
    plt.xlabel('iterations')
    plt.ylabel(name)
    plt.legend(names)
    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()
    plt.close(fig)


def plot_precentile(arr_ref, arr_sim, num_bins=100, 
    show_top_percentile=0.1, ignore_diagonal=True):
    
    def _ignore_reference_nans(arr_ref, arr_sim):
        assert arr_ref.shape == arr_sim.shape, "Expected two arrays of identical shape."
        idx_not_nans = np.where(np.isnan(arr_ref) == False)
        arr_sim = arr_sim[idx_not_nans]
        arr_ref = arr_ref[idx_not_nans]
        return arr_ref, arr_sim

    if ignore_diagonal:
        np.fill_diagonal(arr_ref, np.nan)

    arr_ref, arr_sim = _ignore_reference_nans(arr_ref, arr_sim)
    start = int(arr_sim.shape[0] * show_top_percentile / 100)
    idx = np.argpartition(arr_sim, -start)
    starting_point = arr_sim[idx[-start]]
    if starting_point == 0:
        print("not enough datapoints != 0 above given top-precentile")

    # Remove all data below show_top_percentile
    low_as = np.where(arr_sim < starting_point)[0]

    length_selected = arr_sim.shape[0] - low_as.shape[0]  # start+1

    data = np.zeros((2, length_selected))
    data[0, :] = np.delete(arr_sim, low_as)
    data[1, :] = np.delete(arr_ref, low_as)
    data = data[:, np.lexsort((data[1, :], data[0, :]))]

    ref_score_cum = []

    for i in range(num_bins):
        low = int(i * length_selected / num_bins)
        # high = int((i+1) * length_selected/num_bins)
        ref_score_cum.append(np.mean(data[1, low:]))
    ref_score_cum = np.array(ref_score_cum)
    x_percentiles = (show_top_percentile / num_bins * (1 + np.arange(num_bins)))[::-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(
        x_percentiles,
        ref_score_cum,
        color='black')
    return ref_score_cum


def plot_distribution(data, subject, xlabel, ylabel, plot_density=False, bins=10):
    from sklearn.neighbors import KernelDensity

    counts, bins = np.histogram(data, bins=bins, density=True)
    cc = counts / counts.sum()
    per = cc * 100.
    
    fig=plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(bins[:-1], bins, weights=per, width=0.08, color='grey', edgecolor='k', linewidth=2, alpha=0.7)

    if plot_density:
        # KDE
        xx = np.linspace(0., 1., 100)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(data[:,np.newaxis])
        log_dens = kde.score_samples(xx)
        ax.plot(xx, np.exp(log_dens) / counts.sum() * 100., linestyle='-', color='red', linewidth=2)

    # Ticks
    # xmajor_ticks = np.arange(0., 1.1, 0.1)
    # xminor_ticks = np.arange(0., 1.05, 0.05)
    # # ymajor_ticks = np.arange(0., 0.35, 0.05)
    # # yminor_ticks = np.arange(0., 0.35, 0.05)

    # ax.set_xticks(xmajor_ticks)
    # ax.set_xticks(xminor_ticks, minor=True)
    # ax.set_yticks(ymajor_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    # ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.title("Distribution of %s scores" % subject)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xlabel("structural similarity (Tanimoto)")
    # plt.ylabel("fraction of scores [%]")
    plt.show(fig)
    return cc


import pandas as pd
import numpy as np
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from itertools import cycle

def multi_index_heatmap(df, feature_column_name, row_index_columns, sample_columns, 
    feature_names_width=1.2, scale=1.0, unit=1., space=0.02, legend_offset=0.5, 
    heatmap_cell_width=1.5, heatmap_padding=0.1, colorbar_width=0.5, colorbar_padding=0.):
    # Extract labels:
    row_labels = {index_name: np.sort(df[index_name].unique()) for index_name in row_index_columns}
    # column_labels = {index_name: np.sort(df[index_name].unique()) for index_name in column_index_columns}
    # Specify GridSpec:
    widths = np.array(
        [unit*feature_names_width] + [unit] * len(row_labels) + 
        [unit*heatmap_padding] + 
        [unit*heatmap_cell_width] * len(sample_columns) + 
        [unit*colorbar_width])
    heights = np.array([unit] * (len(df) + 1) + [legend_offset*unit])
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots(
        ncols=len(widths), nrows=len(heights), subplot_kw=dict(frameon=False),
        constrained_layout=False, gridspec_kw=gs_kw, 
        figsize=(scale * widths.sum(), scale * heights.sum()))
    # Create a column for colorbar:
    gs = axs[1, -1].get_gridspec()
    for ax in axs[1:-1, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[1:-1, -1])
    # Find maximum value for samples
    np_sample = df[sample_columns].to_numpy()
    min_s, max_s = np_sample.min(), np_sample.max()
    # Define color mappings:
    ## Categorical:
    from matplotlib.cm import get_cmap
    cmaps = cycle(['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', ])
    def map_disc_(items, cmap_name):
        cmap_ = get_cmap(cmap_name)
        inter_ = 0.3
        slope_ = (1. - inter_) / len(items)
        return {value: cmap_(inter_ + items.tolist().index(value) * slope_) for value in items}
    row_colors = {label: map_disc_(items, cmap_name) \
        for (label, items), cmap_name in zip(row_labels.items(), cmaps)}
    # column_colors = {label: map_disc_(items, cmap_name) \
        # for (label, items), cmap_name in zip(column_labels.items(), cmaps)}
    ## Continuous:
    def lin_cmap_(inter, value, min_value, max_value):
        return inter + (value - min_value) * ((1. - inter) / np.abs(max_value - min_value))
    newgreys_ = lambda x: get_cmap('Greys')(lin_cmap_(0.2, x, min_s, max_s))
    newgreys = ListedColormap(newgreys_(np.linspace(min_s, max_s, 256)))
    # Apply settings for axes:
    for r, row in enumerate(axs):
        for c, ax in enumerate(row):
            ax.set(xticks=[], yticks=[])
            for _, spine in ax.spines.items():
                spine.set_visible(False)
            if r == len(axs) - 1:
                ax.patch.set_alpha(0.)
                continue
            if r >= 1 and c == 0:
                ax.text(0, 0.5, df[feature_column_name][r - 1], 
                    verticalalignment='center')
            if r == 0 and c > len(row_labels) + 1 and c < (len(row_labels) + 1 + len(sample_columns) + 1):
                ax.text(0.5, 0.5, sample_columns[c - len(row_labels) - 2], 
                    verticalalignment='center', horizontalalignment='center')
            if r >= 1 and c >= 1 and c < len(row_labels) + 1:
                var_name = list(row_labels.keys())[c - 1]
                var_value = df[var_name][r - 1]
                ax.add_patch(pch.Rectangle((0., 0.), 1., 1., 
                    facecolor=row_colors[var_name][var_value], edgecolor='none', label=var_value))
            elif r >= 1 and c >= 1 and c > len(row_labels) + 1 and c < (len(row_labels) + 1 + len(sample_columns) + 1):
                cmap = get_cmap('Greys')
                ax.patch.set_alpha(0.)
                var_name = sample_columns[c - len(row_labels) - 2]
                var_value = df[var_name][r - 1]
                ax.add_patch(pch.Rectangle((0., 0.), 1., 1., 
                    facecolor=newgreys_(var_value), edgecolor='none'))
    # Build the legend:
    sizes = [ll.shape[0] for ln, ll in row_labels.items()]
    largest_label_inx = np.argmax(sizes)
    ## Categorical variables:
    hndl = []
    for i, (lbl, colors_) in enumerate(row_colors.items()):
        size, empty = len(colors_), []
        if i >= 1:
            m = sizes[largest_label_inx] - sizes[i - 1]
            if m > 0:
                empty = [Line2D([], [], label='', alpha=0.)] * m
        hndl += empty + [Line2D([], [], label=lbl, alpha=0.)] + [
            pch.Patch(facecolor=color, edgecolor="k", label=label, alpha=0.7) 
            for label, color in colors_.items()
        ]
    hndl += [Line2D([], [], label='', alpha=0.)] * (sizes[largest_label_inx] - sizes[i])
    ## Continuous variables:
    fig.colorbar(ScalarMappable(
        norm=mpl.colors.Normalize(vmin=min_s, vmax=max_s), 
        cmap=newgreys), cax=axbig)
    legend = fig.legend(handles=hndl, loc='lower center', 
        handlelength=scale*1.4, handleheight=scale*1.6, 
        ncol=len(row_index_columns), labelspacing=.0)
    legend.get_frame().set_alpha(0.)
    plt.subplots_adjust(hspace=space, wspace=space)
    return fig
