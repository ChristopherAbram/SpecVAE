import os
import numpy as np
# import config
import matplotlib
# if config.use_agg:
    # matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_spectra_grid(model, data_batch, dirpath='.', epoch=0, device=None, transform=None):
    name = "%s_spec_%s" % (model.get_name(), epoch)
    x_batch = data_batch[0]
    data_batch2 = model(x_batch)
    resolution, max_mz = model.resolution, model.max_mz

    if transform:
        data_batch = transform(x_batch)
        data_batch2 = transform(data_batch2)

    if hasattr(model, 'lattice') and epoch % 5 == 0:
        plot_spectra_(model.lattice(grid=(17, 17), zrange=(-75, 75), device=device), grid=(17, 17), figsize=(34, 18),
            filepath=os.path.join(dirpath, 'lattice_%s.png' % epoch), 
            resolution=resolution, max_mz=max_mz)

    plot_spectra_compare(data_batch, data_batch2, grid=(3, 3), figsize=(20, 20), 
        filepath=os.path.join(dirpath, name), resolution=resolution, max_mz=max_mz)


def plot_spectra_(spectra, grid=(4, 3), figsize=(17, 9), filepath=None, resolution=0.05, max_mz=2500):
    fig, axs = plt.subplots(grid[0], grid[1], figsize=figsize)
    for i, ax in enumerate(axs.flat):
        if i >= spectra.shape[0]:
            break

        mz = np.arange(0, max_mz, step=resolution)
        ax.plot(mz, spectra[i].tolist(), color='blue')
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
    plt.close(fig)


def plot_spectra_compare(spectra1, spectra2, grid=(4, 3), figsize=(17, 9), filepath=None, resolution=0.05, max_mz=2500):
    fig, axs = plt.subplots(grid[0], grid[1], figsize=figsize)
    for i, ax in enumerate(axs.flat):
        if i >= spectra1.shape[0] or i >= spectra2.shape[0]:
            break

        mz = np.arange(0, max_mz, step=resolution)
        ax.plot(mz, (-spectra2[i]).tolist(), color='red')
        ax.plot(mz, spectra1[i].tolist(), color='blue')
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
    plt.close(fig)


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


def plot_distribution(data, subject, xlabel, ylabel, plot_density=False):
    from sklearn.neighbors import KernelDensity

    counts, bins = np.histogram(data, bins=10, density=True)
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
    xmajor_ticks = np.arange(0., 1.1, 0.1)
    xminor_ticks = np.arange(0., 1.05, 0.05)
    # ymajor_ticks = np.arange(0., 0.35, 0.05)
    # yminor_ticks = np.arange(0., 0.35, 0.05)

    ax.set_xticks(xmajor_ticks)
    ax.set_xticks(xminor_ticks, minor=True)
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