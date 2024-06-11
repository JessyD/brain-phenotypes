from __future__ import annotations

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.extmath import svd_flip
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl


def corr(x,y):
    return (normalize(x) * normalize(y)).mean(axis=0)

def normalize(a):
    return (a - a.mean(axis=0,keepdims=True))/a.std(axis=0,keepdims=True)

def compute_r2(y_true, y_pred):
    r2 = np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in
        range(y_true.shape[1])])
    return r2

def compute_svd(dataTrainingZ, pov=0.95, verbose=None):
    """
    Compute SVD of the data
    Parameters
    ----------
    dataTrainingZ: Input data. Should already be z-scored.
    pov: percentage of variance explained (default is 95%)

    Returns
    -------
    U: orthonormal matrix obtained from SVD (left singular vector)
    Vt: orthonormal matrix obtained from SVD (right singular vector)
    s: singular values
    S: matrix form of s
    sinv: inverse of the singular values
    Sinv: matrix form of sinv
    Us: U scaled by s
    usmean: demeaned US
    usstdv: subtracted standard deviation from US
    explained_variance_ratio: percentage of variance explained by the components
    n_components: number of components that explain the passed variance
    """
    # compute the SVD of the Z-scored training data
    U, s, Vt = np.linalg.svd(dataTrainingZ, full_matrices=False, compute_uv=True)
    # To more info on why we are flipping the sign check this post
    # (https://stackoverflow.com/questions/44765682/in-sklearn-decomposition-pca-why-are-components-negative)
    # flip eigenvectors' sign to enforce deterministic output
    U, Vt = svd_flip(U, Vt)

    # find # dimensions for >= 95% of the variance
    explained_variance = (s ** 2) / len(U)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    ratio_cumsum = stable_cumsum(explained_variance_ratio)
    if 0 < pov < 1:
        n_components = np.searchsorted(ratio_cumsum, pov, side='right') + 1
    else:
        n_components = min(dataTrainingZ.shape)
    if verbose:
        print('%0.2f of variance => %d SVD components' % (pov, n_components))

    # trim all the matrices accordingly (and make s and sinv diagonal matrices)
    U = U[:, :n_components]
    Vt = Vt[:n_components, :]
    s = s[:n_components]
    S = np.diag(s)
    sinv = 1 / s
    Sinv = np.diag(sinv)
    explained_variance_ratio = explained_variance_ratio[:n_components]

    # for the svd we will scale u by s and therefore use U*s
    Us = U * s
    # Z-score Us for saving
    usmean = np.mean(Us, axis=0)
    usstdv = np.std(Us, axis=0, ddof=1)
    return U, Vt, s, S, sinv, Sinv, Us, usmean, usstdv, explained_variance_ratio, n_components


def plot_total_explained_variance(output_figure, exp_var, cum_sum, color):
    plt.figure(figsize=(4, 2))
    plt.bar(range(0,len(exp_var)), exp_var, alpha=0.5, align='center',
            color=color)
    plt.step(range(0,len(cum_sum)), cum_sum, where='mid',
            label='Cumulative', color=color)

    # add line with the total amount of variance explained
    idx_var = np.argwhere(cum_sum >= 0.95)
    plt.axvline(x=idx_var[0], color='k', linestyle='--')
    plt.ylabel('Explained variance')
    plt.xlabel('Singular Value index')
    plt.legend(loc='upper left', frameon=False)
    if color == 'g':
        plt.title('HCP')
    elif color == 'b':
        plt.title('PNC')
    elif color == 'darkorchid':
        plt.title('HBN')
    plt.tight_layout()
    # only add number from the HCP plot, as from PNC plot it can easily be inferred
    if color == 'g': # HCP
        ticks = [x for x in range(0, len(cum_sum), 20)]
        ticks.append(idx_var[0][0])
        plt.xticks(sorted(ticks))
    #plt.text(idx_var[0], -.5, f'{idx_var[0]}',
    #            ha='center', va='top')
    plt.grid(True,  alpha=0.5, linestyle="--")

    plt.savefig(output_figure, dpi=300)

    plt.close()

def get_phenotypes(phenotype_path, data_type):
    """Load the type of phenotype to be used
    data_type: Which type of data to use (numeric, categorical, ordinal, ar all)
    """
    train_phenotype = pd.read_csv(phenotype_path / f'phenotype_measures_imputation-imputed_{data_type}_training_zscored.txt',
                                  index_col='participant')
    testing_phenotype = pd.read_csv(
        phenotype_path / f'phenotype_measures_imputation-imputed_{data_type}_testing_zscored.txt',
        index_col='participant')
    val_phenotype = pd.read_csv(phenotype_path / f'phenotype_measures_imputation-imputed_{data_type}_validation_zscored.txt',
                                index_col='participant')

    # sort columns alphabetically
    train_phenotype = train_phenotype.sort_index(axis=1)
    val_phenotype = val_phenotype.sort_index(axis=1)

    # Load family information for the subjects
    train_demographics = pd.read_csv(phenotype_path / 'phenotype_measures_separate_training.txt',
                                     index_col='participant')
    testing_demographics = pd.read_csv(phenotype_path / 'phenotype_measures_separate_testing.txt',
                                       index_col='participant')
    val_demographics = pd.read_csv(phenotype_path / 'phenotype_measures_separate_validation.txt',
                                   index_col='participant')
    return train_phenotype, testing_phenotype, val_phenotype, train_demographics, testing_demographics, val_demographics


def get_fc(training_df, validation_df):
    training_data = []
    for index, row in training_df.iterrows():
        training_data.append(np.load(row['path']))
    training_data = np.array(training_data)

    validation_data = []
    for index, row in validation_df.iterrows():
        validation_data.append(np.load(row['path']))
    validation_data = np.array(validation_data)
    return training_data, validation_data


def get_fc_df(prediction_experiment_path):
    training_df = pd.read_csv(prediction_experiment_path / 'examples-training.csv',
                              delimiter=',', index_col='participant')
    validation_df = pd.read_csv(prediction_experiment_path / 'examples-validation.csv',
                                delimiter=',', index_col='participant')
    return training_df, validation_df

def transform_fc(training_data, validation_data, data_normalisation, latent_images, latent_images_type, output_path):
    """Perform the normalisation or dimensionality reduction according to your needs"""
    if data_normalisation == 'centered-row':
        source_mean_train = np.mean(training_data, axis=1).reshape(-1, 1)
        source_mean_val = np.mean(validation_data, axis=1).reshape(-1, 1)
        fc_train = training_data - source_mean_train
        fc_validation = validation_data - source_mean_val
    elif data_normalisation == 'centered-column':
        # using the mean from the training data
        source_mean = np.mean(training_data, axis=0).reshape(1, -1)
        fc_train = training_data - source_mean
        fc_validation = validation_data - source_mean
        np.savez((output_path / 'imaging_data_stats.npz'), source_mean=source_mean)
    elif data_normalisation == 'zscored-column':
        source_mean = np.mean(training_data, axis=0).reshape(1, -1)
        source_std = np.std(training_data, axis=0).reshape(1, -1)
        fc_train = (training_data - source_mean) / source_std
        fc_validation = (validation_data - source_mean) / source_std
        np.savez((output_path / 'imaging_data_stats.npz'), source_mean=source_mean, source_std=source_std)
    elif data_normalisation == 'raw':
        print("Use images as it is...")

    if latent_images:
        if latent_images_type == 'svd':
            raise ValueError('This is not implemented yet!')

    return fc_train, fc_validation

def plot_scores(metrics, dataset, targets, color, figs_dir, latent_targets_type):

    plt.rcParams.update({
        'font.size' : 28,                   # Set font size to 11pt (28)
        'axes.labelsize': 28,               # -> axis labels
        'legend.fontsize': 28,              # -> legends
        'axes.titlesize': 28,
        #'font.family': 'lmodern',
        #'text.usetex': True,
        #'text.latex.preamble': (            # LaTeX preamble
        #    r'\usepackage{lmodern}'
            # ... more packages if needed
        #),
        'axes.xmargin': 0.02
    })
    # save just the correlation
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(30, 9), sharex=True)

    # Correlation
    corr_mean = np.mean(metrics['Correlation'], axis=0)
    corr_std = np.std(metrics['Correlation'], axis=0)

    # Sort the results by performance, if we are not using SVDif latent_targets_type in ['None', 'reconstruction_SVD']:
    if latent_targets_type in ['None', 'reconstructed_SVD']:
        arg_sort = np.argsort(-corr_mean)
        corr_mean = corr_mean[arg_sort]
        corr_std = corr_std[arg_sort]
        targets = targets[arg_sort]
    else:
        arg_sort = range(len(metrics["Correlation"]))
    ax.plot(targets, corr_mean, 'o-', label=dataset,
            color='k')
    ax.fill_between(targets, corr_mean - corr_std, corr_mean + corr_std, alpha=0.3, facecolor=color)
    ax.set_title(f'{dataset}')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.grid(alpha=.2)
    if dataset == 'PNC':
        ax.set_ylim([-0.1, 0.4])
    elif dataset == 'HCP':
        if latent_targets_type == "None":
            ax.set_ylim([-0.1, 0.7])
        else:
            ax.set_ylim([-0.1, 0.5])
    plt.tight_layout()
    plt.savefig(figs_dir / f'evaluation_correlation_{dataset}.pdf')
    plt.close()

    # save both R^2 and Correlation
    # ------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(30, 18), sharex=True)

    ax[0].plot(targets, corr_mean, 'o-', label=dataset,
            color='k')
    ax[0].fill_between(targets, corr_mean - corr_std, corr_mean + corr_std, alpha=0.3, facecolor=color)
    ax[0].set_title(f'{dataset} - Correlation')
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(90)
    ax[0].grid(alpha=.2)
    if dataset == 'PNC':
        ax[0].set_ylim([-0.1, 0.4])
    elif dataset == 'HCP':
        if latent_targets_type == "None":
            ax[0].set_ylim([-0.1, 0.7])
        else:
            ax[0].set_ylim([-0.1, 0.5])


    # R2
    r2_mean = np.mean(metrics['R2'], axis=0)
    r2_std = np.std(metrics['R2'], axis=0)

    # Sort the results by performance, if we are not using SVD. Use the same order as the correlation
    if latent_targets_type in ['None', 'reconstructed_SVD']:
        r2_mean = r2_mean[arg_sort]
        r2_std = r2_std[arg_sort]

    ax[1].plot(targets, r2_mean, 'o-', label=dataset,
            color='k')
    ax[1].fill_between(targets, r2_mean - r2_std, r2_mean + r2_std, alpha=0.3,
            facecolor=color)
    ax[1].set_title('$R^2$')
    ax[1].grid(alpha=.2)
    if dataset == 'PNC':
        ax[1].set_ylim([-0.1, 0.12])
    elif dataset == 'HCP':
        ax[1].set_ylim([-0.1, 0.20])

    # If not SVD
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(90)

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=1)
    plt.tight_layout()
    plt.savefig(figs_dir / f'evaluation_metrics_{dataset}.pdf')
    return targets, arg_sort



def plot_scores_reconstruction(metrics, dataset, targets, color, figs_dir, components_dict):
    plt.rcParams.update({
        'font.size': 16,                   # Set font size to 11pt
        'axes.labelsize': 16,               # -> axis labels
        'legend.fontsize': 16,              # -> legends
        #'font.family': 'lmodern',
        #'text.usetex': True,
        #'text.latex.preamble': (            # LaTeX preamble
        #    r'\usepackage{lmodern}'
            # ... more packages if needed
        #),
        'axes.xmargin': 0.02
    })
    n_components = len(components_dict)
    fig, ax = plt.subplots(1, 1, figsize=(18, 10), sharex=True)

    # Correlation
    corr_mean = np.mean(metrics['Correlation'], axis=0)

    # Sort the results by performance, we are using the same order as those from the bootstrap
    path = figs_dir.parent
    tmp = np.load(path / f"predictions-{dataset.lower()}_targets-zscore_column" / "metrics.npz", allow_pickle=True)
    sorted_targets = tmp["sorted_targets"]

    idx = 1
    alphas = np.arange(0.2, 1, 1/(n_components + 1))
    for component in components_dict.keys():
        ax.plot(sorted_targets, components_dict[component], 'o-', label=component,
                   color=color, alpha=alphas[idx-1])
        ax.set_title('$R^2$')
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        ax.grid(alpha=.2)
        idx += 1
    #if dataset == 'PNC':
    #    ax.set_ylim([-1, 1])
    #    ax.set_ylim([-1, 1])
    #elif dataset == 'HCP':
    #    ax[idx].set_ylim([-0.1, 0.65])

    # If not SVD
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / 'r2_components.pdf')
    plt.close()
