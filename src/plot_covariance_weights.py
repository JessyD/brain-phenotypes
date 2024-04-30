import numpy
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import re
import matplotlib.lines as lines


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figs_dir_pnc", help="Path where the figures will be saved")
    parser.add_argument("--figs_dir_hcp", help="Path where the figures will be saved")
    parser.add_argument("--figs_dir_hbn", help="Path where the figures will be saved")
    args = parser.parse_args()
    return args

def load_data(figs_dir):

    # load the data
    with open(figs_dir / 'covariate_regression.pickle', 'rb') as handle:
        cov_regression = pickle.load(handle)

    beta_coeffs = cov_regression['beta_coeffs']
    #labels = cov_regression['labels']
    cov_train = cov_regression['cov_train']
    train_phenotype = cov_regression['index']


    # Average the results from all repetitions
    beta_mean = np.mean(beta_coeffs, axis=0)
    beta_std = np.std(beta_coeffs, axis=0)
    labels = (np.asarray(["{:.2f} \n ({:.2f})".format(mean, std) if abs(mean) > .1 else "" for mean, std in zip(beta_mean.flatten(), beta_std.flatten())])
                  ).reshape(beta_coeffs.shape[1:])
    df_beta_coeffs = pd.DataFrame(beta_mean,
                                    columns=cov_train.values,
                                    index=train_phenotype.values
                                    )

    # rename the columns and make them more plot friendly
    df_beta_coeffs = df_beta_coeffs.rename({
        'age_at_scan_zscore': f'age',
        'sex_at_birth_reg': f'sex',
        'age_sex': r'age $\times$ sex',
        'age2': r'age$^{2}$',
        'age2_sex': r'age$^{2} \times$ sex'
        }, axis='columns')

    return df_beta_coeffs, labels, train_phenotype

def plot_covariates_heatmap(df_beta_coeffs_hcp, labels_hcp,
        train_phenotype_hcp,
        df_beta_coeffs_pnc, labels_pnc, train_phenotype_pnc, out_path,
        df_beta_coeffs_hbn, labels_hbn,
        annotate=False):

    # set font size depending if we want to annotate the plot or not
    if annotate:
        plt.rcParams.update({
            'font.size' : 25,                   # Set font size to 11pt
            'axes.labelsize': 25,               # -> axis labels
            'legend.fontsize': 25,              # -> legends
            'font.family': 'serif',
            #'font.serif': ['Computer Modern'],
            #'text.usetex': True,
            #'text.latex.preamble': (            # LaTeX preamble
            #    r'\usepackage{lmodern}'
            #    # ... more packages if needed
            #)
        })
    else:
        plt.rcParams.update({
            'font.size' : 30,                   # Set font size to 11pt
            'axes.labelsize': 30,               # -> axis labels
            'legend.fontsize': 30,              # -> legends
            'font.family': 'serif',
            #'font.serif': ['Computer Modern'],
            #'text.usetex': True,
            #'text.latex.preamble': (            # LaTeX preamble
            #    r'\usepackage{lmodern}'
            #    # ... more packages if needed
            #)
        })


    figsize = (32, 32)
    fig_label = ['a', 'b', 'c']
    # Split the targets that have Unadjusted and those that do not have
    # adjusted information
    pattern = re.compile(r"(adj|AgeAdj|Unadj)")
    adj_unadj_elements = []
    remaining_elements = []

    # initialise list to store the original indices
    adj_unadj_indices = []
    remaining_indices = []

    # iterate through the phenotypes
    for i, item in enumerate(train_phenotype_hcp.to_list()):
        # Check if item matches the pattern
        if pattern.search(item):
            adj_unadj_elements.append(item)
            adj_unadj_indices.append(i)
        else:
            remaining_elements.append(item)
            remaining_indices.append(i)
    # Sort both lists alphabetically
    adj_unadj_elements.sort()
    remaining_elements.sort()

    # Draw histogram with the beta coefficients - annotated (with 100 repetitions)
    fig, (cax, ax) = plt.subplots(nrows=2, ncols=4, figsize=figsize,
            gridspec_kw={"height_ratios":[0.025, 1]})
    sns.heatmap(df_beta_coeffs_hcp.iloc[adj_unadj_indices],
                cmap='bwr',
                annot=labels_hcp[adj_unadj_indices] if annotate else False,
                ax=ax[0],
                vmin=-1.6, # values were set based on maximum scores
                vmax=1.6,
                fmt='',
                cbar=False,
                )
    cbar = fig.colorbar(ax[0].get_children()[0], cax=cax[0], orientation="horizontal",
                        ticks=[-1.6, 0, 1.6])
    cbar.ax.set_xticklabels(['-1.6', '0', '1.6'])  # vertically oriented colorbar
    # picture label
    cax[0].text(-2.5, .1, fig_label[0], size=45, weight='bold')
    for tick in ax[0].get_xticklabels():
                tick.set_rotation(90)

    # Add some horizontal lines to separate those adj from the undj pairs
    ax[0].hlines([3, 5, 14, 16, 19, 21, 27, 29, 31, 32, 34, 36], *ax[0].get_xlim(),
            color='k', linestyles='dotted')

    sns.heatmap(df_beta_coeffs_hcp.iloc[remaining_indices],
                cmap='bwr',
                annot=labels_hcp[remaining_indices] if annotate else False,
                ax=ax[1],
                vmin=-1.6, # values were set based on maximum scores
                vmax=1.6,
                fmt='',
                cbar=False,
                )
    cbar = fig.colorbar(ax[1].get_children()[0], cax=cax[1], orientation="horizontal",
                        ticks=[-1.6, 0, 1.6])
    cbar.ax.set_xticklabels(['-1.6', '0', '1.6'])   # vertically oriented colorbar
    #plt.axhline(y=2, color='black', linewidth=2, clip_on=False)
    #plt.line([1, 0],[0, 1], color="k", linewidth=1, clip_on=False)
    #cax[0].add_artist(lines.Line2D([0, 1], [0, 1]))
    for tick in ax[1].get_xticklabels():
                tick.set_rotation(90)


    # The main difference is that for PNC we do not define a max range
    sns.heatmap(df_beta_coeffs_pnc, cmap='bwr',
                annot=labels_pnc if annotate else False,
                ax=ax[2],
                vmin=-1.6, # values were set based on maximum scores
                vmax=1.6,
                fmt='',
                cbar=False,
                )
    cbar = fig.colorbar(ax[2].get_children()[0], cax=cax[2], orientation="horizontal",
                        ticks=[-1.6, 0, 1.6])
    cbar.ax.set_xticklabels(['-1.6', '0', '1.6'])
    #ax[2].text(-1.5, -0.7, fig_label[1], size=20, weight='bold')
    cax[2].text(-2.5, 0.1, fig_label[1], size=45, weight='bold')
    for tick in ax[2].get_xticklabels():
                tick.set_rotation(90)


    # The main difference is that for HBN we do not define a max range
    sns.heatmap(df_beta_coeffs_hbn, cmap='bwr',
                annot=labels_hbn if annotate else False,
                ax=ax[3],
                vmin=-1.6, # values were set based on maximum scores
                vmax=1.6,
                fmt='',
                cbar=False,
                )
    cbar = fig.colorbar(ax[3].get_children()[0], cax=cax[3], orientation="horizontal",
            ticks=[-1.6, 0, 1.6])
    cbar.ax.set_xticklabels(['-1.6', '0', '1.6'])  # vertically oriented colorbar
    #ax[2].text(-1.5, -0.7, fig_label[1], size=20, weight='bold')
    cax[3].text(-2.5, 0.1, fig_label[2], size=45, weight='bold')
    for tick in ax[3].get_xticklabels():
                tick.set_rotation(90)

    plt.tight_layout()
    if annotate:
        plt.savefig(out_path / f'beta_coeffs_heatmap_annotated.pdf')
    else:
        plt.savefig(out_path / f'beta_coeffs_heatmap.pdf')

    plt.close()

    # Now plot HBN results alone
    #fig, ax = plt.subplots(figsize=figsize)
    fig, (cax, ax) = plt.subplots(nrows=2, ncols=1, figsize=figsize,
                                  gridspec_kw={"height_ratios": [0.025, 1]})

    # The main difference is that for HBN we do not define a max range
    im = sns.heatmap(df_beta_coeffs_hbn, cmap='bwr',
                annot=labels_hbn if annotate else False,
                ax=ax,
                cbar_ax=cax,
                cbar_kws={'orientation': 'horizontal'},
                vmin=-1.6, # values were set based on maximum scores
                vmax=1.6,
                fmt='',
                cbar=True,
                )
    #cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    #cbar = fig.colorbar(im, cax=cax, orientation="horizontal",
    #        ticks=[-1.6, 0, 1.6])
    #cbar.ax.set_xticklabels(['-1.6', '0', '1.6'])  # vertically oriented colorbar
    #cax.text(-2.5, 0.1, fig_label[2], size=45, weight='bold')
    for tick in ax.get_xticklabels():
                tick.set_rotation(90)

    plt.tight_layout()
    plt.savefig(out_path / f'beta_coeffs_heatmap_HBN.pdf')


def main(figs_dir_hcp, figs_dir_pnc, figs_dir_hbn):

    # where to save the output figure
    out_path = Path('/Users/jdafflon/Code/brain-phenotypes/outputs/covariates')
    out_path.mkdir(exist_ok=True, parents=True)

    df_beta_coeffs_hcp, labels_hcp, train_phenotype_hcp = load_data(figs_dir_hcp)
    df_beta_coeffs_pnc, labels_pnc, train_phenotype_pnc = load_data(figs_dir_pnc)
    df_beta_coeffs_hbn, labels_hbn, train_phenotype_hbn = load_data(figs_dir_hbn)

    # Draw histogram of the beta coefficients
    # datset: HCP
    fig = plt.figure(figsize=(8,16))
    ax = fig.gca()
    hist = df_beta_coeffs_hcp.hist(bins=20, ax=ax)
    plt.savefig(figs_dir_hcp / 'beta_coeffs_dist.pdf')
    # datset: PNC
    fig = plt.figure(figsize=(8,16))
    ax = fig.gca()
    hist = df_beta_coeffs_pnc.hist(bins=20, ax=ax)
    plt.savefig(figs_dir_pnc / 'beta_coeffs_dist.pdf')


    # plot heatmaps with annotation
    plot_covariates_heatmap(df_beta_coeffs_hcp, labels_hcp,
        train_phenotype_hcp,
        df_beta_coeffs_pnc, labels_pnc, train_phenotype_pnc, out_path,
        df_beta_coeffs_hbn, labels_hbn,
        annotate=True)

    # plot heatmaps without annotation
    plot_covariates_heatmap(df_beta_coeffs_hcp, labels_hcp,
        train_phenotype_hcp,
        df_beta_coeffs_pnc, labels_pnc, train_phenotype_pnc, out_path,
        df_beta_coeffs_hbn, labels_hbn,
        annotate=False)

if __name__ == '__main__':
    args = parse_args()
    figs_dir_pnc = Path(args.figs_dir_pnc)
    figs_dir_hcp = Path(args.figs_dir_hcp)
    figs_dir_hbn = Path(args.figs_dir_hbn)
    main(figs_dir_hcp, figs_dir_pnc, figs_dir_hbn)
