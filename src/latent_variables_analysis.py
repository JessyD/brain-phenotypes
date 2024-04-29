import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import imageio
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from util import compute_svd, get_phenotypes, plot_total_explained_variance

def organise_covariate(train_demographics):
    # zscore age
    with pd.option_context('mode.chained_assignment', None):
        age_mean = train_demographics['age_at_scan'].mean()
        age_std = train_demographics['age_at_scan'].std()
        train_demographics.loc[:, 'age_at_scan_zscore'] = (train_demographics.loc[:, 'age_at_scan'] - age_mean) / age_std
        # brainblocks standard: 1(female) & 2(male). Now: 0 (female) & 1 (male)
        train_demographics.loc[:, 'sex_at_birth_reg'] = train_demographics.loc[:, 'sex_at_birth'] - 1
        # define covariate interactions
        train_demographics.loc[:, 'age_sex'] = train_demographics.loc[:, 'age_at_scan_zscore'] * train_demographics.loc[:,'sex_at_birth_reg']

        train_demographics.loc[:, 'age2'] = train_demographics.loc[:, 'age_at_scan_zscore'] ** 2
        train_demographics.loc[:, 'age2_sex'] = train_demographics.loc[:, 'age2'] * train_demographics.loc[:, 'sex_at_birth_reg']

    return train_demographics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--figs_dir", help="Path where the figures will be saved")
    parser.add_argument("--project_dir", help="Root directory")
    parser.add_argument("--regress_out_covariates", action='store_true',
            help="Specify if covariates should be regressed out")
    parser.add_argument("--dataset", help="which dataset to use", choices=["HCP",
        "PNC", "HBN"])

    args = parser.parse_args()
    return args

def main(random_seed, args):
    np.random.seed(random_seed)
    print(f"Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # load the config file
    config = OmegaConf.load(args.config_file)

    # Load the phenotype data
    project_dir = Path(args.project_dir)
    brainblocks_version = config['brainblocks']["bbversion"]
    phenotype_path = project_dir / config["dataset"] / f'brainblocks-{brainblocks_version}' / "phenotype"

    train_phenotype, _, val_phenotype, train_demographics, _, val_demographics = \
        get_phenotypes(phenotype_path, config['prediction']['data_type'])
    full_phenotype = pd.concat([train_phenotype, val_phenotype])
    full_demographics = pd.concat([train_demographics, val_demographics])

    if args.regress_out_covariates:
        # regress age and sex from the phenotypes
        reg = LinearRegression()
        cov_train = organise_covariate(full_demographics)

        cov_train = cov_train.drop(['family_ID', 'age_at_scan', 'sex_at_birth'], axis=1)
        reg.fit(cov_train, full_phenotype)
        y_train_predict = reg.predict(cov_train)
        full_phenotype = full_phenotype - y_train_predict

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(exist_ok=True, parents=True)

    # create gif folder
    gif_dir = figs_dir / 'gif'
    gif_dir.mkdir(exist_ok=True, parents=True)

    # Perform the dimensionality reduction
    U_split1, Vt_split1, _, _, _, _, Us_split1, usmean_split1, usstdv_split1, \
        explained_variance_ratio_split1, n_component_split1 = compute_svd(full_phenotype, pov=1)

    # plot the total explained variance
    output_figure = figs_dir / 'total_var_explained.pdf'
    if args.dataset == 'PNC':
        color = 'b'
    elif args.dataset == 'HCP':
        color = 'g'
    elif args.dataset == 'HBN':
        color = 'darkorchid'
    cum_sum = np.cumsum(explained_variance_ratio_split1)
    plot_total_explained_variance(output_figure,
            explained_variance_ratio_split1, cum_sum, color)

    # Find how many components are needed to explain at least 95% of the variance
    print(f"Explain at least 95% of the variance:")
    idx_var = np.argwhere(cum_sum >= 0.95)
    print(f"# components needed = {idx_var[0]}")
    print(f"Explained variance for the first 5 components: {cum_sum[:5]}")
    print(f"Explained variance for the first 2 components: {cum_sum[:2]}")


    print(f"How much variance is explained by the first component: {explained_variance_ratio_split1[0]:.3f}")
    # Visualise US
    fig = plt.figure()
    im = plt.imshow(Us_split1, cmap='bwr', aspect='auto')
    plt.colorbar(im, shrink=0.5)
    plt.tight_layout()
    plt.savefig(figs_dir / f'Us.pdf')
    plt.close()

    # make a tiny gif on how the components are changing
    all_componentes = full_phenotype.shape[1]
    for n_component in range(1, all_componentes + 1):
        recons = Us_split1[:, :n_component] @ Vt_split1[:n_component, :]
        fig = plt.figure(figsize=(22, 22))
        im = plt.imshow(recons, cmap='bwr', aspect='auto', vmin=-6, vmax=6)
        plt.colorbar(im, shrink=0.5)
        plt.title(f"Reconstruction: # components: {n_component}")
        plt.xticks(range(len(full_phenotype.columns)), full_phenotype.columns.to_list(), rotation=45)
        plt.tight_layout()
        plt.savefig(gif_dir / f'components_loadings_{n_component}.png')
        plt.close()

    frames = []
    for n_component in range(1, all_componentes + 1):
        image = imageio.v2.imread(gif_dir / f'components_loadings_{n_component}.png')
        frames.append(image)
    imageio.mimsave(gif_dir / f'components_loadings.gif',  # output gif
                    frames,  # array of input frames
                    duration=all_componentes * 6)  # duration=in seconds (1000 * 1/frames_per_second).

    # Get the top 10 components
    dict = {}
    cols = full_phenotype.columns.to_numpy()
    top_n = 5
    for idx, component in enumerate(range(n_component)):
        dict[idx] = cols[np.argsort(Vt_split1[component, :])[::-1][:top_n]]
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(figs_dir / 'top_5_components.csv')


if __name__ == '__main__':
    random_seed = 2
    args = parse_args()
    main(random_seed, args)
