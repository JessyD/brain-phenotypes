"""
This script computes the prediction target with bootstrap.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from tqdm import tqdm

from util import (get_phenotypes, get_fc, get_fc_df, transform_fc, plot_scores,
        compute_svd, corr, normalize, compute_r2)

plt.rcParams.update({
    'font.size' : 28,                   # Set font size to 11pt
    'axes.labelsize': 28,               # -> axis labels
    'legend.fontsize': 28,              # -> legends
    #'font.family': 'lmodern',
    #'text.usetex': True,
    #'text.latex.preamble': (            # LaTeX preamble
    #    r'\usepackage{lmodern}'
        # ... more packages if needed
    #),
    'axes.xmargin': 0.02
})

def my_resample(ids, fam_id_training, replace=True, n_samples=4, random_state=0):
    np.random.seed(random_state)
    np.random.shuffle(fam_id_training.to_numpy())
    resampled = []
    idx = 0
    for fam in np.unique(fam_id_training.to_numpy()):
        if idx < n_samples:
            i_fam = ids[(fam_id_training == fam).to_numpy()]
            resampled.append(i_fam)
            idx = idx + len(i_fam)

            # resample some families
            if idx < n_samples:
                if replace:
                    if idx % 2:
                        resampled.append(i_fam)
                        idx = idx + len(i_fam)
        else:
            return np.concatenate(resampled)
    return np.concatenate(resampled)

def organise_covariate(train_demographics, val_demographics):
    # zscore age
    with pd.option_context('mode.chained_assignment', None):

        age_mean = train_demographics['age_at_scan'].mean()
        age_std = train_demographics['age_at_scan'].std()
        train_demographics.loc[:, 'age_at_scan_zscore'] = (train_demographics.loc[:, 'age_at_scan'] - age_mean) / age_std
        val_demographics.loc[:, 'age_at_scan_zscore'] = (val_demographics.loc[:, 'age_at_scan'] - age_mean) / age_std
        # brainblocks standard: 1(female) & 2(male). Now: 0 (female) & 1 (male)
        train_demographics.loc[:, 'sex_at_birth_reg'] = train_demographics.loc[:, 'sex_at_birth'] - 1
        val_demographics.loc[:, 'sex_at_birth_reg'] = val_demographics.loc[:, 'sex_at_birth'] - 1
        # define covariate interactions
        train_demographics.loc[:, 'age_sex'] = train_demographics.loc[:, 'age_at_scan_zscore'] * train_demographics.loc[:,'sex_at_birth_reg']
        val_demographics.loc[:, 'age_sex'] = val_demographics.loc[:, 'age_at_scan_zscore'] * val_demographics.loc[:, 'sex_at_birth_reg']

        train_demographics.loc[:, 'age2'] = train_demographics.loc[:, 'age_at_scan_zscore'] ** 2
        val_demographics.loc[:, 'age2'] = val_demographics.loc[:, 'age_at_scan_zscore'] ** 2
        train_demographics.loc[:, 'age2_sex'] = train_demographics.loc[:, 'age2'] * train_demographics.loc[:, 'sex_at_birth_reg']
        val_demographics.loc[:, 'age2_sex'] = val_demographics.loc[:, 'age2'] * val_demographics.loc[:, 'sex_at_birth_reg']

        #train_demographics.loc[:, 'age3'] = train_demographics.loc[:, 'age_at_scan_zscore'] ** 3
        #val_demographics.loc[:, 'age3'] = val_demographics.loc[:, 'age_at_scan_zscore'] ** 3
        #train_demographics.loc[:, 'age3_sex'] = train_demographics.loc[:, 'age3'] * train_demographics.loc[:, 'sex_at_birth_reg']
        #val_demographics.loc[:, 'age3_sex'] = val_demographics.loc[:, 'age3'] * val_demographics.loc[:, 'sex_at_birth_reg']

    return train_demographics, val_demographics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--figs_dir", help="Path where the figures will be saved")
    parser.add_argument("--output_dir", help="Path where the outputs will be saved")
    parser.add_argument("--project_dir", help="Root directory")
    parser.add_argument("--fc_dir", help="Folder containing the .npy with the FC")
    parser.add_argument("--predictions_dir", help="Folder with the prediction IDs")
    parser.add_argument("--phenotype_dir", help="Path where to find the phenotypes")
    parser.add_argument("--experiment_name", help="Name of the experiment")
    parser.add_argument("--regress_out_covariates", action='store_true',
            help="Specify if covariates should be regressed out")
    parser.add_argument("--n_components", type=int,
            help="Number of components to run the analysis on")
    args = parser.parse_args()
    return args



def main(random_seed, args):
    np.random.seed(random_seed)
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    # load the config file
    config = OmegaConf.load(args.config_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(exist_ok=True, parents=True)
    print(f'figs_dir: {figs_dir}')
    #------------------------------------------------------------------------------------------------------------------
    # Phenotypes
    #------------------------------------------------------------------------------------------------------------------
    # get the phenotypes
    project_dir = Path(args.project_dir)
    brainblocks_version = config['brainblocks']["bbversion"]
    phenotype_path = Path(args.phenotype_dir)
    train_phenotype, _, val_phenotype, train_demographics, _, val_demographics = \
        get_phenotypes(phenotype_path, config['prediction']['data_type'])
    
    # clean some of the long labels for visualisation purpose
    # for the purpose of improving the layout, re-write the labels
    train_phenotype.rename({'Language_Task_Story_Avg_Difficulty_Level': 'Language_Task_Story_Avg',
                            'Language_Task_Math_Avg_Difficulty_Level': 'Language_Task_Math_Avg',
                            }, 
                           axis='columns', inplace=True)
    val_phenotype.rename({'Language_Task_Story_Avg_Difficulty_Level': 'Language_Task_Story_Avg',
                            'Language_Task_Math_Avg_Difficulty_Level': 'Language_Task_Math_Avg',
                            }, 
                           axis='columns', inplace=True)

    #------------------------------------------------------------------------------------------------------------------
    # get the images
    #------------------------------------------------------------------------------------------------------------------
    predictions_dir = Path(args.predictions_dir)
    train_df, val_df = get_fc_df(predictions_dir)
    training_data, validation_data = get_fc(train_df, val_df)
    fam_id_training = train_df['family_ID']
    fam_id_validation = val_df['family_ID']
    # If needed perform normalisations on the FC
    fc_train, fc_validation = transform_fc(training_data, validation_data,
                                           config['prediction']['data_normalisation'],
                                           config['prediction']['latent_images'],
                                           config['prediction']['latent_images_type'],
                                           output_dir)


    #------------------------------------------------------------------------------------------------------------------
    # train the ML model
    #------------------------------------------------------------------------------------------------------------------
    # get ids for the specified number of splits
    #n_splits = 10
    n_splits = 100
    #gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=0)
    #splitter = gss.split(train_df, groups=fam_id_training)


    # create correlation scorer
    parameters = {'alpha': 10 ** np.linspace(2, 7, 100)}

    # Note: we will use all nodes available We will also use the lower MSE to
    # refit the model
    metrics = {'Correlation': [],
               "R2": [],
               "MSE": [],
               "best_l2": [],
               "best_alpha": []}

    training_size = int(np.floor(len(train_df) * .7))
    beta_coeffs = np.zeros((n_splits, train_phenotype.shape[1], 5))
    #arg_sorts = np.zeros((n_splits, train_df.shape[1]))
    y_train_recons = []
    y_val_recons = []
    y_train_predicted = []
    y_val_predicted = []
    all_train_orig = []
    all_val_orig = []

    save_y_train_true = []
    save_y_train_predicted = []
    save_y_val_true = []
    save_y_val_predicted = []
    with parallel_backend(backend='loky', n_jobs=1):
        for i_fold in tqdm(range(n_splits), total=n_splits):
            ids = np.array(range(len(train_df)))
            training_index = my_resample(ids, fam_id_training, replace=True,
                    n_samples=len(train_df), random_state=i_fold)
            x_train = fc_train[training_index]
            x_val = fc_validation
            y_train_orig = train_phenotype.iloc[training_index].to_numpy()
            y_val_orig = val_phenotype.to_numpy()

            if args.regress_out_covariates:
                # regress age and sex from the phenotypes
                reg = LinearRegression()
                cov_train, cov_val = organise_covariate(train_demographics.iloc[training_index],
                                                        val_demographics)

                cov_train = cov_train.drop(['family_ID', 'age_at_scan', 'sex_at_birth'], axis=1)
                cov_val = cov_val.drop(['family_ID', 'age_at_scan',  'sex_at_birth'], axis=1)
                reg.fit(cov_train, y_train_orig)
                y_train_predict = reg.predict(cov_train)
                y_val_predict = reg.predict(cov_val)
                train_mae = mean_absolute_error(y_train_orig, y_train_predict)
                val_mae = mean_absolute_error(y_val_orig, y_val_predict)
                print(f'MAE age & sex regression = train: {train_mae}, MAE val: {val_mae}')

                # remove residuals the phenotypes
                y_train_orig = y_train_orig - y_train_predict
                y_val_orig = y_val_orig - y_val_predict

                # save the current ys
                all_train_orig.append(y_train_orig)
                all_val_orig.append(y_val_orig)

                # Get the coefficients
                beta_coeffs[i_fold, :, :] = reg.coef_

            if config['prediction']['latent_targets_type'] == 'SVD':
                # perform the svd
                U, Vt, s, S, _, _, Us, usmean, usstdv, \
                    explained_variance_ratio, n_component = \
                    compute_svd(y_train_orig,
                            pov=1)
                # apply svd to validation
                Us_val = (y_val_orig @ Vt.T)
                y_train = Us
                y_val = Us_val
                targets = range(args.n_components)

            elif config['prediction']['latent_targets_type'] == 'reconstructed_SVD':
                # use only a specified number of components to reconstruct
                # targets
                U, Vt, s, S, _, _, Us, usmean, usstdv, \
                    explained_variance_ratio, n_component = \
                    compute_svd(y_train_orig,
                            pov=1)
                # apply svd to validation
                Us_val = (y_val_orig @ Vt[:args.n_components, :].T)
                y_train_recon = Us[:, :args.n_components] @ Vt[:args.n_components, :]
                y_val_recon = Us_val[:, :args.n_components] @ Vt[:args.n_components, :]
                y_train_recons.append(y_train_recon)
                y_val_recons.append(y_val_recon)
                y_train = y_train_orig
                y_val = y_val_orig
                targets = val_phenotype.columns

            else:
                targets = val_phenotype.columns
                y_train = y_train_orig
                y_val = y_val_orig

            # Do grid search
            best_mse_mean = 100000
            for alpha in parameters['alpha']:
                model = Ridge(alpha=alpha)
                # Fit either the reconstructed (with the specified amount of
                # needed components) or all the data available
                if config['prediction']['latent_targets_type'] in ['SVD', 'None']:
                    model.fit(x_train, y_train)
                elif config['prediction']['latent_targets_type'] == 'reconstructed_SVD':
                    model.fit(x_train, y_train_recon)
                predicted_val = model.predict(x_val)
                # compute mse for the predictions
                if config['prediction']['y_val'] == 'reconstruction':
                    mse_val = np.mean(np.square(y_val_recon - predicted_val), axis=0)
                elif config['prediction']['y_val'] == 'orig':
                    mse_val = np.mean(np.square(y_val - predicted_val), axis=0)
                if mse_val.mean() <= best_mse_mean:
                    best_mse_mean = mse_val.mean()
                    predicted_train = model.predict(x_train)
                    # Use the original phenotypes values to compute the scores
                    if config['prediction']['y_val'] == 'reconstruction':
                        comp_corr = corr(y_val_recon, predicted_val)
                        comp_r2 = compute_r2(y_val_recon, predicted_val)
                    elif config['prediction']['y_val'] == 'orig':
                        comp_corr = corr(y_val, predicted_val)
                        comp_r2 = compute_r2(y_val, predicted_val)

                    # append predicted values to the list to save
                    best_y_train = y_train
                    best_y_train_pred = predicted_train
                    best_y_val_true = y_val
                    best_y_val_pred = predicted_val

                    best_l2 = alpha
                    best_mse = mse_val
                    best_alpha = alpha

                    if config['prediction']['latent_targets_type'] == 'reconstructed_SVD':
                        y_train_predicted = predicted_train
                        y_val_predicted = predicted_val

            # compute the metrics per component
            metrics['Correlation'].append(comp_corr)
            metrics['R2'].append(comp_r2)
            metrics['MSE'].append(best_mse)
            metrics['best_l2'].append(best_l2)
            metrics['best_alpha'].append(best_alpha)

            # save the best metric per bootstrap
            save_y_train_true.append(best_y_train)
            save_y_train_predicted.append(best_y_train_pred)
            save_y_val_predicted.append(best_y_val_pred)
            save_y_val_true.append(best_y_val_true)


    
    if config['prediction']['latent_targets_type'] == 'reconstructed_SVD':
        # transform reconstruction into numpy array
        y_train_recons = np.array(y_train_recons)
        y_val_recons = np.array(y_val_recons)
        y_train_predicted = np.array(y_train_predicted)
        y_val_predicted = np.array(y_val_predicted)

        all_val_orig = np.array(all_val_orig)
        all_train_orig = np.array(all_train_orig)

        # Compute the predictions on the original space and save these metrics
        output_file = figs_dir / 'reconstructed_predictions.npz'
        np.savez(output_file, y_train_recons=y_train_recons,
                y_val_recons=y_val_recons, y_train_orig=all_train_orig,
                y_val_orig=all_val_orig, y_val_predicted=y_val_predicted,
                y_train_predicted=y_train_predicted)

    # save the predicted and true vales
    output_file = figs_dir / 'predictions.npz'

    save_y_train_true = np.array(save_y_train_true)
    save_y_train_predicted = np.array(save_y_train_predicted)
    save_y_val_true = np.array(save_y_val_true)
    save_y_val_predicted = np.array(save_y_val_predicted)

    np.savez(output_file, y_train_true=save_y_train_true,
            y_train_predicted=save_y_train_predicted,
            y_val_true=save_y_val_true,
            y_val_predicted=save_y_val_predicted)


    # save pickle
    with open(figs_dir / 'metrics.pickle', 'wb') as handle:
        pickle.dump(metrics,handle)
    sorted_targets, arg_sort = plot_scores(metrics, config['dataset'], targets,
            config['plot']['color_mat'],
                figs_dir, config['prediction']['latent_targets_type'])

    # save the results into a dataframe
    #results_df = pd.DataFrame(metrics)
    #results_df.to_csv(figs_dir / 'predictions_cv_val.csv')

    print("")
    print(f"Best regularisation:")
    value, counts = np.unique(metrics['best_l2'], return_counts=True)
    dic_counts = dict(zip(value, counts))
    print(dic_counts)

    if args.regress_out_covariates:
        # Plot image with the beta coefficients
        beta_mean = np.mean(beta_coeffs, axis=0)
        beta_std = np.std(beta_coeffs, axis=0)
        labels = (np.asarray(["{:.2f} \n ({:.2f})".format(mean, std) if abs(mean) > .1 else "" for mean, std in zip(beta_mean.flatten(), beta_std.flatten())])
                  ).reshape(beta_coeffs.shape[1:])
        
        # save the regression coefficients. This values will be used for plottin purpuses later.
        cov_regression = {}
        cov_regression['beta_coeffs'] = beta_coeffs
        cov_regression['labels'] = labels
        cov_regression['cov_train'] = cov_train.columns
        cov_regression['index'] = train_phenotype.columns
        with open(figs_dir / 'covariate_regression.pickle', 'wb') as handle:
            pickle.dump(cov_regression, handle)
        

    # save scorer information to plot it later
    output_file = figs_dir / 'metrics.npz'
    np.savez(output_file, metrics=metrics, config=config, targets=targets,
            sorted_targets=sorted_targets, arg_sort=arg_sort)

    #if args.regress_out_covariates:
    #    df_beta_coeffs.to_csv(figs_dir / 'beta_coefficients.csv')


if __name__ == '__main__':
    random_seed = 2
    args = parse_args()
    main(random_seed, args)
