#!/bin/bash

# conda activate bblocks-phenotype

# Path to where the code is. Change accordingly
project_dir="/Users/jdafflon/Code/brain-phenotypes"
#project_dir="/home/jdafflon/bblocks-phenotypes"
experiment_name_hcp="regress_covariates_predictions-hcp-latent_targets_svd-zscore_y_val_orig_column-083_component"
experiment_name_pnc="regress_covariates_predictions-pnc-latent_targets_svd-zscore_y_val_orig_column-039_component"
experiment_name_hbn="regress_covariates_predictions-hbn-latent_targets_svd-zscore_y_val_orig_column-029_component"
figs_dir_hcp="${project_dir}/outputs/HCP/${experiment_name_hcp}"
figs_dir_pnc="${project_dir}/outputs/PNC/${experiment_name_pnc}"
figs_dir_hbn="${project_dir}/outputs/HBN/${experiment_name_hbn}"

# run script
python ${project_dir}/src/plot_covariance_weights.py \
	--figs_dir_hcp=${figs_dir_hcp} \
	--figs_dir_pnc=${figs_dir_pnc} \
	--figs_dir_hbn=${figs_dir_hbn} \
