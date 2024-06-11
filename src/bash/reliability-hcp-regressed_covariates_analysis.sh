#!/bin/bash

# Path to where the code is. Change accordingly
#project_dir="/home/jdafflon/bblocks-phenotypes"
#data_dir="/Datasets/BrainBlocks/DerivativeDatasets"
project_dir="/Users/jdafflon/Code/brain-phenotypes"
data_dir="/Users/jdafflon/Code/brain-phenotypes/data"
config_file="${project_dir}/config/HCP/predictions-hcp-latent_targets_svd-zscore_column.yaml"
figs_dir="${project_dir}/outputs/HCP/reliability"

# run script
python ${project_dir}/src/latent_variables_analysis.py \
	--config_file=${config_file} \
	--figs_dir=${figs_dir} \
	--project_dir=${data_dir} \
	--regress_out_covariates \
	--dataset=HCP
