#!/bin/bash

# Path to where the code is. Change accordingly
project_dir="/home/jdafflon/brain-phenotypes"
phenotype_dir="/Datasets/BrainBlocks/DerivativeDatasets/HBN/brainblocks-0.1.0/phenotype"
figs_dir="${project_dir}/figs/HBN"
config_file="${project_dir}/config/HBN/predictions-hbn-latent_targets-zscore_column.yaml"
experiment_name="regress_covariates_predictions-hbn-latent_targets-zscore_column"
output_dir="${project_dir}/outputs/HBN/${experiment_name}"
fc_dir="${project_dir}/data/HBN/brainblocks-0.1.0"
predictions_dir="/Datasets/BrainBlocks/DerivativeDatasets/HBN/brainblocks-0.1.0/prediction_experiments/predict-PHENOTYPE+age_at_scan+family_ID+sex_at_birth-from_task-rest_atlas-Schaefer2018400Parcels17Networksorder_space-MNI_desc-clean_bold_conndata-network_connectivity"
n_components=29

# run script
python ${project_dir}/src/fc_prediction.py \
	--config_file=${config_file} \
	--figs_dir=${output_dir} \
	--output_dir=${output_dir} \
	--project_dir=${project_dir} \
	--fc_dir=${fc_dir} \
	--predictions_dir=${predictions_dir} \
	--phenotype_dir=${phenotype_dir} \
	--experiment_name=${experiment_name} \
    --regress_out_covariates \
    --n_components=${n_components}

