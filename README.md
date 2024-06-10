# brain-phenotypes
Code for the paper "Reliability and predictability of phenotype information from functional connectivity in large imaging datasets".
The accompanying pre-registration can be found at: https://osf.io/7hr9w/

## Abstract
One of the central objectives of contemporary
neuroimaging research is to create predictive models that can disentangle the
connection between patterns of functional connectivity across the entire brain
and various behavioral traits. Previous studies have shown that models trained
to predict behavioral features from the individual's functional connectivity
have modest to poor performance. 
In this study, we trained models that predict observable individual traits (phenotypes) and their corresponding singular
value decomposition (SVD) representations -- herein referred to as \emph{latent
phenotypes} from resting state functional connectivity. For this task, we predicted phenotypes in two large neuroimaging datasets: the Human Connectome Project (HCP) and the Philadelphia
Neurodevelopmental Cohort (PNC). We illustrate the importance of regressing out confounds, which could significantly influence phenotype prediction. Our findings reveal that both phenotypes and their corresponding latent phenotypes yield similar predictive performance. Interestingly, only the first five latent phenotypes were reliably identified, and using just these reliable phenotypes for predicting phenotypes yielded a similar performance to using all latent phenotypes. This suggests that the predictable information is present in the first latent phenotypes, allowing the remainder to be filtered out without any harm in performance. This study sheds light on the intricate relationship
between functional connectivity and the predictability and reliability of
phenotypic information, with potential implications for enhancing predictive
modeling in the realm of neuroimaging research.

## Run the code
- The code was run and tested using Python 3.8.17. 
- To install the required packages, run `pip install -r requirements.txt`

## Citation
If you find this code useful, please cite the following paper:
```
@article{dafflon2024reliability,
  title={Reliability and predictability of phenotype information from functional connectivity in large imaging datasets},
  author={Dafflon, Jessica and Moraczewski, Dustin and Earl, Eric and Nielson, Dylan M and Loewinger, Gabriel and McClure, Patrick and Thomas, Adam G and Pereira, Francisco},
  journal={arXiv preprint arXiv:2405.00255},
  year={2024}
}
```