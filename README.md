Instructions to reproduce the experiments in Predicting Rice Phenotypes with Meta and Multi-target Learning. 

Steps.
1. First download the genotype (x) and phenotype (ys) files from http://dx.doi.org/10.17632/wxm9p2fhjt.1.
2. Copy these files to input/data/processed/
3. Run the R scripts in the order in which they have been numbered.

File names and corresponding experiments.
0_base.R - Base case single target experiments.
1_fa.R - Framework A experiments.
2_fb.R - Framework B experiments.
3_mtrs.R - Multi-target regressor stacking.
4_erc.R - Ensemble of regressor chains.
5_erc.R - Ensemble of regressor chains corrected.

6-8 - Unweighted averaging of the predictions made by the base learners in 3-5 above.
