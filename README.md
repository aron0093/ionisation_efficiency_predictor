# ionisation_efficiency_predictor
MS/MS analysis is routinely performed for the identification and quantification of proteins/peptides in biological samples. The ionisation efficiency of peptides dependas partly on experitmental conditions and on the peptide itself. The quantification values for each peptide then will be a combination of sample concentration of peptide and its ionisation efficiency.

## binary_comparison_trainer.py
Binary classification of peptide pairs by taking the ratio of their abundance derived values. Peptides in a pair should have the same sample concentration. The output can be used to create a ranked list of peptides by a sorting algorithm.

# Requirements:
* Anaconda python3
* keras 2.0.5
* theano/tensorflow
* sklearn

