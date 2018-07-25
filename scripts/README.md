## binary_comparison_trainer_vx.0.py

Given the size of, the dataset (n x n comparisons for all n peptides in a protein) and the array representation (120 x 21) the input to the autoencoder is fed in batches that are supplied by a generator. 

Version 1.0 of this script does not support multi-processing for this task thus may be significantly slower depending on the size of the data and computational resources available. This version must be used on windows which does not support multi-processing.

For more details check,

[https://github.com/keras-team/keras/pull/8662](https://github.com/keras-team/keras/pull/8662)


