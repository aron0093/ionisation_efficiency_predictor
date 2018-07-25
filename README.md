# ionisation_efficiency_predictor

Peptide abundances reported by MS/MS experiments are greatly influenced by their ionisation efficiency, which itself depends upon intrinsic and experimental factors The abundance values for each peptide depend on the sample concentration of peptide and it's ionisation efficiency.

## binary_comparison_trainer.py

A convolutional layer based autoencoder coupled with a deep neural network predicts the order of ionisation efficiencies for peptides, from their sequence and, therefore, can assist the quantification of peptides, thus improving its accuracy. The autoencoder is used to generate vector representations of peptides that compresses the sequence information and also acts like a regulariser. The deep learning classifier compares peptide pairs to predict the order of the ionisation efficiency.

Binary classification of peptide pairs (labels) is done by taking the ratio of their abundance values. Peptides in a pair should have the same sample concentration if they belong to the same protein or if the experiment is set up to be so. 

The trained classifier can be used to create a ranked list of peptides (ionisation efficiency) by a sorting algorithm.

# Requirements:

* Anaconda python3
* keras 2.2.0
* tensorflow 1.8.0
* scikit-learn

## processing

* Scripts to process peptide abundance data.

**Notes:**

* The windows version of this script (v1.0) does not support multi-processing and thus may be significantly slower depending on the size of the data and computational resources available. For more details check, [https://github.com/keras-team/keras/pull/8662](https://github.com/keras-team/keras/pull/8662)

* The sample data is useful for understanding the operation of the scripts. It does not generate a very useful model. The model requires large amount of data from diverse experimental conditions to achieve good training and validation accuracy!