'''
Author: Revant Gupta
        SciLife Summer Fellow 2017
        Lukas Käll Lab

Description:

* MS/MS analysis is routinely performed for the identification and quantification of proteins/peptides in biological samples.
* The ionisation efficiency of peptides depends partly on experitmental conditions and on the peptide itself.
* The quanitifcation of a peptide will in part depend on how efficiently it was ionised.
* In this code we use preprocessed MS/MS quanitifcation data that uses a single value for each peptide which is derived from the quanitification in multiple runs.
* For peptide with equal sample concentration (i.e. belonging to the same protein) we assume the higher value implies higher ionisation efficiency.
* The code generates a trained model that can be used to create a ranked list of peptides by ionisation efficiency by sorting the binary comparisons.
* The code can additionally output a trained autoencoder that converts peptide representations into 240 feature encodings.
* The code can optionally be run using pre-trained models (autoencoder and classifier).

'''
# coding: utf-8

# General Libraries

import pandas as pd
import numpy as np
import random as ra
import re
import os
import time
import itertools
import sys
sys.path.insert(0, '../utils')
from sklearn.utils import shuffle

# Machine Learning

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, Dropout, Reshape, Input, concatenate
from keras.utils import Sequence
from keras import optimizers as opt
from keras import regularizers as reg
from keras import initializers as init
from keras.models import load_model

# Custom

from threadsafe import threadsafe_generator

# Arguments

import argparse

apars = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

apars.add_argument('-train', '--train_data', default='../sample_data/train.tsv',
                       help='''Training data set''')

apars.add_argument('-val', '--validation_data', default=None,
                       help='''Validation data set (Optional)''')

apars.add_argument('-out', '--trained_classifier', default = '../models/classifier.h5',
                        help = ''' Save classifier weights after training''')

apars.add_argument('-out_e', '--trained_encoder', default = '../models/encoder.h5',
                        help = ''' Save encoder weights after training''')

apars.add_argument('-epochs_c', '--classifier_epochs', default = 2000,
                        help = ''' Number of training epochs for the classifier''')

apars.add_argument('-epochs_e', '--encoder_epochs', default = 25,
                        help = ''' Number of training epochs for the autoencoder''')

apars.add_argument('-batch', '--batch_size', default = 500,
                        help = ''' Batch size. Must be equal to or less than the number of data points in either set.''')

apars.add_argument('-workers', default = 4,
                        help = ''' Number of processes to spin up while training and testing a model.''')

apars.add_argument('--model', default = None,
                        help = ''' Load existing classifier model for further training. (Optional)''')

apars.add_argument('--encoder', default = None,
                        help = ''' Load existing autoencoder model for encoding data. (Optional)''')

apars.add_argument('--log', default = None,
                        help = ''' Training logs. (Optional)''')

apars.add_argument('--verbose', default = 1,
                        help = ''' Verbosity (0,1,2). (Optional)''')
                        

args = apars.parse_args()

epochs_c = int(args.classifier_epochs) # Select
epochs_e = int(args.encoder_epochs) # Select
batch_size = int(args.batch_size) # Minimum number of data points required 

  #--------------------------------------------------------------------------------------------------------------------------------------------#  
   # Data processing scripts for raw MS data can be found in the processing folder of the repository. This script only allows a specific format.
   #-------------------------------------------------------------------------------------------------------------------------------------------#

# List of amino acids in peptides and replacing M[16] as X

aa_alphabet= ['K', 'E', 'G', 'V', 'P', 'D', 'M', 'Y', 'T', 'A', 'R', 'F', 'N', 'Q', 'L', 'S', 'I','C', 'H', 'W', 'X']

# Generate training logs

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(str(args.log), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

if args.log != None:    
    sys.stdout = Logger()

###################################################################################################################################################################################
###################################################################################################################################################################################

# Function to generate compared peptide pairs and training targets.

def data_preprocessing(inp):

    data = pd.read_csv(inp, sep = '\t')

    cols = ['peptide', 'protein','value']

    data = data[cols]

    # Creating protein wise arrays for comparison

    prot_info = []
    io_eff = []

    protein_list = list(set(data['protein']))

    for i in range(len(protein_list)):

        temp = data[data['protein'] == protein_list[i]]
        temp.reset_index(drop = True, inplace = True)

        info_array = []
        io_array = []    
        
        for t, info in enumerate(temp['peptide']):

            info_array.append(info)
            io_array.append(float(temp['value'][t]))

        prot_info.append(info_array)
        io_eff.append(io_array)

    del data # Free up space by removing data that is no longer needed

    # Generating comparisons for training input

    comparisons = []
    reverse = []

    drop_count = 0

    for j, prot in enumerate(prot_info):
  
        comparison_number = len(prot)

        if comparison_number == 1:
            drop_count += 1
            continue

        # a = re.randint(0, comparison_number -1 )
        # b = re.randint(0, comparison_number -1 ) # Random number of pairs equal to the number of peptides
        
        for a in range(comparison_number):

            for b in range(a+1,comparison_number): # All possible pairs

                assert a!=b
                    
                if float(io_eff[j][a])>float(io_eff[j][b]):

                    comparisons.append([prot[a], prot[b]])
                    reverse.append([prot[b], prot[a]])
   
                if float(io_eff[j][a])<float(io_eff[j][b]):

                    comparisons.append([prot[b], prot[a]])
                    reverse.append([prot[a], prot[b]])

    del prot_info # Cleanup
    del io_eff # Cleanup
        
    # Generating the information set 

    target_len = len(comparisons)

    comparisons.extend(reverse)           
    
    targets = [1]*(int(target_len))+[0]*(int(target_len))
    
    if drop_count > 0:    
        
        print("%d proteins were dropped as they only had one peptide."%(drop_count))

    assert len(comparisons) == len(targets)

    return comparisons, targets

###################################################################################################################################################################################
###################################################################################################################################################################################

# Function to build peptide representations

def pep_rep_builder(comparison):

    # Function provides peptide representations for both peptides within one comparison

    rep_list = []

    for info in comparison:

        peptide = info

        rep = np.ones((1,120, len(aa_alphabet)), dtype=np.float)*0.0000000001
        
        for j in range(len(peptide)):

            rep[0,60-len(peptide)+j, aa_alphabet.index(peptide[j])] = 1 # Peptide representation
            rep[0,119-(60-len(peptide)+j), aa_alphabet.index(peptide[j])] = 1 # Mirrored representation
            
            rep_list.append(rep)

    return rep_list


# Infinite generator to yield inputs and targets (required later for autoencoder fitting)

@threadsafe_generator
def auto_input_gen(inputs, batch_size):

    # Shuffle data

    indices = shuffle(list(range(len(inputs))), random_state= 0)

    steps = list(range(int(len(inputs)/batch_size)+1))

    for j in itertools.cycle(steps):

        input_array_a = []
        input_array_b = []
        target_array = []
    
        for i in range(j,j+batch_size):

            input_array_a.append(np.array(pep_rep_builder(inputs[indices[i]])[0]))
            input_array_b.append(np.array(pep_rep_builder(inputs[indices[i]])[1]))
            target_array.append(np.concatenate((np.array(pep_rep_builder(inputs[indices[i]])[0]), np.array(pep_rep_builder(inputs[indices[i]])[1])), axis = -2))
                
        yield ([np.array(input_array_a), np.array(input_array_b)], np.array(target_array))

# Infinite generator to yield inputs (required later for encoder output)

@threadsafe_generator
def auto_pred_gen(inputs, batch_size):

    # Shuffle data

    indices = list(range(len(inputs)))

    steps = list(range(int(len(inputs)/batch_size)+1))

    for j in itertools.cycle(steps):

        input_array_a = []
        input_array_b = []
    
        for i in range(j,j+batch_size):

            input_array_a.append(np.array(pep_rep_builder(inputs[indices[i]])[0]))
            input_array_b.append(np.array(pep_rep_builder(inputs[indices[i]])[1]))
                
        yield ([np.array(input_array_a), np.array(input_array_b)])


###################################################################################################################################################################################
###################################################################################################################################################################################

# The autoencoer to generate 120 element encodings for the classifier.

def Auto_CNN():

    # Optimizer is Adam RMSprop with Nesterv Momentum

    nadam = opt.Nadam(lr=0.015, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    # Optimizer is SGD with Nesterov Momemntum

    #sgd = opt.SGD(lr = 0.15, momentum = 0.9, nesterov = True)    

    # Convolutions of both kinds on each peptide representation

    # First peptide branch

    model_a_inp = Input(shape = (1,120,len(aa_alphabet)))
    model_a = Conv2D(800,(6,len(aa_alphabet)), strides=(1,1), padding='valid', 
                        data_format='channels_first', activation='relu', use_bias=False, kernel_initializer='glorot_uniform')(model_a_inp)
    model_a = Flatten()(model_a)

    # Second peptide branch

    model_b_inp = Input(shape = (1,120,len(aa_alphabet)))
    model_b = Conv2D(800,(6,len(aa_alphabet)), strides=(1,1), padding='valid', 
                        data_format='channels_first', activation='relu', use_bias=False, kernel_initializer='glorot_uniform')(model_b_inp)
    model_b = Flatten()(model_b)

    # Final model

    cnv = concatenate([model_a, 
                       model_b, 
                     ])

    # Auto-encoder

    encoder = Dense(240, activation='relu', use_bias=True, bias_initializer = 'random_normal', kernel_initializer='glorot_uniform')(cnv) # Encoder

    x = Dense(5040, activation='relu', use_bias=True, bias_initializer = 'random_normal', kernel_initializer='glorot_uniform')(encoder)

    decoder = Reshape((1,240,len(aa_alphabet)))(x) # Decoder

    model = Model(inputs = [model_a_inp, model_b_inp], outputs = decoder)

    model.compile(optimizer= nadam, loss='mse', metrics = ['accuracy'])

    encoder = Model(inputs = [model_a_inp, model_b_inp], outputs = encoder)

    return model, encoder

# Binary classifier is a deep neural network that is trained on the encoded peptide pairs.

def clas_NN():

    # Optimizer is Adam RMSprop with Nesterv Momentum

    #nadam = opt.Nadam(lr=0.015, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    # Optimizer is SGD with Nesterov Momemntum

    sgd = opt.SGD(lr = 0.0015, momentum = 0.9, nesterov = True)    

    model = Sequential()

    model.add(Dense(240, activation='relu', use_bias=True, bias_initializer = 'random_normal', kernel_initializer='glorot_uniform', input_shape = (240,)))

    for i in range(8):

        model.add(Dense(240, activation='relu', use_bias=True, bias_initializer = 'random_normal', kernel_initializer='glorot_uniform'))
        model.add(Dropout(0.5))

    model.add(Dense(240, activation='relu', use_bias=True, bias_initializer = 'zeros', kernel_initializer='glorot_uniform'))
    model.add(Dense(1, activation='sigmoid', use_bias=True, bias_initializer = 'zeros', kernel_initializer='glorot_uniform'))

    model.compile(optimizer= sgd, loss='binary_crossentropy', metrics = ['binary_accuracy'])

    return model

###################################################################################################################################################################################
###################################################################################################################################################################################

def main(args):

    start = time.time()

    print('\n')
    print('Processing data...')
    print('\n')

    train_inp, train_tar = data_preprocessing(args.train_data)

    if args.validation_data != None:

        val_inp, val_tar = data_preprocessing(str(args.validation_data))

    print('\n')
    print('Running autoencoder...')
    print('\n')

    if args.encoder != None:

        encoder = load_model(str(args.encoder))

    else:

        model, encoder = Auto_CNN()

        if args.validation_data != None:

            #model.fit_generator(auto_input_gen(train_inp, batch_size), (int(len(train_inp)/batch_size)+1), epochs=epochs_e, verbose = args.verbose, 
            #                validation_data=auto_input_gen(val_inp, batch_size), validation_steps=(int(len(val_inp)/batch_size)+1),               
            #                use_multiprocessing=True, max_queue_size=1, workers=args.workers)

            # Including validation data in the training data for the autoencoder (validation is relevant to the classifier)

            enc_inp = train_inp + val_inp

            ahist = model.fit_generator(auto_input_gen(enc_inp, batch_size), (int(len(enc_inp)/batch_size)+1), epochs=epochs_e, verbose = args.verbose,                                      
                            use_multiprocessing=True, max_queue_size=1, workers=args.workers)

        else:

            ahist = model.fit_generator(auto_input_gen(train_inp, batch_size), (int(len(train_inp)/batch_size)+1), epochs=epochs_e, verbose = args.verbose, 
                            use_multiprocessing=True, max_queue_size=1, workers=args.workers)

        encoder.save(args.trained_encoder)

        print('\n')
        print('Final autoencoder accuracy: {acc} %'.format(acc = np.round(float(ahist.history['acc'][-1])*100.0, 2)))
        print('\n')

    print('\n')
    print('Generating encoded data...')
    print('\n')

    enc_data = pd.DataFrame()

    train_X = encoder.predict_generator(auto_pred_gen(train_inp, batch_size), (int(len(train_inp)/batch_size)+1), 
                                        use_multiprocessing=True, max_queue_size=1, workers=args.workers)
    train_Y = train_tar+train_tar[:int(len(train_X)-len(train_tar))]

    if args.validation_data != None:
    
        val_X = encoder.predict_generator(auto_pred_gen(val_inp, batch_size), (int(len(val_inp)/batch_size)+1), 
                                        use_multiprocessing=True, max_queue_size=1, workers=args.workers)
        val_Y = val_tar+val_tar[:int(len(val_X)-len(val_tar))]


    print('\n')
    print('Training classifier...')
    print('\n')

    if args.model != None:

        classiier = load_model(str(args.model)) # Train an existing model

    else:

        classifier = clas_NN()


    if args.validation_data != None:

        chist = classifier.fit(train_X, train_Y, batch_size = 2000, epochs = epochs_c, verbose = args.verbose, validation_data = (val_X, val_Y))

        print('\n')
        print('Final classifier training and validation accuracy: {acc} %, {val_acc} %'.format(acc = np.round(float(chist.history['binary_accuracy'][-1])*100.0, 2), 
                                                                                      val_acc = np.round(float(chist.history['val_binary_accuracy'][-1])*100.0, 2)))
        print('\n')
    else:

        chist = classifier.fit(train_X, train_Y, batch_size = 2000, epochs = epochs_c, verbose = args.verbose)

        print('\n')
        print('Final classifier training accuracy: {acc} %'.format(acc = np.round(float(chist.history['binary_accuracy'][-1]*100.0, 2))))
        print('\n')


    classifier.save(args.trained_classifier)

    print('\n')

    print('The script took %.2f seconds to run'%(time.time()-start))

###################################################################################################################################################################################
###################################################################################################################################################################################

if __name__ == '__main__':

    __spec__ = None

    main(args)