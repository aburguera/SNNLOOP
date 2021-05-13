#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : LoopModelTest
# Description : Usage example of LoopModel
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# SET GPU
###############################################################################

# These must be the first lines to execute. Restart the kernel before.
# If you don't have GPU/CUDA or this does not work, just set it to False or
# (preferrably) remove these two lines.
from utils import set_gpu
set_gpu(True)

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
from automodel import AutoModel
from loopreader import LoopReader
from loopmodel import LoopModel
from loopgenerator import LoopGenerator
from utils import build_reader_basename,montage

###############################################################################
# PARAMETERS
###############################################################################

#------------------------------------------------------------------------------
# LOOPREADER PARAMETERS
#------------------------------------------------------------------------------

# A loop reader is, basically, in charge of pre-processing an UCAMGEN dataset
# to be used under a particular configuration. Please check loopreadertest.py
# for more information about loopreaders and the UCAMGEN repository for more
# information about the UCAMGEN format.
PATH_TRAIN_DATASET='../../DATA/LOOPDSTRSMALL'
PATH_TEST_DATASET='../../DATA/LOOPDSTRSMALL'
DATA_CONTINUOUS=False;
DATA_GROUPS=[[0,0.5],[0.5,2]]
DATA_SEPARATION=3
DATA_BALANCE=True
DATA_INVERT=False
DATA_MIRROR=True
DATA_CATEGORICAL=False
DATA_NORMALIZE_NO_CATEGORICAL=False

#------------------------------------------------------------------------------
# LOOPGENERATOR PARAMETERS
#------------------------------------------------------------------------------

# A loop generator inherits from Keras Sequences and can be used to train/
# test/... a Keras model. Please check loopgeneratortest.py for more
# information.
IMG_SIZE=(64,64)
BATCH_SIZE=10

#------------------------------------------------------------------------------
# LOOPMODEL PARAMETERS
#------------------------------------------------------------------------------

# Dense layers to use after the siamese branches to perform classification
# or regression. For example, [128,16] means two dense layers, the first
# one with 128 units and the second one with 2 units. These layers are
# followed by the output layer.
DENSE_LAYERS=[128,16]

# Used to decide the output layer size. Since in this example the output is
# not categorical (DATA_CATEGORICAL=False) we set CATEGORICAL_CLASSES to 0.
# This ensures a single unit in the output layer.
CATEGORICAL_CLASSES=0

# Loss function to use. Since we have two classes in this example (two groups
# defined in the loop reader), let's use binary crossentropy
LOSS_FUNCTION='binary_crossentropy'

# Metrics to measure during training, evaluation, testing, ... Must be a list.
# Since we have 2 classes in this example, let's measure the accuracy.
LOOP_METRICS=['accuracy']

# The loop model is not aimed at regression (since the reader output is
# not set to continuous)
DO_REGRESSION=False

# Let's re-train the feature extractor (encoder). This improves the results
# and, being already pre-trained, it begins with quite good weights and
# is relatively fast.
RETRAIN_FEATURE_EXTRACTOR=True

# Number of training epochs.
LOOP_EPOCHS=10

#------------------------------------------------------------------------------
# OTHER PARAMETERS
#------------------------------------------------------------------------------

# In this example, the train set is split into train and validation data.
# Thus, 2 readers will be created (train and test) but 3 generators used
# (train, validation and test). To accomplish this, the data from the train
# reader is split into train and validation using the following proportion.
VAL_SPLIT=0.2

# The feature extraction is performed by the encoder part of an autoencoder.
# It is extremely advisable for the feature extractor to be pre-trained
# even if it has to be re-trained again as part of the loop detector. In this
# example we will use a pre-trained feature extractor (encoder of an pre-
# trained autoencoder), and the following is the base name to load it.
# Please check the repository AUTOENCODER to learn about this aspect.
AUTOENCODER_BASENAME='../../DATA/MODELS/AUTOENCODER_128_128_32_EPOCHS100'

###############################################################################
# CREATE THE TRAIN AND VALIDATION LOOP GENERATORS
###############################################################################

# Create and save or load the train loop reader
theReader=LoopReader()
readerParams={'basePath':PATH_TRAIN_DATASET,
              'outContinuous':DATA_CONTINUOUS,
              'theGroups':DATA_GROUPS,
              'stepSeparation':DATA_SEPARATION,
              'doBalance':DATA_BALANCE,
              'doInvert':DATA_INVERT,
              'doMirror':DATA_MIRROR,
              'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
              'toCategorical':DATA_CATEGORICAL}
readerBaseName=build_reader_basename(**readerParams)
if theReader.is_saved(readerBaseName):
    theReader.load(readerBaseName)
else:
    theReader.create(**readerParams)
    theReader.save(readerBaseName)

# Split the reader loop specs into train and validation
numSpecs=theReader.loopSpecs.shape[1]
cutIndex=int(VAL_SPLIT*numSpecs)
trainSpecs=theReader.loopSpecs[:,cutIndex:]
valSpecs=theReader.loopSpecs[:,:cutIndex]

# Build train and validation data generators. Note that the train generator
# is randomized whilst the validation generator is not. Randomizing the train
# generator helps in reducing overfitting but randomizing the validation
# usually only leads to larger computation times. That is why validation
# generator is not randomized.
trainGenerator=LoopGenerator(trainSpecs,
                             theReader.get_image,
                             imgSize=IMG_SIZE,
                             batchSize=BATCH_SIZE,
                             doRandomize=True)
valGenerator=LoopGenerator(valSpecs,
                           theReader.get_image,
                           imgSize=IMG_SIZE,
                           batchSize=BATCH_SIZE,
                           doRandomize=False)

###############################################################################
# CREATE THE LOOP DETECTOR MODEL
###############################################################################

# Load the feature extractor. Actually, the whole autoencoder is loaded
# but only the encoder will be used as feature extractor. Please check the
# AUTOENCODER repository for more information.
autoModel=AutoModel()
autoModel.load(AUTOENCODER_BASENAME)

# Create the model
loopModel=LoopModel()
loopModel.create(featureExtractorModel=autoModel.encoderModel,
                 denseLayers=DENSE_LAYERS,
                 categoricalClasses=CATEGORICAL_CLASSES,
                 lossFunction=LOSS_FUNCTION,
                 theMetrics=LOOP_METRICS,
                 doRegression=DO_REGRESSION,
                 trainFeatureExtractor=RETRAIN_FEATURE_EXTRACTOR)

###############################################################################
# TRAIN THE MODEL
###############################################################################

print('[TRAINING]')
loopModel.fit(x=trainGenerator,validation_data=valGenerator,epochs=LOOP_EPOCHS)
print('[TRAINING READY]')

###############################################################################
# CREATE THE TEST LOOP GENERATOR
###############################################################################

# Create and save or load the test loop reader. Please not that since we are
# overwriting the train/validation reader, the train and validation loop
# generator will no longer be valid.
theReader=LoopReader()
readerParams={'basePath':PATH_TEST_DATASET,
              'outContinuous':DATA_CONTINUOUS,
              'theGroups':DATA_GROUPS,
              'stepSeparation':DATA_SEPARATION,
              'doBalance':DATA_BALANCE,
              'doInvert':DATA_INVERT,
              'doMirror':DATA_MIRROR,
              'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
              'toCategorical':DATA_CATEGORICAL}
readerBaseName=build_reader_basename(**readerParams)
if theReader.is_saved(readerBaseName):
    theReader.load(readerBaseName)
else:
    theReader.create(**readerParams)
    theReader.save(readerBaseName)

# Build test data generator. Note that, contrarily to the train reader,
# this one is not split. Also note that, similary to the validation generator,
# the test one is not randomized.
testGenerator=LoopGenerator(theReader.loopSpecs,
                            theReader.get_image,
                            imgSize=IMG_SIZE,
                            batchSize=BATCH_SIZE,
                            doRandomize=False)

###############################################################################
# EVALUATE AND SAVE THE MODEL
###############################################################################

print('[EVALUATING]')
loopModel.evaluate(testGenerator)
print('[EVALUATED]')

###############################################################################
# PRINT AND PLOT SOME RESULTS
###############################################################################

# Print the evaluation results. Note that they are saved. So, they can be also
# accessed when the model is loaded.

print('[EVALUATION RESULTS]')
loopModel.print_evaluation()

# Get the first batch
# The data in X contains two nparrays. Each array is a batch of images. Each
# images in one array relates to the corresponding image in the other array
# depending on the values of y, which state the class (in this case, class=0
# means low or no overlap and class 1 means large overlap)
[X,y]=testGenerator.__getitem__(0)

# Let's predict the first batch. The output is rounded since we have 2 classes.
# That is, we round the prediction to be 0 or 1.
thePredictions=np.round(loopModel.predict(X))

# To provide a "clear" representation, let's change the red channel of the
# images to their predicted class (so, class 1 will have be sort of red and
# the other ones sort of... non-red). Note that LoopGenerator can also work
# with grayscale images. So, this "approach" to visualize loop information
# would not work with grayscale images. Obviously.
for i in range(X[0].shape[0]):
    X[0][i,:,:,0]=thePredictions[i]
    X[1][i,:,:,0]=thePredictions[i]

# Now plot the modified images using montage
montage(X[0])
montage(X[1])