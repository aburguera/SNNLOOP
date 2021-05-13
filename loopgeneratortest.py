#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : LoopGeneratorTest
# Description : Usage example of LoopGenerator
#               LoopGenerators inherit from the Keras Sequences so that
#               they can be used to train, test, validate and predict using
#               Keras models.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from utils import montage
from utils import build_reader_basename
from loopreader import LoopReader
from loopgenerator import LoopGenerator

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
PATH_DATASET='../../DATA/LOOPDSTRSMALL'
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

# The image size must coincide with the Neural Network input image sizes.
# The LoopGenerator resizes the images in the dataset to this size on-the-fly
# if they are not of this size. Thus, to speed up the process it is better
# to already have a LoopReader (and, thus, an UCAMGEN dataset) whose images
# are of this size.
IMG_SIZE=(64,64)

# The size of the batches used to train/test/validate/... the Neural Network.
BATCH_SIZE=10

# If True, the DataGenerator will shuffle the loops when being created and
# after each epoch. It is advisable to set it to True to train and to False
# to validate and test.
DATA_RANDOMIZE=True

###############################################################################
# PREPARE THE LOOP READER
###############################################################################

# Prepare the loop reader parameters into a convenience dictionary.
readerParams={'basePath':PATH_DATASET,
              'outContinuous':DATA_CONTINUOUS,
              'theGroups':DATA_GROUPS,
              'stepSeparation':DATA_SEPARATION,
              'doBalance':DATA_BALANCE,
              'doInvert':DATA_INVERT,
              'doMirror':DATA_MIRROR,
              'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
              'toCategorical':DATA_CATEGORICAL}

# Build the reader base name using the convience function
readerBaseName=build_reader_basename(**readerParams)

# Check if the reader is already saved. If so, load it. Otherwise, build it
# and save it.
theReader=LoopReader()
if theReader.is_saved(readerBaseName):
    theReader.load(readerBaseName)
else:
    theReader.create(**readerParams)
    theReader.save(readerBaseName)

###############################################################################
# CREATE THE LOOP GENERATOR
###############################################################################

theGenerator=LoopGenerator(loopSpecs=theReader.loopSpecs,
                           imgGetter=theReader.get_image,
                           imgSize=IMG_SIZE,
                           batchSize=BATCH_SIZE,
                           doRandomize=DATA_RANDOMIZE)

###############################################################################
# PRINT AND PLOT SOME INFO FROM THE LOOP GENERATOR
###############################################################################

# Print number of loops and batches
print('* NUMBER OF LOOPS: %d'%theGenerator.numLoops)
print('* NUMBER OF BATCHES: %d'%theGenerator.__len__())

# Get the first batch
[X,y]=theGenerator.__getitem__(0)

# The data in X contains two nparrays. Each array is a batch of images. Each
# images in one array relates to the corresponding image in the other array
# depending on the values of y, which state the class (in this case, class=0
# means low or no overlap and class 1 means large overlap)

# To provide a "clear" representation, let's change the red channel of the
# images to the class they belong (so, class 1 will have be sort of red and
# the other ones sort of... non-red). Note that LoopGenerator can also work
# with grayscale images. So, this "approach" to visualize loop information
# would not work with grayscale images. Obviously.
for i in range(X[0].shape[0]):
    X[0][i,:,:,0]=y[i]
    X[1][i,:,:,0]=y[i]

# Now plot the modified images using montage
montage(X[0])
montage(X[1])