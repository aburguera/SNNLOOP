#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : preparestats
# Description : Evaluates different loop detector configurations and
#               computes different metrics.
#               Please note that this approach:
#               * The loop readers are assumed to be already created and
#                 saved using the utils.build_reader_basename file names.
#               * Models are saved using the utils.build_loopmodel_basename
#                 nomenclature.
#               * The stats are saved with a _STATS suffix using a base name
#                 constructed using utils.build_loopmodel_basename.
#               * Check utils.multiclass_stats to see which stats are
#                 computed and how they are saved.
#               * Definitely: almost everything used by this function needs
#                 refactoring. Should it be refactored? Yes. Will I refactor
#                 it? Probably not. These "loop through different configs"
#                 scripts are mainly for internal use, so I'll probably not
#                 improve them.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 7-April-2021 - Creation
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

from looptools import loopeval

###############################################################################
# PARAMETERS
###############################################################################

#------------------------------------------------------------------------------
# AUTOENCODER (FEATURE EXTRACTOR) RELATED PARAMETERS
#------------------------------------------------------------------------------

# List of autoencoder filters to use. They should be previously trained,
# evaluated and saved. Please check AUTOENCODER repository (main.py or
# autotools.py) to learn how to do it.
AUTO_FILTERS=[[128,128,16],     # Size of latent space = 1024
              [128,4],          # Size of latent space = 1024
              [128,8],          # Size of latent space = 2048
              [128,128,32],     # Size of latent space = 2048
              [128,16],         # Size of latent space = 4096
              [128,128,64],     # Size of latent space = 4096
              [128,32],         # Size of latent space = 8192
              [128,128,128]]    # Size of latent space = 8192

# Number of training epochs for each of the previous autoencoder configs.
# Used only to build the filename to load them.
AUTO_EPOCHS=100

#------------------------------------------------------------------------------
# LOOP READER (DATASET PRE-PROCESSOR) RELATED PARAMETERS
#------------------------------------------------------------------------------

# The LREADER_* params are related to loop readers. These parameters are
# mostly used to build the file names and load the pre-processed loopreaders.
# Check readercreator.py to see how to build the pre-processed loopreaders and
# save them to disk. Some of these parameters might also be used later for
# purposes other than loop readers.

# Groups in which loops are divided depending on the overlap ratio
LREADER_GROUPS_LIST=[[[0],[0.5,2]],
                     [[0,0.5],[0.5,2]],
                     [[0,0.33],[0.33,0.66],[0.66,2]]]

# Combination of categorical encoding and non-categorical.
# False means that the class is stored as a number between 0 (full overlap) to
# 1 (no overlap).Please note that this is inverted with respect to the encoding
# in the raw datasets (doInvert=True must be used to create the readers) to
# facilitate the use of some loss functions.
# True means that classes are categorically encoded. The number of classes is
# the length of the corresponding groups list.
LREADER_CATEGORICAL_ENCODING=[True,False]

#------------------------------------------------------------------------------
# LOOP DETECTOR MODEL RELATED PARAMETERS
#------------------------------------------------------------------------------

# Model input shape.
MODEL_INPUT_SHAPE=(64,64,3)

# Model dense layers configurations. Each sub-list is one configuration.
# Each number in the sub-list denotes the number of units in one dense
# layer between the siamese feature extractors and the output layer.
MODEL_LAYERS=[[32,16],
              [128,64,32,16]]

# Train feature extractor
MODEL_TRAIN_ENCODER=[False,True]

# Number of epochs to train
MODEL_EPOCHS=10

#------------------------------------------------------------------------------
# PATHS
#------------------------------------------------------------------------------

# Paths
PATH_TRAIN='../../DATA/LOOPDSTR'    # Train images path
PATH_TEST='../../DATA/LOOPDSTS'     # Test images path
PATH_MODELS='../../DATA/MODELS/'    # Model storage path
PATH_RESULTS='../../DATA/RESULTS/'  # Results storage path

###############################################################################
# LOOP THROUGH ALL CONFIGURATIONS
###############################################################################

numConfigs=len(AUTO_FILTERS)*len(MODEL_LAYERS)*len(LREADER_GROUPS_LIST)*len(LREADER_CATEGORICAL_ENCODING)*len(MODEL_TRAIN_ENCODER)
curConfig=1

for autoFilters in AUTO_FILTERS:
    for denseLayers in MODEL_LAYERS:
        for theGroups in LREADER_GROUPS_LIST:
            for isCategorical in LREADER_CATEGORICAL_ENCODING:
                for trainEncoder in MODEL_TRAIN_ENCODER:

                    print('[[ EVALUATING %d of %d ]]'%(curConfig,numConfigs))
                    curConfig+=1

                    # Compute and save the stats
                    loopeval(pathTest=PATH_TEST,
                             pathModels=PATH_MODELS,
                             pathResults=PATH_RESULTS,
                             autoFilters=autoFilters,
                             autoEpochs=AUTO_EPOCHS,
                             inputShape=MODEL_INPUT_SHAPE,
                             denseLayers=denseLayers,
                             theGroups=theGroups,
                             isCategorical=isCategorical,
                             trainEncoder=trainEncoder,
                             modelEpochs=MODEL_EPOCHS,
                             invertLabels=False)

                    print('[[ PROCESSED ]]')