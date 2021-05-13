#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import os
from loopmodel import LoopModel
from loopreader import LoopReader
from loopgenerator import LoopGenerator
from automodel import AutoModel
from utils import build_model_basename,build_loopmodel_basename,build_reader_basename
from utils import multiclass_stats
from time import time

###############################################################################
# Loads a pre-trained feature extractor (encoder part of an autoencoder)
# and builds a dictionary that can be used (via **) as input of the loop
# model create method.
# Input  : pathModels - Path to the models folder
#          autoFilters - List of filters used to build the autoencoder.
#          autoEpochs - Autoencoder training epochs
#          denseLayers - List of integers. The number of items in the list
#                        is the number of dense layers to be appended to
#                        the shared part of the model before the
#                        output layer. The specific values are the number
#                        of units in each layer.
#          numGroups - Number of classes (for categorical encoding) or 0
#                      if no categorical encoding.
#          isCategorical - Use categorical encoding (True/False)
#          trainFeatureExtractor - True/False : Re-train the encoder?
# Output : loopModelParams - Dictionary ready to pass (with **) as input
#                            to the loopmodel.create method.
# Note   : The autoencoder is not trained here. It must be already trained
#          and saved. The filename must be AUTOENCODER_F0_F1_..._EPOCHSX
#          where F0, F1, ... are the values in autoFilters and X is the
#          value in autoEpochs.
###############################################################################
def get_loopmodel_parameters(pathModels,autoFilters,autoEpochs,denseLayers,
                             numGroups,isCategorical,trainFeatureExtractor):

    # Load the autoencoder model
    autoModel=AutoModel()
    autoBaseName=build_model_basename(pathModels,'AUTOENCODER',autoFilters,autoEpochs)
    autoModel.load(autoBaseName)

    # Build the loop model creation dictionary (to pass as create params)
    loopModelParams={'featureExtractorModel':autoModel.encoderModel,
                     'denseLayers':denseLayers,
                     'categoricalClasses':[0,numGroups][isCategorical],
                     'lossFunction':['mse','categorical_crossentropy'][isCategorical],
                     'theMetrics':['mae','categorical_accuracy'][isCategorical],
                     'doRegression':False,
                     'trainFeatureExtractor':trainFeatureExtractor}

    return loopModelParams

###############################################################################
# BUILDS, TRAINS, EVALUATES AND SAVES ONE LOOP DETECTOR MODEL
# Input  : pathTrain  - Path to the pre-processed training loop reader
#          pathTest   - Path to the pre-processed test loop reader
#          pathModels - Path to the folder to save the models and load autoen-
#                       coders.
#          splitVal   - Ratio of training data to be used as validation data.
#          autoFilters - List of filters used to create the autencoder.
#          autoEpochs - Autoencoder training epochs. Both autoFilters and
#                       autoEpochs are only used to build the filename to
#                       load the pre-trained autoencoder.
#          inputShape - Shape of the input images (rows, cols, channels)
#          denseLayers - List of neurons in the dense layers following the
#                       siamese feature extractors.
#          theGroups  - List of [min,max] overlapping ratios used to divide
#                       and/or balance the dataset.
#          isCategorical - Use categorical encoding.
#          trainEncoder - Train the feature extractors
#          loopEpochs - Number ot training epochs
#          labelsInverted - Use doInvert in the loop reader
# Note   : Please note that this function is just to help in training,
#          testing and saving multiple configurations. Aside of that, it is
#          not particularly useful. Please check main.py to see how this
#          function is used.
###############################################################################
def loopbuild(pathTrain,pathTest,pathModels,splitVal,autoFilters,autoEpochs,
              inputShape,denseLayers,theGroups,isCategorical,trainEncoder,
              loopEpochs,labelsInverted):
    # Build loop model base file name
    baseName=build_loopmodel_basename(pathModels=pathModels,
                                      autoFilters=autoFilters,
                                      autoEpochs=autoEpochs,
                                      denseLayers=denseLayers,
                                      theGroups=theGroups,
                                      isCategorical=isCategorical,
                                      trainFeatureExtractor=trainEncoder,
                                      loopEpochs=loopEpochs,
                                      labelsInverted=labelsInverted)

    print('* CURRENT CONFIGURATION IS %s'%baseName)

    # Create the loop detector model
    theModel=LoopModel()

    ###########################################################################
    # CHECK IF THE MODEL IS ALREADY TRAINED AND EVALUATED
    ###########################################################################

    # Check if already saved
    if theModel.is_saved(baseName):
        print('* CONFIGURATION ALREADY TRAINED AND TESTED.')
    else:

    ###########################################################################
    # PREPARE TRAINING AND VALIDATION DATA
    ###########################################################################

        print('* PREPARING TRAINING AND VALIDATION DATA')

        # Loop reader parameters dictionary. Defines the parameters so that
        # utils.build_reader_basename can build the proper reader base name to load
        # the reader.
        lReaderParams={'basePath':pathTrain,
                       'outContinuous':False,
                       'theGroups':theGroups,
                       'stepSeparation':3,
                       'doBalance':True,
                       'doInvert':labelsInverted,
                       'doMirror':True,
                       'normalizeNoCategorical':True,
                       'toCategorical':isCategorical}

        # Load the pre-processed reader. Note that the reader must be already
        # pre-processed and saved with the appropriate filename (see build_reader_
        # basename). Check readercreator.py.
        theReader=LoopReader()
        theReader.load(build_reader_basename(**lReaderParams))

        # Split the reader loop specs into train and validation
        numSpecs=theReader.loopSpecs.shape[1]
        cutIndex=int(splitVal*numSpecs)
        trainSpecs=theReader.loopSpecs[:,cutIndex:]
        valSpecs=theReader.loopSpecs[:,:cutIndex]

        # Build train and validation data generators
        trainGenerator=LoopGenerator(trainSpecs,theReader.get_image,imgSize=inputShape[:2],batchSize=150,doRandomize=True)
        valGenerator=LoopGenerator(valSpecs,theReader.get_image,imgSize=inputShape[:2],batchSize=150,doRandomize=False)

    ###########################################################################
    # CREATE AND TRAIN THE MODEL
    ###########################################################################

        print('* CREATING THE LOOP DETECTION MODEL')

        # Build the model parameters
        loopModelParams=get_loopmodel_parameters(pathModels=pathModels,
                                                 autoFilters=autoFilters,
                                                 autoEpochs=autoEpochs,
                                                 denseLayers=denseLayers,
                                                 numGroups=len(theGroups),
                                                 isCategorical=isCategorical,
                                                 trainFeatureExtractor=trainEncoder)

        # Create the model
        theModel.create(**loopModelParams)

        print('* TRAINING THE MODEL')

        # Train the model
        theModel.fit(x=trainGenerator,validation_data=valGenerator,epochs=loopEpochs)

    ###########################################################################
    # PREPARE TEST DATA
    ###########################################################################

        print('* PREPARING TEST DATA')

        # Prepare evaluation data
        lReaderParams['basePath']=pathTest
        theReader=LoopReader()
        theReader.load(build_reader_basename(**lReaderParams))

        # Prepare test data generator
        testGenerator=LoopGenerator(theReader.loopSpecs,theReader.get_image,imgSize=inputShape[:2],batchSize=150,doRandomize=False)

    ###########################################################################
    # EVALUATE AND SAVE THE MODEL
    ###########################################################################

        print('* EVALUATING THE MODEL')

        # Evaluate the model
        theModel.evaluate(testGenerator)

        print('* SAVING THE TRAINED AND TESTED MODEL')

        # Save it
        theModel.save(baseName)

###############################################################################
# EVALUATES AND COMPUTES DIFFERENT PER-CLASS STATS FOR THE SPECIFIED MODEL.
# Input  : pathTest   - Path to the pre-processed test loop reader
#          pathModels - Path to the folder to save the models and load autoen-
#                       coders.
#          pathResults - Path to the folder to save the evaluation results.
#          autoFilters - List of filters used to create the autencoder.
#          autoEpochs - Autoencoder training epochs. Both autoFilters and
#                       autoEpochs are only used to build the filename to
#                       load the pre-trained autoencoder.
#          inputShape - Shape of the input images (rows, cols, channels)
#          denseLayers - List of neurons in the dense layers following the
#                       siamese feature extractors. Used only to build
#                       filenames.
#          theGroups  - List of [min,max] overlapping ratios used to divide
#                       and/or balance the dataset. Used only to build the
#                       file names.
#          isCategorical - Use categorical encoding.
#          trainEncoder - Train the feature extractors
#          modelEpochs - Number ot training epochs
#          invertLabels - Use doInvert in the loop reader
# Note   : Please note that this function is just to help in obtaining some
#          per-class stats of a trained model. The model is NOT trained here,
#          neither the keras "evaluate" method is called here. The model must
#          be already saved in pathTest with the base name created by
#          build_loopmodel_basename.
#          Note that this function, aside of helping in obtaining stats for
#          different configurations, is not particularly useful. Check
#          computestats.py to see how it is used.
###############################################################################
def loopeval(pathTest,pathModels,pathResults,autoFilters,autoEpochs,
              inputShape,denseLayers,theGroups,isCategorical,trainEncoder,
              modelEpochs,invertLabels):

    ###########################################################################
    # CHECK IF EVALUATION ALREADY PERFORMED
    ###########################################################################

    # Build the base file name to save THE STATS(build_loopmodel_basename can
    # also be used to build this name).
    statsBaseName=build_loopmodel_basename(pathResults,autoFilters,autoEpochs,denseLayers,
                                      theGroups,isCategorical,trainEncoder,
                                      modelEpochs,invertLabels)

    if os.path.exists(statsBaseName+'_STATS.pkl'):
        print('* CONFIGURATION %s ALREADY TRAINED AND TESTED.'%(statsBaseName+'_STATS.pkl'))
        return

    print('* CURRENT CONFIGURATION IS %s'%statsBaseName)

    ###########################################################################
    # LOAD THE PRE-TRAINED MODEL
    ###########################################################################

    print('* LOADING PRE-TRAINED LOOP DETECTOR MODEL.')

    # Load the model
    baseName=build_loopmodel_basename(pathModels,autoFilters,autoEpochs,denseLayers,
                                      theGroups,isCategorical,trainEncoder,
                                      modelEpochs,invertLabels)

    theModel=LoopModel()
    theModel.load(baseName)

    ###########################################################################
    # LOAD THE TEST DATA (LOOP READERS MUST BE PRE-PROCESSED AND SAVED)
    ###########################################################################

    print('* LOADING AND PREPARING TEST DATA.')

    # Load the test data reader
    lReaderParams={'basePath':pathTest,
                   'outContinuous':False,
                   'theGroups':theGroups,
                   'stepSeparation':3,
                   'doBalance':True,
                   'doInvert':False,
                   'doMirror':True,
                   'normalizeNoCategorical':True,
                   'toCategorical':isCategorical}

    testReader=LoopReader()
    testReader.load(build_reader_basename(**lReaderParams))

    # Prepare the test generator
    testGenerator=LoopGenerator(testReader.loopSpecs,testReader.get_image,imgSize=inputShape[:2],batchSize=150,doRandomize=False)

    ###########################################################################
    # PREDICT USING THE MODEL
    ###########################################################################

    print('* PREDICTING THE TEST DATA.')

    # Predict loops using the model. We transpose them in order to have the
    # predictions in the same format that the ground truth.
    tStart=time()
    thePredictions=theModel.predict(testGenerator,verbose=1).transpose()

    # Compute the elapsed time per loop predicted.
    timePerLoop=(time()-tStart)/testReader.loopSpecs.shape[1]

    # Get the ground truth from the reader. Note that this is only a valid ground
    # truth if the generator was not used by any means before the predictions are
    # done AND the generator is NOT randomized.
    groundTruth=testReader.loopSpecs[2:,:]

    ###########################################################################
    # IF OUTPUT IS NOT CATEGORICAL, CONVERT TO CATEGORICAL
    ###########################################################################

    # If the current model output is not categorical, convert it to categorical
    # Note that the non-categorical stats (MAE and MSE) have already been
    # computed during build/train/evaluate
    if not isCategorical:

        print('* NON-CATEGORICAL DETECTED: CONVERTING TO CATEGORICAL.')

        numGroups=len(theGroups)
        # Prepare storage for the categorical version of predictions and ground
        # truth.
        newPredictions=np.zeros((numGroups,thePredictions.shape[1]))
        newGroundTruth=np.zeros((numGroups,thePredictions.shape[1]))

        # Make the ground truth integer-valued
        intGroundTruth=np.round(groundTruth*(numGroups-1))

        # Let's make them categorical
        for i in range(numGroups):
            curCenter=(i+1)/(numGroups+1)
            newPredictions[i,:]=1-np.abs(thePredictions-curCenter)
            newGroundTruth[i,:]=(intGroundTruth==i).astype('int')

        # Normalize them
        newPredictions=newPredictions/np.sum(newPredictions,axis=0)

        # Store them back into the original variables
        thePredictions=newPredictions
        groundTruth=newGroundTruth

    ###########################################################################
    # COMPUTE AND SAVE THE STATS
    ###########################################################################

    print('* COMPUTING THE STATS.')

    curStats,theAccuracy=multiclass_stats(groundTruth,thePredictions,timePerLoop,baseFName=statsBaseName)

    print('  + ACCURACY      : %.6f'%theAccuracy)
    print('  + TIME PER LOOP : %.6f'%timePerLoop)