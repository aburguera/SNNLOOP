#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : utils
# Description : Several utilities with no specific order.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 21-March-2021 - Creation
#               01-April-2021 - Added set_gpu, contrastive_loss and
#                               build_model_basename.
#               13-April-2021 - Stats functions added/completed.
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import sys
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage import img_as_float
import tensorflow.keras.backend as K
import tensorflow as tf
import os
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from pickle import dump

###############################################################################
# MONTAGE
# Creates an image containing all the images in the input. All the images
# in the input must be the same size.
# Input : theBatch : NumPy array of size (n,r,c,chan)
#                    The images to show. The first dimension (n) is the number
#                    of images. The second and third dimension (r,c) are the
#                    image sizes (rows and columns). The last dimension is the
#                    number of channels. In grayscale images it can either be
#                    1 or do not exist.
#         doPlot   : Boolean, optional. Plot the resulting image. The default
#                    is True.
#         savePath : String, optional. File name of the image to save or None
#                    to not save. The default is None.
# Output :outImage : NumPy matrix. The resulting image.
###############################################################################
def montage(theBatch,doPlot=True,savePath=None):
    # Determine the input dimensions
    if len(theBatch.shape)==4:
        (nItems,nRowsImage,nColsImage,nChan)=theBatch.shape
    elif len(theBatch.shape)==3:
        (nItems,nRowsImage,nColsImage)=theBatch.shape
        nChan=1
    else:
        sys.exit('Input shape must be (n,r,c,ch) or (n,r,c) where n is the number of images, r and c are the number of rows and columns and ch is the number of channels. This last dimension can be ommited for grayscale images.')

    # Determine the output dimensions
    nCols=int(np.ceil(np.sqrt(nItems)))
    nRows=int(np.ceil(nItems/nCols))

    # Create the output matrix
    outImage=np.zeros((nRows*nRowsImage,nCols*nColsImage,nChan)).astype('float')

    # Convert the batch to float images
    theBatch=img_as_float(theBatch)

    # Build the output matrix
    batchIndex=0
    for r in range(nRows):
        if batchIndex>=nItems:
            break
        for c in range(nCols):
            if batchIndex>=nItems:
                break
            outImage[r*nRowsImage:(r+1)*nRowsImage,c*nColsImage:(c+1)*nColsImage]=theBatch[batchIndex]
            batchIndex=batchIndex+1

    # If plot requested, do it
    if doPlot:
        plt.figure()
        plt.imshow(outImage)
        plt.axis('off')
        plt.show()

    # If save requested, do it
    if not (savePath is None):
        imsave(savePath,outImage)

    return outImage

###############################################################################
# SIMPLE TEXT PROGRESS BAR
###############################################################################
def progress_bar(curValue,maxValue):
    thePercentage=curValue/maxValue
    curSize=int(50*thePercentage)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %d%%" % ('='*curSize, int(thePercentage*100)))
    sys.stdout.flush()

###############################################################################
# SET GPU ON/OFF
# MUST BE THE FIRST CALL. CANNOT BE CHANGED WHEN SET. TO CHANGE, RESTART KERNEL
###############################################################################
def set_gpu(useGPU=True):
    # Please note that this is a mix of cabala and magic (probably black magic)
    # from:
    # https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
    # This works on MY computer (Ubuntu 20 and CUDA Toolkit 10.1)
    if not useGPU:
        import os
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        import tensorflow as tf
        physicalDevices=tf.config.experimental.list_physical_devices('GPU')
        assert len(physicalDevices) > 0, "Not enough GPU hardware devices available"
        config=tf.config.experimental.set_memory_growth(physicalDevices[0], True)

###############################################################################
# CONTRASTIVE LOSS
# Source: https://www.pyimagesearch.com/2021/01/18/contrastive-loss-for-siamese-networks-with-keras-and-tensorflow/
###############################################################################
def contrastive_loss(y, preds, margin=1):
    # Explicitly cast the true class label data type to the predicted one.
    y = tf.cast(y, preds.dtype)
    # Calculate both losses (for loops and non loops)
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    # Compute the contrastive loss
    loss = K.mean((1-y) * squaredPreds + y * squaredMargin)
    return loss

###############################################################################
# CONVENIENCE FUNCTION TO BUILD BASE MODEL FILE NAMES
###############################################################################
def build_model_basename(thePath,thePrefix,theList,theEpochs):
    baseName=os.path.join(thePath,thePrefix)
    for curItem in theList:
        baseName+='_%d'%curItem
    baseName+='_EPOCHS%d'%theEpochs
    return baseName

###############################################################################
# CONVENIENCE FUNCTION TO BUILD LOOP MODEL FILE NAMES
# Input  : pathModels - Path to the models folder
#          autoFilters - List of filters used to build the associated
#                        autoencoder.
#          autoEpochs - Autoencoder training epochs
#          denseLayers - List of integers. The number of items in the list
#                        is the number of dense layers to be appended to
#                        the shared part of the model before the
#                        output layer. The specific values are the number
#                        of units in each layer.
#          theGroups - List of [min,max] overlap values used to define the
#                      groups and/or to balance the dataset.
#          isCategorical - Use categorical encoding (True/False)
#          trainFeatureExtractor - True/False : Re-train the encoder?
#          loopEpochs - Number of loop detector model training epochs
# Output : Basename to save/load the model
# Note   : The same function with the appropriate path in pathModels can be
#          used to build the base name to store results (accuracies, ...)
###############################################################################
def build_loopmodel_basename(pathModels,autoFilters,autoEpochs,denseLayers,
                             theGroups,isCategorical,trainFeatureExtractor,
                             loopEpochs,labelsInverted):
    baseName=build_model_basename(pathModels,'LOOP_AUTO',autoFilters,autoEpochs)
    baseName+=build_model_basename('','_DENSE',denseLayers,loopEpochs)
    baseName+=['_NOTCATEGORICAL','_CATEGORICAL'][isCategorical]

    for curGroup in theGroups:
        baseName+='_G'
        strTemp=[]
        for curItem in curGroup:
            strTemp.append(str(curItem).replace('.',''))
        baseName+='_'.join(strTemp)

    baseName+=['_ENCODERFREEZED','_ENCODERTRAINED'][trainFeatureExtractor]
    if not labelsInverted:
        baseName+='_LABELSNOTINVERTED'

    return baseName

###############################################################################
# CONVENIENCE FUNCTION TO BUILD LOOPREADER FILE NAMES
###############################################################################
def build_reader_basename(basePath,
                          outContinuous,
                          theGroups,
                          stepSeparation,
                          doBalance,
                          doInvert,
                          doMirror,
                          normalizeNoCategorical,
                          toCategorical):

    # If theGroups is an integer, convert it to ranges
    if type(theGroups)==int:
        theGroups=[[i/theGroups,(i+1)/theGroups] for i in range(theGroups)]
    # Make the last boundary to be 2 to prevent problems with the right-
    # open intervals.
    theGroups[-1][-1]=2

    # Put header
    strOut=os.path.join(basePath,'LREADER')

    # State groups definition
    for curGroup in theGroups:
        strOut+='_G'
        strTemp=[]
        for curItem in curGroup:
            strTemp.append(str(curItem).replace('.',''))
        strOut+='_'.join(strTemp)

    # State group separation
    strOut+='_STEP'+str(stepSeparation)

    # State all the boolean parameters
    strOut+=['_DISCRETE','_CONTINUOUS'][outContinuous]
    strOut+=['_UNBALANCED','_BALANCED'][doBalance]
    strOut+=['_NOTINVERTED','_INVERTED'][doInvert]
    strOut+=['_NOTMIRRORED','_MIRRORED'][doMirror]
    strOut+=['_NOTNORMALIZED','_NORMALIZED'][normalizeNoCategorical]
    strOut+=['_NOTCATEGORICAL','_CATEGORICAL'][toCategorical]

    return strOut

###############################################################################
# GET FILE NAMES IN PATH AND SPLITS THEM IF REQUESTED
# Gets all the name of the files in thePath with the extension theExtension and
# randomly shuffles the list.
# If smallSplitRatio==None, returns that list.
# If smallSplitRatio!=None, returns two list. The first contains (1-ratio) of
# the original list items and the second one contains ratio of the original
# list items.
###############################################################################
def get_filenames(thePath,theExtension,smallSplitRatio=None):
    # Get all the filenames and shuffle them
    fileNames=[os.path.join(thePath,f) for f in os.listdir(thePath) if os.path.isfile(os.path.join(thePath,f)) and f.endswith(theExtension)]
    np.random.shuffle(fileNames)

    # If no split requested, return the list
    if smallSplitRatio is None:
        return fileNames

    # Divide the file list in two sets
    numFiles=len(fileNames)
    cutIndex=int(smallSplitRatio*numFiles)

    # Return the two sets
    return fileNames[cutIndex:],fileNames[:cutIndex]

###############################################################################
# COMPUTE MULTI-CLASS (INCLUDING THE BINARY CASE) STATS
# Input  : theTruth - True classes. NxS nparray where N is the number of
#                     classes and S in the number of samples. theTruth[i,j]
#                     is 1 if sample j is of class i or 0 otherwise.
#          theScores - Predicted probabilities/scores. nparray of the same
#                      size that theTruth. theScores[i,j] contains the score
#                      or probability of sample j to be of class i.
#          timePerLoop - Prediction time per loop. Must be provided externally
#                        and is used only to be saved together with the other
#                        stats.
#          baseFName - Base name used to build file names to save the ROC
#                      curves and the stats or "None" to not save.
#          doPlot    - Boolean. On-Screen plot (True/False)
#          xxxFontSize - Font sizes in different parts of the ROC curve plot.
# Output : outMatrix - NxM nparray where N is the number of classes and M
#                      is N+6. The format is:
#                      * outMatrix[:,:N] is the confusionMatrix. Rows are the
#                        true classes and columns are the predicted ones.
#                      * outMatrix[:,N] is the ROC-AUC for each class.
#                      * outMatrix[:,N+1] is the per-class accuracy
#                      * outMatrix[:,N+2] is the per-class precision
#                      * outMatrix[:,N+3] is the per-class recall
#                      * outMatrix[:,N+4] is the per-class fall-out
#                      * outMatrix[:,N+5] is the per-class F1-Score
# Note   : The overall accuracy is not provided, though can be easily computed
#          from outMatrix:
#          confusionMatrix=outMatrix[:,:outMatrix.shape[0]]
#          theAccuracy=np.sum(np.diag(confusionMatrix))/np.sum(confusionMatrix)
###############################################################################
def multiclass_stats(theTruth,theScores,timePerLoop,baseFName=None,doPlot=False,legendFontSize=12,labelFontSize=12,ticksFontSize=12):
    # Line styles. Be careful: more than 8 classes lead to repeated styles.
    styleList=['k-', 'r--', 'g:', 'b-.','b-', 'g--', 'r:', 'k-.']

    # Prepare figure only if it has to be saved or shown
    if not (baseFName is None) or doPlot:
        theFigure=plt.figure()

    # Init storage and parameters. The outMatrix is initialized to -1 to make
    # it easy detect holes (which should not happen, but just in case)
    numClasses=theTruth.shape[0]
    outMatrix=np.zeros((numClasses,numClasses+6))-1

    # For each class
    for i in range(numClasses):
        # Get the slices
        curTruth=theTruth[i,:]
        curScore=theScores[i,:]

        # Get the ROC data
        [theFPR,theTPR,theThresholds]=roc_curve(curTruth,curScore)
        # Get the AUC and store it at the appropriate column
        outMatrix[i,numClasses+0]=auc(theFPR,theTPR)
        # Plot the current ROC curve if necessary
        if not (baseFName is None) or doPlot:
            plt.plot(theFPR,theTPR,styleList[i%8],clip_on=False,linewidth=2)

    # If on-screen plot or save curve to file requested
    if not (baseFName is None) or doPlot:
        # Plot the legend
        plt.legend(['CLASS %d - AUC %.3f'%(i,theAUC) for i,theAUC in enumerate(outMatrix[:,numClasses+0])],fontsize=legendFontSize)

        # Plot the axes labels
        plt.xlabel('FPR',fontsize=labelFontSize)
        plt.ylabel('TPR',fontsize=labelFontSize)

        # Limit the axes strictly to (0,1)
        plt.xlim(0,1)
        plt.ylim(0,1)

        # Set remaining font size
        plt.xticks(fontsize=ticksFontSize)
        plt.yticks(fontsize=ticksFontSize)

        # Activate the grid
        plt.grid()

        # Show the image (could be removed if on-screen plot is not wanted?)
        plt.show()

        # Save figure if requested
        if not (baseFName is None):
            rocFileName=baseFName+'_ROC.eps'
            plt.savefig(rocFileName)

        # Close the figure if on-screne plot is not required
        if not doPlot:
            plt.close(theFigure)

    # Get the numeric version of ground truth and predictions
    intGroundTruth=np.argmax(theTruth,axis=0)
    intPredictions=np.argmax(theScores,axis=0)

    # Compute and store the confusion matrix
    confusionMatrix=confusion_matrix(intGroundTruth,intPredictions)
    outMatrix[:,:numClasses]=confusionMatrix

    # Compute stats for each class
    for i in range(numClasses):
        # Get per-class recall, precision, F1-Score and accuracy
        theRecall=confusionMatrix[i,i]/np.sum(confusionMatrix[i,:])
        thePrecision=confusionMatrix[i,i]/np.sum(confusionMatrix[:,i])
        f1Score=2*(thePrecision*theRecall)/(thePrecision+theRecall)
        negativeCases=np.delete(confusionMatrix,i,0)
        fallOut=np.sum(negativeCases[:,i])/np.sum(negativeCases)
        negativeCases=np.delete(negativeCases,i,1)
        theAccuracy=(confusionMatrix[i,i]+np.sum(negativeCases))/np.sum(confusionMatrix)

        # Store them in the output matrix
        outMatrix[i,numClasses+1]=theAccuracy
        outMatrix[i,numClasses+2]=thePrecision
        outMatrix[i,numClasses+3]=theRecall
        outMatrix[i,numClasses+4]=fallOut
        outMatrix[i,numClasses+5]=f1Score

    # Compute the overall accuracy
    theAccuracy=np.sum(np.diag(confusionMatrix))/np.sum(confusionMatrix)

    # Save the stats if requested
    if not (baseFName is None):
        statsFileName=baseFName+'_STATS.pkl'
        with open(statsFileName,'wb') as outFile:
            dump([outMatrix,theAccuracy,timePerLoop],outFile)

    return outMatrix,theAccuracy