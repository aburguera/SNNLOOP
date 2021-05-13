#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : LoopGenerator
# Description : Keras data generator (inherits from Sequence) to feed loop
#               siamese loop detector models.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 5-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from keras.utils import Sequence
from skimage.transform import resize
import numpy as np
from skimage import img_as_float

class LoopGenerator(Sequence):

###############################################################################
# CONSTRUCTOR
# Input  - loopSpecs : nparray of N rows and M cols. First and second row
#                      contain image identifiers. Third row onward contain
#                      the loop information (continuous, discrete, categori-
#                      cal, ...). This array typically comes from a LoopReader.
#          imgGetter : Function that given an ID returns the image.
#          imgSize   : Desired image size. Please note that the loop generator
#                      resizes the images if necessary, but does NOT change
#                      the number of channels.
#          batchSize : Wel... the batch size.
#          doRandomize : True - The loopSpecs are shuffled at the beginning
#                        after each epoch. False - Do not shuffle.
###############################################################################
    def __init__(self,loopSpecs,imgGetter,imgSize=(64,64),batchSize=10,doRandomize=True):
        # Store input parameters
        self.loopSpecs=loopSpecs.copy()
        self.imgGetter=imgGetter
        self.imgSize=imgSize
        self.batchSize=batchSize
        self.doRandomize=doRandomize

        # Get the number of loops
        self.numLoops=self.loopSpecs.shape[1]

        # Randomize if requested
        self.on_epoch_end()
###############################################################################
# EXECUTED AFTER EACH EPOCH. SHUFFLES DATA IF REQUESTED.
###############################################################################
    def on_epoch_end(self):
        if self.doRandomize:
            np.random.shuffle(self.loopSpecs.transpose())

###############################################################################
# GET THE NUMBER OF BATCHES
###############################################################################
    def __len__(self):
        return int(np.ceil(self.numLoops/float(self.batchSize)))

###############################################################################
# PROVIDES THE REQUESTED BATCH
###############################################################################
    def __getitem__(self,theIndex):
        X1=[]
        X2=[]
        y=[]

        # Compute start and end image indexes in the requested batch
        bStart=max(theIndex*self.batchSize,0)
        bEnd=min((theIndex+1)*self.batchSize,self.numLoops)

        # For each image in the batch
        for i in range(bStart,bEnd):
            # Read the images
            firstImage=img_as_float(self.imgGetter(self.loopSpecs[0,i]))
            secondImage=img_as_float(self.imgGetter(self.loopSpecs[1,i]))

            # Resize only if necessary
            if firstImage.shape[:2]!=self.imgSize:
                firstImage=resize(firstImage,self.imgSize)
            if secondImage.shape[:2]!=self.imgSize:
                secondImage=resize(secondImage,self.imgSize)

            # Append the images and the ground truth
            X1.append(firstImage)
            X2.append(secondImage)
            y.append(self.loopSpecs[2:,i])

        # Return the same data as X and y
        return [np.array(X1),np.array(X2)],np.array(y)