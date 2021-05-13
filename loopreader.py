#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : LoopReader
# Description : Reads and pre-processes UCAMGEN loop datasets.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import os
from keras.utils import to_categorical
from skimage.io import imread
from pickle import load,dump

class LoopReader:

###############################################################################
# CONSTRUCTOR
###############################################################################
    def __init__(self):
        self.loopSpecs=None
        self.basePath=None
        self.creationParameters=None

###############################################################################
# BUILDS FROM THE UCAMGEN DATASET
# Input: basePath      - Path of the UCAMGEN generated dataset.
#        outContinuous - True: Outputs a continuous value between 0 and 1
#                              representing the overlap between images.
#                        False: Discrete depending on the group where the
#                               actual overlap falls. Also between the
#                               range [0,1].
#        theGroups     - If discrete output, the overlap ranges where each
#                        class belongs [min val,max val). If doBalance==True,
#                        balance is performed for these ranges.
#                        If continuous output, the ranges are used only to
#                        balance the dataset if doBalance==True.
#                        If one group is a single-element list (i.e. [0]) then
#                        the group contains the loops with that overlap.
#                        If an integer is provided, the ranges are automati-
#                        cally created.
#        stepSeparation- Min number of steps between loop images
#        doBalance     - If True, the number of pairs in each range in
#                        theGroups is set to the number of pairs in the range
#                        with the least pairs in the original dataset.
#        doInvert      - The output overlap is inverted (1-overlap)
#        doMirror      - Replicate every (i,j) loop with the (j,i)
#        normalizeNoCategorical - If regression or categorical requested,
#                        this parameter has no effect. Otherwise, if True,
#                        the class number is changed to a value between 0
#                        and 1.
#        toCategorical - If continuous output, does nothing. If discrete,
#                        the output is converted to categorical.
###############################################################################
    def create(self,
               basePath,
               outContinuous=False,
               theGroups=[[0],[0.5,2]],
               stepSeparation=3,
               doBalance=True,
               doInvert=True,
               doMirror=True,
               normalizeNoCategorical=True,
               toCategorical=False):

        #######################################################################
        # Store the creation parameters
        #######################################################################

        self.creationParameters={
            'basePath':basePath,
            'outContinuous':outContinuous,
            'theGroups':theGroups,
            'stepSeparation':stepSeparation,
            'doBalance':doBalance,
            'doInvert':doInvert,
            'doMirror':doMirror,
            'normalizeNoCategorical':normalizeNoCategorical,
            'toCategorical':toCategorical
            }

        #######################################################################
        # Prepare theGroups
        #######################################################################

        # Create the group ranges if an integer has been passed
        if type(theGroups)==int:
            theGroups=[[i/theGroups,(i+1)/theGroups] for i in range(theGroups)]
        # Make the last boundary to be 2 to prevent problems with the right-
        # open intervals.
        theGroups[-1][-1]=2

        #######################################################################
        # Load overlap matrix
        #######################################################################

        print('LOADING OVERLAP MATRIX. PLEASE BE PATIENT.')

        # Read the matrix and divide by 100 to have values in [0,1]
        overlapMatrix=np.loadtxt(os.path.join(basePath,'OVERLAP.csv'),delimiter=',')/100

        # Put a negative value in the lower diagonal to avoid picking repeated
        # loops.
        overlapMatrix[np.tril_indices_from(overlapMatrix)]=-100

        #######################################################################
        # Get the loops in each group
        #######################################################################

        print('SPLITTING DATA INTO GROUPS.')

        self.loopSpecs=[]
        for curGroup in theGroups:

            # If the group has min and max, search for loops within this range
            # of overlap.
            if len(curGroup)==2:
                [rGroup,cGroup]=np.where((overlapMatrix>=curGroup[0])&(overlapMatrix<curGroup[1]))

            # If the group is defined with a single number, search for loops
            # with that exact overlap.
            else:
                [rGroup,cGroup]=np.where(overlapMatrix==curGroup[0])

            # Kepp only those loops separated at least stepSeparation
            i=np.where(np.abs(rGroup-cGroup)>=stepSeparation)
            curLoop=np.vstack((rGroup[i],cGroup[i]))

            # Shuffle the data to ease cutting it later
            np.random.shuffle(curLoop.transpose())

            # Store it in the loopSpecs list
            self.loopSpecs.append(curLoop.astype('int'))

        #######################################################################
        # Balance the groups if requested
        #######################################################################

        if doBalance:

            print('BALANCING DATA.')

            # Get the smallest group size
            minSize=np.min([curLoop.shape[1] for curLoop in self.loopSpecs])

            # Cut the groups (easy since they are shuffled)
            self.loopSpecs=[self.loopSpecs[i][:,:minSize] for i in range(len(self.loopSpecs))]

        #######################################################################
        # Prepare continuous tags
        #######################################################################

        # If continuous output...
        if outContinuous:

            print('PREPARING CONTINUOUS OUTPUT.')

            # If inversion is requested, pick the overlaps from the overlap
            # matrix and place 1 minus the overlaps as third loopSpecs row.
            if doInvert:
                self.loopSpecs=[np.vstack((self.loopSpecs[i],1-overlapMatrix[self.loopSpecs[i][0,:],self.loopSpecs[i][1,:]])) for i in range(len(self.loopSpecs))]

            # If inversion is not requested, use the overlaps as they are in
            # the overlap matrix.
            else:
                self.loopSpecs=[np.vstack((self.loopSpecs[i],overlapMatrix[self.loopSpecs[i][0,:],self.loopSpecs[i][1,:]])) for i in range(len(self.loopSpecs))]

        #######################################################################
        # Prepare discrete tags
        #######################################################################

        # If discrete output...
        else:

            print('PREPARING DISCRETE OUTPUT.')

            # For each loopSpec
            for i in range(len(self.loopSpecs)):

                # If inversion requested, labels will range from num specs
                # minus one to zero
                if doInvert:
                    curTag=len(self.loopSpecs)-i-1

                # If no inversion is requested labels will range from zero to
                # num specs minus one
                else:
                    curTag=i

                # Build the tags for each item in the current loop spec
                theTags=np.zeros((1,self.loopSpecs[i].shape[1]))+curTag

                # If categorical output requested, convert the tags
                if toCategorical:
                    theTags=to_categorical(theTags.transpose(),len(self.loopSpecs)).transpose()

                # If no categorical but normalized requested, make the tags
                # range between 0 and 1.
                elif normalizeNoCategorical:
                    theTags=theTags/(len(self.loopSpecs)-1)

                # Stack the tags to the current loop spec group
                self.loopSpecs[i]=np.vstack((self.loopSpecs[i],theTags))

        #######################################################################
        # Join all the groups in loopSpecs
        #######################################################################

        print('JOINING ALL GROUPS.')

        # Join the loop specs groups
        self.loopSpecs=np.concatenate([curGroup for curGroup in self.loopSpecs],axis=1)

        # Mirror them if requested
        if doMirror:

            print('MIRRORING DATA.')

            mirroredSpecs=self.loopSpecs.copy()
            mirroredSpecs[[0,1]]=mirroredSpecs[[1,0]]
            self.loopSpecs=np.concatenate((self.loopSpecs,mirroredSpecs),axis=1)

        print('SHUFFLING DATA.')

        # Shuffle them to remove any sense of order
        np.random.shuffle(self.loopSpecs.transpose())

        print('DONE!')

###############################################################################
# CHECKS IF THE READER IS SAVED
###############################################################################
    def is_saved(self,baseName):
        return os.path.exists(baseName+'.pkl')

###############################################################################
# SAVES THE READER
###############################################################################
    def save(self,baseName,forceOverwrite=False):
        if (not forceOverwrite) and self.is_saved(baseName):
            print('[ABORTING] File %s.pkl already exists.'%baseName)
            return
        with open(baseName+'.pkl','wb') as outFile:
            dump([self.loopSpecs,self.creationParameters],outFile)

###############################################################################
# LOADS THE READER
###############################################################################
    def load(self,baseName):
        if not self.is_saved(baseName):
            print('[ABORTING] Reader file %s.pkl not found.'%baseName)
            return
        with open(baseName+'.pkl','rb') as inFile:
            [self.loopSpecs,self.creationParameters]=load(inFile)
        self.basePath=self.creationParameters['basePath']

###############################################################################
# CONSTRUCTS THE FILE NAME OF THE SPECIFIED IMAGE INDEX
###############################################################################
    def get_filename(self,idImage):
        return os.path.join(self.basePath,'IMAGES','IMAGE%05d.png'%(idImage+1))

###############################################################################
# RETURNS THE IMAGE CORRESPONDING TO THE PROVIDED INDEX
###############################################################################
    def get_image(self,idImage):
        return imread(self.get_filename(idImage))