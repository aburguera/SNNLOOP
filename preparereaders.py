#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : preparereaders
# Description : Pre-processes different configurations of loop datasets
#               to speed up further process.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 7-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

from loopreader import LoopReader
from utils import build_reader_basename

###############################################################################
# PARAMETERS TO EXPLORE
###############################################################################

# Groups to divide or balance theloops
GROUPS_LIST=[
    [[0],[0.5,2]],
    [[0,0.5],[0.5,2]],
    [[0,0.33],[0.33,0.66],[0.66,2]]
    ]

# [0,1] representation or categorical encoding.
# Note that to have [0,1] when categorical is false, normalizeNoCategorical
# must be True
CATEGORICAL_ENCODING=[False,True]

# Dataset paths to use
PATH_LIST=['../../DATA/LOOPDSTR','../../DATA/LOOPDSTS']

# Parameters dictionary. Useful to use the same parameters both to build the
# file name and create the reader if the file does not exist.
theParams={'basePath':None,
           'outContinuous':False,
           'theGroups':None,
           'stepSeparation':3,
           'doBalance':True,
           'doInvert':False,
           'doMirror':True,
           'normalizeNoCategorical':True,
           'toCategorical':None}

###############################################################################
# LOOP THROUGH ALL THE PARAMETERS
###############################################################################

numCombinations=len(GROUPS_LIST)*len(CATEGORICAL_ENCODING)*len(PATH_LIST)
curCombination=1

# For each dataset
for curPath in PATH_LIST:
    theParams['basePath']=curPath

    # For each group division
    for curGroups in GROUPS_LIST:
        theParams['theGroups']=curGroups
        # For categorical and non-categorical encoding
        for isCategorical in CATEGORICAL_ENCODING:
            # Set the parameters
            theParams['toCategorical']=isCategorical

###############################################################################
# IF THE READER IS NOT SAVED, CREATE IT AND SAVE
###############################################################################

            # Create the reader. It is advisable to create it at each
            # iteration to free memory (since overlap matrices can be
            # extremely large)
            theReader=LoopReader()

            # Build the base name
            baseName=build_reader_basename(**theParams)
            print('[[ %d of %d : PROCESSING %s ]]'%(curCombination,numCombinations,baseName))

            # Check if already saved
            if theReader.is_saved(baseName):
                print('* ALREADY SAVED. SKIPPING PROCESS.')
            else:
                print('* CREATING READER. PLEASE BE PATIENT.')
                theReader.create(**theParams)
                print('* SAVING READER.')
                theReader.save(baseName)
            print('[[ FINISHED %s ]]'%baseName)
            curCombination+=1