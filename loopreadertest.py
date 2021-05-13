#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Name        : LoopReaderTest
# Description : Usage example of LoopReader
#               LoopReaders are in charge or pre-processing UCAMGEN datasets
#               targetting one specific configuration.
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 4-April-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# IMPORTS
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from loopreader import LoopReader
from utils import build_reader_basename

###############################################################################
# PARAMETERS
###############################################################################

# Path to the dataset. Must be created by UCAMGEN or have the same format.
# Please check the UCAMGEN repository for more information.
PATH_DATASET='../../DATA/LOOPDSTRSMALL'

# Type of loop information: True -> Aimed at regression, False -> Aimed at
# classification.
DATA_CONTINUOUS=False;

# Groups in which the actual overlap in the dataset is split. The groups are
# used to create the classes (only if DATA_CONTINUOUS=False) and to balance the
# dataset (independently of DATA_CONTINUOUS). Balancing means that the number
# of items in each group is randomly selected to be the same.
# DATA_GROUPS can be:
# 1.- A list of lists. Each item in the list can be:
#     1.1.- A single value. In this case the group refers to those items whose
#           overlap is EXACTLY that value. Thus, be careful or you will
#           end up with an empty group. This option is meant to be used
#           to search non-loops by using a 0.
#     1.2.- Two values. These values define a right-open interval. Thus, the
#           group will contain items whose overlap is greater or equal than
#           the first value and smaller than the second one. That is why if
#           you want to include, for example, overlaps between 0.5 and 1 you
#           have to put a value larger than 1 as the second value (2 is used
#           in the example).
# 2.- An integer. The integer represents the number of groups. The intervals
#     are, in this case, homogeneously distributed within the groups.
# Please take into account that balancing the groups means making them all of
# the same size that the smallest. Thus, empty groups can lead to execution
# errors. Also, very small groups would dramatically reduce the number of
# samples in the loop reader.
DATA_GROUPS=[[0,0.5],[0.5,2]]

# Number of frames two images must be separated to be considered a loop. Use
# this parameter to prevent considering consecutive images as actual loops.
DATA_SEPARATION=3

# Balance the data. Balancing has been described before. Accordingly, if
# continuous output is requested (DATA_CONTINUOUS=True) but no balancing
# is desired (DATA_BALANCE=False), then the DATA_GROUPS are meaningless.
DATA_BALANCE=True

# Invert the overlap before proceeding. This is useful if instead of an overlap
# metric a distance metric is preferred. Not inverting means that two equal
# images have a metric (overlap) of 1 whilst inverting means that two equal
# images have a metric (distance) of 0.
DATA_INVERT=False

# If True, every loop between images (i,j) is included twice: one as a loop
# between (i,j) and another one as a loop between (j,i). If the target system
# (Neural Network or other) implicitly considers symmetry between loops, then
# it is advisable to use DATA_MIRROR=False. Otherwise, it is better to use
# DATA_MIRROR=True
DATA_MIRROR=True

# If continuous output requested (DATA_CONTINUOUS=True), DATA_CATEGORICAL
# does nothing. Otherwise, it selects between an integer representation of
# the group (0, 1, 2) when DATA_CATEGORICAL=False or a One-Hot representation
# of the group (001,010,100) if DATA_CATEGORICAL=True
DATA_CATEGORICAL=False

# If continuous (DATA_CONTINUOUS=True) or categorical (DATA_CATEGORICAL=True)
# output, does nothing. Otherwise, normalizes the group number to be between
# 0 and 1 (DATA_NORMALIZE_NO_CATEGORICAL=True) or leaves it between 0 and
# len(DATA_GROUPS)-1 (DATA_NORMALIZE_NO_CATEGORICAL=False).
DATA_NORMALIZE_NO_CATEGORICAL=False

###############################################################################
# CREATE THE LOOP READER
###############################################################################

# Create the object
theReader=LoopReader()

# Prepare the parameters into a convenience dictionary
theParameters={'basePath':PATH_DATASET,
               'outContinuous':DATA_CONTINUOUS,
               'theGroups':DATA_GROUPS,
               'stepSeparation':DATA_SEPARATION,
               'doBalance':DATA_BALANCE,
               'doInvert':DATA_INVERT,
               'doMirror':DATA_MIRROR,
               'normalizeNoCategorical':DATA_NORMALIZE_NO_CATEGORICAL,
               'toCategorical':DATA_CATEGORICAL}

# Use the utils convenience function to build a filename to save the reader
baseName=build_reader_basename(**theParameters)

# Check if the reader is already saved.
if theReader.is_saved(baseName):
    # If saved, load it.
    print('[LOADING READER]')
    theReader.load(baseName)
    print('[READER LOADED]')
else:
    # If not saved, build it
    print('[BUILDING READER]')
    theReader.create(**theParameters)
    print('[READER BUILT]')
    # and save it
    print('[SAVING READER]')
    theReader.save(baseName)
    print('[READER SAVED]')

###############################################################################
# PLOT TWO EXAMPLES
###############################################################################

# At this point, theReader.loopSpecs has three rows. First row and second row
# identify two images and the third row states the group where it belongs.
# The group format depends on the output format (thus, if categorical output
# is selected, the number of rows will not be 3 but 2+num_groups).
# The images can be accessed from the id by means of get_image.

# Get one loop in group 0 (overlap within interval [0,0.5) in this case). Just
# change the second index to see other image pairs in this group.
iGroup0=np.where(theReader.loopSpecs[2,:]==0)[0][0]

# Get the two images in that loop
firstImage=theReader.get_image(theReader.loopSpecs[0,iGroup0])
secondImage=theReader.get_image(theReader.loopSpecs[1,iGroup0])

# Plot them
plt.figure()
plt.suptitle('LOW OVERLAP')
plt.subplot(1,2,1)
plt.imshow(firstImage)
plt.subplot(1,2,2)
plt.imshow(secondImage)
plt.show()

# Now, repeat the process for group 1. Yes, defining a function is better than
# copy/pasting code, but this is meant to be an example to clarify how the
# LoopReader can be used.
iGroup1=np.where(theReader.loopSpecs[2,:]==1)[0][0]
firstImage=theReader.get_image(theReader.loopSpecs[0,iGroup1])
secondImage=theReader.get_image(theReader.loopSpecs[1,iGroup1])
plt.figure()
plt.suptitle('LARGE OVERLAP')
plt.subplot(1,2,1)
plt.imshow(firstImage)
plt.subplot(1,2,2)
plt.imshow(secondImage)
plt.show()