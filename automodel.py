#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
# Name        : AutoModel
# Description : Simple wrapper to ease the acces to the autoencoder Keras model
# Author      : Antoni Burguera (antoni dot burguera at uib dot com)
# History     : 31-March-2021 - Creation
# Citation    : Please, refer to the README file to know how to properly cite
#               us if you use this software.
###############################################################################

###############################################################################
# CONSTRUCTOR. PLEASE NOTE THAT THIS DOES NOT CREATE THE MODEL.
###############################################################################

from keras.models import Model,load_model
from keras.layers import Conv2D, Conv2DTranspose, Input, BatchNormalization, LeakyReLU
from modelwrapper import ModelWrapper

class AutoModel(ModelWrapper):

###############################################################################
# CONSTRUCTOR. PLEASE NOTE THAT THIS DOES NOT CREATE THE MODEL.
###############################################################################
    def __init__(self):
        self.encoderModel=None
        self.decoderModel=None
        super().__init__()

###############################################################################
# CREATES THE MODEL
###############################################################################
    def create(self,inputShape=(64,64,3),theFilters=[16,16,32]):
        # Create the encoder layers
        encoderInput=Input(shape=inputShape,name='encoder_input')
        x=encoderInput
        for curFilter in theFilters:
            x=Conv2D(curFilter,kernel_size=(3,3),strides=2,padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            x=BatchNormalization()(x)

        encoderOutput=x

        # Create the decoder layers
        decoderInput=Input(shape=tuple(encoderOutput.get_shape()[1:]), name='decoder_input')
        x=decoderInput
        for curFilter in theFilters[::-1]:
            x=Conv2DTranspose(curFilter,kernel_size=(3,3),strides=2,padding='same')(x)
            x=LeakyReLU(alpha=0.2)(x)
            x=BatchNormalization()(x)
        decoderOutput=Conv2DTranspose(inputShape[2],kernel_size=(3,3),padding='same')(x)

        # Create the models
        self.encoderModel=Model(encoderInput,encoderOutput,name='encoder_model')
        self.decoderModel=Model(decoderInput,decoderOutput,name='decoder_model')
        self.theModel=Model(encoderInput,self.decoderModel(self.encoderModel(encoderInput)),name='autoencoder_model')

        # Compile them
        self.theModel.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','mae'])
        self.encoderModel.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','mae'])
        self.encoderModel.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','mae'])

###############################################################################
# PRINTS THE SUMMARY OF ALL MODELS
###############################################################################
    def summary(self):
        super().summary()
        self.encoderModel.summary()
        self.decoderModel.summary()

###############################################################################
# SAVE THE MODELS IF POSSIBLE
###############################################################################
    def save(self,baseName,forceOverwrite=False):
        if super().save(baseName,forceOverwrite):
            self.encoderModel.save(baseName+'_ENC.h5')
            self.decoderModel.save(baseName+'_DEC.h5')

###############################################################################
# LOAD THE MODELS
###############################################################################
    def load(self,baseName):
        self.encoderModel=load_model(baseName+'_ENC.h5')
        self.decoderModel=load_model(baseName+'_DEC.h5')
        super().load(baseName)

###############################################################################
# OUTPUT THE ENCODER PREDICTIONS
###############################################################################
    def encode(self,theImages):
        return self.encoderModel.predict(theImages)

###############################################################################
# OUTPUT THE DECODER PREDICTIONS
###############################################################################
    def decode(self,theFeatures):
        return self.decoderModel.predict(theFeatures)