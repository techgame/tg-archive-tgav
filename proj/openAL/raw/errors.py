##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2006  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import al, alc, alut

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALException(Exception):
    exceptionFmt = 'AL exception %s(0x%x) %s'
    errorMap = {
            al.AL_NO_ERROR: 'AL_NO_ERROR',
            al.AL_INVALID_NAME: 'AL_INVALID_NAME',
            al.AL_INVALID_ENUM: 'AL_INVALID_ENUM',
            al.AL_INVALID_VALUE: 'AL_INVALID_VALUE',
            al.AL_INVALID_OPERATION: 'AL_INVALID_OPERATION',
            al.AL_ILLEGAL_COMMAND: 'AL_ILLEGAL_COMMAND',
            al.AL_ILLEGAL_ENUM: 'AL_ILLEGAL_ENUM',
            al.AL_OUT_OF_MEMORY: 'AL_OUT_OF_MEMORY',
            }

    def __init__(self, error, alErrorString='', callInfo=None):
        self.error = error
        self.errorString = alErrorString
        excStr = self.exceptionFmt % (self.errorMap.get(error, "???"), error, alErrorString)
        Exception.__init__(self, excStr)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALCException(ALException):
    exceptionFmt = 'ALC exception %s(0x%x) %s'
    errorMap = {
            alc.ALC_NO_ERROR: 'ALC_NO_ERROR',
            alc.ALC_INVALID_DEVICE: 'ALC_INVALID_DEVICE',
            alc.ALC_INVALID_CONTEXT: 'ALC_INVALID_CONTEXT',
            alc.ALC_INVALID_ENUM: 'ALC_INVALID_ENUM',
            alc.ALC_INVALID_VALUE: 'ALC_INVALID_VALUE',
            }

class ALUTException(ALException):
    exceptionFmt = 'ALUT exception %s(0x%x) %s'
    errorMap = {
            alut.ALUT_ERROR_NO_ERROR: 'ALUT_ERROR_NO_ERROR',
            alut.ALUT_ERROR_OUT_OF_MEMORY: 'ALUT_ERROR_OUT_OF_MEMORY',
            alut.ALUT_ERROR_INVALID_ENUM: 'ALUT_ERROR_INVALID_ENUM',
            alut.ALUT_ERROR_INVALID_VALUE: 'ALUT_ERROR_INVALID_VALUE',
            alut.ALUT_ERROR_INVALID_OPERATION: 'ALUT_ERROR_INVALID_OPERATION',
            alut.ALUT_ERROR_NO_CURRENT_CONTEXT: 'ALUT_ERROR_NO_CURRENT_CONTEXT',
            alut.ALUT_ERROR_AL_ERROR_ON_ENTRY: 'ALUT_ERROR_AL_ERROR_ON_ENTRY',
            alut.ALUT_ERROR_ALC_ERROR_ON_ENTRY: 'ALUT_ERROR_ALC_ERROR_ON_ENTRY',
            alut.ALUT_ERROR_OPEN_DEVICE: 'ALUT_ERROR_OPEN_DEVICE',
            alut.ALUT_ERROR_CLOSE_DEVICE: 'ALUT_ERROR_CLOSE_DEVICE',
            alut.ALUT_ERROR_CREATE_CONTEXT: 'ALUT_ERROR_CREATE_CONTEXT',
            alut.ALUT_ERROR_MAKE_CONTEXT_CURRENT: 'ALUT_ERROR_MAKE_CONTEXT_CURRENT',
            alut.ALUT_ERROR_DESTROY_CONTEXT: 'ALUT_ERROR_DESTROY_CONTEXT',
            alut.ALUT_ERROR_GEN_BUFFERS: 'ALUT_ERROR_GEN_BUFFERS',
            alut.ALUT_ERROR_BUFFER_DATA: 'ALUT_ERROR_BUFFER_DATA',
            alut.ALUT_ERROR_IO_ERROR: 'ALUT_ERROR_IO_ERROR',
            alut.ALUT_ERROR_UNSUPPORTED_FILE_TYPE: 'ALUT_ERROR_UNSUPPORTED_FILE_TYPE',
            alut.ALUT_ERROR_UNSUPPORTED_FILE_SUBTYPE: 'ALUT_ERROR_UNSUPPORTED_FILE_SUBTYPE',
            alut.ALUT_ERROR_CORRUPT_OR_TRUNCATED_DATA: 'ALUT_ERROR_CORRUPT_OR_TRUNCATED_DATA',
            }

