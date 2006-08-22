#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import al, alc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALException(Exception):
    errorFormat = 'AL exception %s(0x%x) %s'
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

    def __init__(self, alError, moreInfo=None):
        errString = self.errorFormat % (self.errorMap.get(alError, "???"), alError, moreInfo or '')
        Exception.__init__(self, errString)
        self.value = alError

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALCException(ALException):
    errorFormat = 'ALC exception %s(0x%x) %s'
    errorMap = {
            alc.ALC_NO_ERROR: 'ALC_NO_ERROR',
            alc.ALC_INVALID_DEVICE: 'ALC_INVALID_DEVICE',
            alc.ALC_INVALID_CONTEXT: 'ALC_INVALID_CONTEXT',
            alc.ALC_INVALID_ENUM: 'ALC_INVALID_ENUM',
            alc.ALC_INVALID_VALUE: 'ALC_INVALID_VALUE',
            }

