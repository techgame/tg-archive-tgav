##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from TG.interop.dyndll import dynLibs

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_openALlib = None
def getOpenALLib():
    global _openALlib
    if _openALlib is None:
        try: 
            _openALlib = dynLibs.find('OpenAL32', False)
        except OSError:
            pass

        if _openALlib is None:
            _openALlib = dynLibs.find('OpenAL', False)

    return _openALlib

_alutLib = None
def getALUTLib():
    global _alutLib
    if _alutLib is None:
        try:
            _alutLib = dynLibs.find('ALUT', False)
        except OSError:
            _alutLib = getOpenALLib()
    return _alutLib

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Exception classes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class OpenALException(Exception):
    error = None
    def __init__(self, errstr, error=None, callResult=None, cFunc=None):
        Exception.__init__(self, errstr)
        self.error = error
        self.cFunc = cFunc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALException(OpenALException):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALCException(OpenALException):
    pass

class ALCGeneralException(ALCException):
    pass

class ALCDeviceException(ALCException):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALUTException(OpenALException):
    pass

