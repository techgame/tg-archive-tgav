##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2007  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import *
import _ctypes_support

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#openALLib = _ctypes_support.loadFirstLibrary('OpenAL', 'OpenAL32')
#alutLib = _ctypes_support.loadFirstLibrary('ALUT') or openALLib

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        return fn
        fnErrCheck = errcheck
        if fn.__name__ in errorFuncNames:
            bindErrorFunc = True
        else:
            bindErrorFunc = False
            if not errcheck:
                fnErrCheck = _getErrorCheckForFn(fn)

        result = _ctypes_support.attachToLibFn(fn, restype, argtypes, fnErrCheck, openALLib)

        if bindErrorFunc:
            _bindError(result)

        return result
    return bindFuncTypes

