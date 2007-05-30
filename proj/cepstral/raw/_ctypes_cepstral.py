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

cepstralLib = _ctypes_support.loadFirstLibrary('swift')

errorFuncNames = set([
        'swift_event_get_error',
        'swift_strerror',
        ])

def swiftCheckError(result, func, args):
    if result.value != 0:
        import errors
        raise errors.CepstralError(
                result.value, 
                swift_strerror(result), 
                callInfo=(func, args, result))
    return result

def _getErrorCheckForFn(fn, restype):
    if restype and restype.__name__ == 'swift_result_t':
        return swiftCheckError
    return None

def _bindError(errorFunc, g=globals()):
    g[errorFunc.__name__] = errorFunc

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        fnErrCheck = errcheck
        if fn.__name__ in errorFuncNames:
            bindErrorFunc = True
        else:
            bindErrorFunc = False
            if not errcheck:
                fnErrCheck = _getErrorCheckForFn(fn, restype)

        result = _ctypes_support.attachToLibFn(fn, restype, argtypes, fnErrCheck, cepstralLib)

        if bindErrorFunc:
            _bindError(result)

        return result
    return bindFuncTypes

