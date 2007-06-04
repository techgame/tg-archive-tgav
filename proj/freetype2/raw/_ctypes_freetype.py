#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import *
import _ctypes_support

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

freetypeLib = _ctypes_support.loadFirstLibrary('freetype6', 'freetype')

errorFuncNames = set([])
errorSkipNames = set([
        'FT_Get_Kerning'])

def cleanupNamespace(namespace):
    _ctypes_support.scrubNamespace(namespace, globals())

def checkFreetypeError(ftError, func, args):
    if ftError.value != 0:
        import errors
        raise errors.FreetypeException(ftError.value, 
                callInfo=(func, args, ftError))
    return ftError

def _getErrorCheckForFn(fn, restype):
    if fn.__name__ in errorSkipNames:
        return None

    if restype and restype.__name__ == 'FT_Error':
        return checkFreetypeError
    return None

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        fnErrCheck = errcheck
        if errcheck is None:
            fnErrCheck = _getErrorCheckForFn(fn, restype)

        return _ctypes_support.attachToLibFn(fn, restype, argtypes, fnErrCheck, freetypeLib)
    return bindFuncTypes


