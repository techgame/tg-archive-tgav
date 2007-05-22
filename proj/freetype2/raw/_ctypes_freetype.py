#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import *
import _ctypes_support

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

freetypeLib = _ctypes_support.loadFirstLibrary('freetype6', 'freetype')

def cleanupNamespace(namespace):
    _ctypes_support.scrubNamespace(namespace, globals())

class FreetypeException(Exception):
    def __init__(self, ftError, moreInfo):
        Exception.__init__(self, '%s (%s)' % (ftError, moreInfo))
        self.ftError = ftError

def checkFreetypeError(ftError, func, args):
    #print '%s%r -> %s' % (func.__name__, args, ftError)
    if ftError != 0:
        raise FreetypeException(ftError, '%s%r' % (func.__name__, args))
    return ftError

def bind(restype, argtypes, errcheck=None):
    if errcheck:
        errcheck = checkFreetypeError
    def bindFuncTypes(fn):
        return _ctypes_support.attachToLibFn(fn, restype, argtypes, errcheck, freetypeLib)
    return bindFuncTypes

