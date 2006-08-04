#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import *
import _ctypes_support

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

freetypeLib = _ctypes_support.loadFirstLibrary('freetype')

def cleanupNamespace(namespace):
    _ctypes_support.scrubNamespace(namespace, globals())

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        return _ctypes_support.attachToLibFn(fn, restype, argtypes, errcheck, freetypeLib)
    return bindFuncTypes

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Arg(object):
    def __init__(self, type=None):
        pass
    def from_param(self, obj):
        return obj._handle_
class Ref(object):
    def __init__(self, type=None):
        pass
    def from_param(self, obj):
        return byref(obj._handle_)

