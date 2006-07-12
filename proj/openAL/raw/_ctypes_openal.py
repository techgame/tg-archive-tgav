#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import *
import _ctypes_support

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

openALLib = _ctypes_support.loadFirstLibrary('OpenAL', 'OpenAL32')
#alutLib = _ctypes_support.loadFirstLibrary('ALUT')

def cleanupNamespace(namespace):
    _ctypes_support.scrubNamespace(namespace, globals())

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        return _ctypes_support.attachToLibFn(fn, restype, argtypes, errcheck, openALLib)
    return bindFuncTypes

