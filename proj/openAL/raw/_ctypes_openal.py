#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import *
import _ctypes_support

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_errorExclusions = set([])#'alcCloseDevice', 'alcMakeContextCurrent'])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

openALLib = _ctypes_support.loadFirstLibrary('OpenAL', 'OpenAL32')
alutLib = _ctypes_support.loadFirstLibrary('ALUT') or openALLib

errorFuncNames = set(['alGetError', 'alcGetError', 'alcGetContextsDevice'])
alGetError = None
alcGetError = None
alcGetContextsDevice = None
alutGetError = None
alutGetErrorString = None

def alCheckError(result, func, args):
    #print '>>> %s%r -> %r ' % (func.__name__, args, result)
    alErr = alGetError()
    if alErr != 0:
        import errors
        raise errors.ALException(alErr, callInfo=(func, args, result))
    return result

def alcCheckDeviceError(result, func, args):
    #print '>>> %s%r -> %r ' % (func.__name__, args, result)
    device = args[0]
    if device:
        alcErr = alcGetError(device)

        if alcErr != 0:
            import errors
            raise errors.ALCException(alcErr, callInfo=(func, args, result))
    return result

def alcCheckContextError(result, func, args):
    #print '>>> %s%r -> %r ' % (func.__name__, args, result)
    context = args[0]
    device = alcGetContextsDevice(context)
    if device:
        alcErr = alcGetError(device)

        if alcErr != 0:
            import errors
            raise errors.ALCException(alcErr, callInfo=(func, args, result))
    return result

def alutCheckError(result, func, args):
    #print '>>> %s%r -> %r ' % (func.__name__, args, result)
    alutErr = alutGetError()
    if alutErr != 0:
        import errors
        raise errors.ALUTException(alutErr, alutGetErrorString(alutErr), callInfo=(func, args, result))
    return result

def _getErrorCheckForFn(fn):
    if fn.__name__.startswith('alc'):
        firstArgName = (fn.func_code.co_varnames[:1] or [None])[0]
        if firstArgName == 'context':
            fnErrCheck = alcCheckContextError
        elif firstArgName in ('device', 'devicename'):
            fnErrCheck = alcCheckDeviceError
        else:
            fnErrCheck = None
    else:
        fnErrCheck = alCheckError
    return fnErrCheck

def _bindError(errorFunc, g=globals()):
    g[errorFunc.__name__] = errorFunc

def cleanupNamespace(namespace):
    _ctypes_support.scrubNamespace(namespace, globals())

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
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

def alutBind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        fnErrCheck = errcheck
        if fn.__name__ in errorFuncNames:
            bindErrorFunc = True
        else:
            bindErrorFunc = False
            if not errcheck:
                fnErrCheck = _getErrorCheckForFn(fn)

        if bindErrorFunc:
            _bindError(result)

        return _ctypes_support.attachToLibFn(fn, restype, argtypes, errcheck, alutLib)
    return bindFuncTypes

