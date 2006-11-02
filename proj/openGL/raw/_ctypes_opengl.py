#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import re
from ctypes import *
import _ctypes_support

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

openGLLib = _ctypes_support.loadFirstLibrary('OpenGL', 'OpenGL32')
#glutLib = _ctypes_support.loadFirstLibrary('GLUT')

getNameToFirstDigit = re.compile(r'(gl[A-Za-z_]+)\d?').match
noErrorCheckAllowed = set([
        'glBegin',
        'glVertex', 'glColor', 'glIndex', 
        'glNormal', 'glEdgeFlag', 
        'glTexCoord', 'glMultiTexCoord', 'glMaterial', 
        'glEvalCoord', 'glEvalPoint', 
        'glArrayElement', 
        'glCallList', 'glCallLists', 
        ])

def canErrorCheckFn(fn):
    name = getNameToFirstDigit(fn.__name__).groups()[0]
    return name not in noErrorCheckAllowed

glGetError = None

def glCheckError(result, func, args):
    #print '>>> %s%r -> %r ' % (func.__name__, args, result)
    err = glGetError()
    if err != 0:
        from errors import GLError
        raise GLError(err, callInfo=(func, args, result))
    return result

def _bindError(errorFunc, g=globals()):
    g[errorFunc.__name__] = errorFunc

def _getErrorCheckForFn(fn):
    if canErrorCheckFn(fn):
        return glCheckError

def cleanupNamespace(namespace):
    _ctypes_support.scrubNamespace(namespace, globals())

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        fnErrCheck = errcheck
        if fn.__name__ == 'glGetError':
            bindErrorFunc = True
        else:
            bindErrorFunc = False
            if not errcheck:
                fnErrCheck = _getErrorCheckForFn(fn)

        result = _ctypes_support.attachToLibFn(fn, restype, argtypes, fnErrCheck, openGLLib)

        if bindErrorFunc:
            _bindError(result)

        return result
    return bindFuncTypes

