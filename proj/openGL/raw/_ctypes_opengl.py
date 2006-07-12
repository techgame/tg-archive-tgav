#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import *
from ctypes.util import find_library

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def loadFirstLibrary(*libraryname):
    for name in libraryname: 
        path = find_library(name)
        if path:
            library = cdll.LoadLibrary(path)
            return library

openGLLib = loadFirstLibrary('OpenGL', 'OpenGL32')
#glutLib = loadFirstLibrary('GLUT')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def attachToLibFn(lib, fn):
    libfn = getattr(lib, fn.__name__, None)
    if libfn is not None:
        libfn.restype = fn.restype
        libfn.argtypes = fn.argtypes
        if fn.errcheck is not None:
            libfn.errcheck = fn.errcheck

        libfn.__doc__ = '%s(%s)' % (fn.__name__, ', '.join(fn.func_code.co_varnames))
    fn.api = libfn
    return libfn

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def bind(restype, argtypes, errcheck=None):
    def bindFuncTypes(fn):
        fn.restype = restype
        fn.argtypes = argtypes
        fn.errcheck = errcheck
        return attachToLibFn(openGLLib, fn) or fn

    return bindFuncTypes

