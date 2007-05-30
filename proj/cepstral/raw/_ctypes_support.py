##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2006  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import new
from ctypes import cdll
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

def attachToLibFn(fn, restype, argtypes, errcheck, lib):
    fn.api = getattr(lib, fn.__name__, None)
    if fn.api is not None:
        fn.api.restype = restype
        fn.api.argtypes = argtypes
        if errcheck is not None:
            fn.api.errcheck = errcheck

        doc = getattr(fn, '__doc__', None)
        if doc:
            fn.api.__doc__ = doc
        return replaceFunctionApi(fn, fn.api)
    else:
        return fn

def replaceFunctionApi(fn, api):
    assert fn.func_defaults == (None,), (fn, fn.func_defaults)
    assert fn.func_code.co_varnames[-1] == '_api_', (fn, fn.func_code.co_varnames)
    result = new.function(fn.func_code, fn.func_globals, fn.func_name, (api,), fn.func_closure)
    result.func_dict = fn.func_dict
    return result

def scrubNamespace(namespace, hostNamespace):
    names = set(namespace.iterkeys()) & set(hostNamespace.iterkeys())
    for n in names:
        if namespace[n] is hostNamespace[n]:
            del namespace[n]

