#!/usr/bin/env python
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

from raw import freetype as FT
from ctypes import byref, cast, c_void_p

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FreetypeLibrary(object):
    _as_parameter_ = None
    _as_parameter_type_ = FT.FT_Library
    _ft_init = staticmethod(FT.FT_Init_FreeType)
    _ft_done = FT.FT_Done_FreeType
    _ft_version = FT.FT_Library_Version

    singleton = None

    def __new__(klass, useSingleton=True, *args, **kw):
        if not useSingleton or klass.singleton is None:
            self = object.__new__(klass, *args, **kw)
            klass.singleton = self
        else: 
            self = klass.singleton
        return self

    def __init__(self):
        self._as_parameter_ = self._as_parameter_type_()
        self._ft_init(byref(self._as_parameter_))

    def __del__(self):
        if self._as_parameter_ is not None:
            self._ft_done()
        self._as_parameter_ = None

    def getVersion(self):
        major = FT.FT_Int(); minor = FT.FT_Int(); patch = FT.FT_Int()
        self._ft_version(byref(major), byref(minor), byref(patch))
        return (major.value, minor.value, patch.value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    print '%s.%s.%s' % library.getVersion()

