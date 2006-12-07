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

from .gl import glGetError
from .glu import gluErrorString

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLError(Exception):
    gluErrorString = staticmethod(gluErrorString)
    glGetError = staticmethod(glGetError)

    fmt = '%s (0x%x)'

    def __init__(self, error, callInfo=None):
        self.error = error
        self.errorString = self.gluErrorString(error)
        Exception.__init__(self, self.fmt % (errorString, error, ))

    @classmethod
    def check(klass, err=None, doRaise=True):
        if err is None:
            err = klass.glGetError()

        if doRaise and err != 0:
            raise klass(err)
        else:
            return ((err != 0), err)

