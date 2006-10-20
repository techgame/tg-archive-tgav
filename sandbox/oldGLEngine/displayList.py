##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import time
from OpenGL import GL

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DisplayList(object):
    _recreateCache = 1
    _listId = None

    def glExecute(self, context):
        if self._recreateCache:
            if __debug__:
                print "Caching Display List %r..." % self.__class__.__name__,
                startTime = time.clock()
            if self._listId is None:
                self._listId = GL.glGenLists(1)
            GL.glNewList(self._listId, GL.GL_COMPILE)
            super(DisplayList, self).GLExecute(context)
            GL.glEndList()
            if __debug__: 
                endTime = time.clock()
                print "Done Caching Display List (%1.4f)" % (endTime-startTime)
            self._recreateCache = 0

        if self._listId is not None:
            GL.glCallList(self._listId)

