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

from OpenGL import GL
from changeBaseMgr import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BufferChangeElement(BitmaskChangeElement):
    """Encapsulates a single collection of buffer attribute changes"""

class DynamicBufferChangeElement(DynamicBitmaskChangeElement):
    """Encapsulates a single collection of attribute changes"""

class BufferTracker(BitmaskChangeTracker):
    _elementAttributeName = 'BufferClear'
    
class DynamicBufferTracker(DynamicBitmaskChangeTracker):
    _elementAttributeName = 'BufferClear'

class BufferEffector(BufferTracker):
    def glExecute(self, context):
        GL.glClear(self.bitmask)

class DynamicBufferEffector(DynamicBufferTracker):
    def glExecute(self, context):
        GL.glClear(self.bitmask)

