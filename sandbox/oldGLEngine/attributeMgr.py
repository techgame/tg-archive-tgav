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
from ChangeBaseMgr import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AttributeChangeElement(BitmaskChangeElement):
    """Encapsulates a single collection of attribute changes"""

class DynamicAttributeChangeElement(DynamicBitmaskChangeElement):
    """Encapsulates a single collection of attribute changes"""

class AttributeTracker(DynamicBitmaskChangeTracker):
    _elementAttributeName = 'AttributeChange'

class AttributeEffector(AttributeTracker):
    def sequenceAdd(self, sequence):
        DynamicBitmaskChangeTracker.SequenceAdd(self, sequence)
        sequence.onBeginExecute.Add(self.glSelect)
        sequence.onEndExecute.Add(self.glDeselect)

    def glSelect(self, context):
        GL.glPushAttrib(self.Bitmask)
        
    def glDeselect(self, context):
        GL.glPopAttrib()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Client Attributes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ClientAttributeChangeElement(BitmaskChangeElement):
    """Encapsulates a single collection of client attribute changes"""

class DynamicClientAttributeChangeElement(DynamicBitmaskChangeElement):
    """Encapsulates a single collection of client attribute changes"""

class ClientAttributeTracker(DynamicBitmaskChangeTracker):
    _elementAttributeName = 'ClientAttributeChange'

class ClientAttributeEffector(ClientAttributeTracker):
    def sequenceAdd(self, sequence):
        DynamicBitmaskChangeTracker.SequenceAdd(self, sequence)
        sequence.onBeginExecute.Add(self.glSelect)
        sequence.onEndExecute.Add(self.glDeselect)

    def glSelect(self, context):
        GL.glPushClientAttrib(self.bitmask)
        
    def glDeselect(self, context):
        GL.glPopClientAttrib()

