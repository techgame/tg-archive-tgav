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

import numpy

from .glArrayBase import GLArrayBase
from .glArrayDataType import GLElementArrayDataType, GLElementRangeDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Element Arrays
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ElementArray(GLArrayBase):
    default = numpy.array([0], 'H')

    gldtype = GLElementArrayDataType()
    gldtype.addFormatGroups('BHLI', (1,))
    gldtype.setDefaultFormat('1H')
    glinfo = gldtype.arrayInfoFor('element_array')

class ElementRange(GLArrayBase):
    default = numpy.array([0, 0], 'L')

    gldtype = GLElementRangeDataType()
    gldtype.addFormatGroups('LI', (2,))
    gldtype.setDefaultFormat('2I')
    glinfo = gldtype.arrayInfoFor('element_range')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = sorted(name for name, value in vars().items() if isinstance(value, type) and issubclass(value, GLArrayBase))

