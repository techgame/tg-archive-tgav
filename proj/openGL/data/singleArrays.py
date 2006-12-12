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

from numpy import atleast_1d
from vertexArrays import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Synonyms for (implied) single entry
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Vertex(VertexArray): 
    _atleast_nd = staticmethod(atleast_1d)
class Vector(VectorArray):
    _atleast_nd = staticmethod(atleast_1d)
class TextureCoord(TextureCoordArray):
    _atleast_nd = staticmethod(atleast_1d)
class MultiTextureCoord(MultiTextureCoordArray):
    _atleast_nd = staticmethod(atleast_1d)
class Normal(NormalArray):
    _atleast_nd = staticmethod(atleast_1d)
class Color(ColorArray):
    _atleast_nd = staticmethod(atleast_1d)
class SecondaryColor(SecondaryColorArray):
    _atleast_nd = staticmethod(atleast_1d)
class ColorIndex(ColorIndexArray):
    _atleast_nd = staticmethod(atleast_1d)
class FogCoord(FogCoordArray):
    _atleast_nd = staticmethod(atleast_1d)
class EdgeFlag(EdgeFlagArray):
    _atleast_nd = staticmethod(atleast_1d)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = sorted(name for name, value in vars().items() if not name.endswith('Array') and isinstance(value, type) and issubclass(value, GLArrayBase))

