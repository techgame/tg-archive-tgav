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

@classmethod
def _normalized(klass, result):
    return atleast_1d(result.squeeze())

class Vertex(VertexArray): 
    _normalized = _normalized
class Vector(VectorArray):
    _normalized = _normalized
class TextureCoord(TextureCoordArray):
    _normalized = _normalized
TexCoord = TextureCoord
class MultiTextureCoord(MultiTextureCoordArray):
    _normalized = _normalized
MultiTexCoord = MultiTextureCoord
class Normal(NormalArray):
    _normalized = _normalized
class Color(ColorArray):
    _normalized = _normalized
class SecondaryColor(SecondaryColorArray):
    _normalized = _normalized
class ColorIndex(ColorIndexArray):
    _normalized = _normalized
class FogCoord(FogCoordArray):
    _normalized = _normalized
class EdgeFlag(EdgeFlagArray):
    _normalized = _normalized

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = sorted(name for name, value in vars().items() if not name.endswith('Array') and isinstance(value, type) and issubclass(value, GLArrayBase))

