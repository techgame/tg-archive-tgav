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

from numpy import atleast_1d, shape
from vertexArrays import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Synonyms for (implied) single entry
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _SingleMixin(object):
    __array_priority__ = 15.0

    @classmethod
    def _normalized(klass, result):
        return atleast_1d(result.squeeze())

    def get(self, at=Ellipsis):
        return self[at]
    def set(self, data, at=Ellipsis, fill=0):
        l = shape(data)
        if not l:
            # data is a single object, so broadcast set it
            self[at] = data
            return self

        l = min(l[-1], self.shape[-1])
        self[at,:l] = data[:l]
        return self

    def setPropValue(self, value):
        return self.set(value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Vertex(_SingleMixin, VertexArray): 
    pass
class Vector(_SingleMixin, VectorArray):
    pass
class TextureCoord(_SingleMixin, TextureCoordArray):
    pass
TexCoord = TextureCoord
class MultiTextureCoord(_SingleMixin, MultiTextureCoordArray):
    pass
MultiTexCoord = MultiTextureCoord
class Normal(_SingleMixin, NormalArray):
    pass
class Color(_SingleMixin, ColorArray):
    pass
class SecondaryColor(_SingleMixin, SecondaryColorArray):
    pass
class ColorIndex(_SingleMixin, ColorIndexArray):
    pass
class FogCoord(_SingleMixin, FogCoordArray):
    pass
class EdgeFlag(_SingleMixin, EdgeFlagArray):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = [
    'Vertex', 'Vector',
    'TexCoord', 'TextureCoord', 'MultiTextureCoord', 'MultiTexCoord',
    'Normal', 'Color', 'SecondaryColor', 'ColorIndex', 'FogCoord',
    'EdgeFlag', ]

