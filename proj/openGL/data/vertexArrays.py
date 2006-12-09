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
from .glArrayDataType import GLArrayDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Utility functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def blend(u0, u1, a):
    amat = numpy.asarray([a, a])
    amat[0] = 1-amat[0]
    return numpy.dot(amat.T, [u0, u1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Data Arrays
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DataArrayBase(GLArrayBase):
    gldtype = GLArrayDataType()

    def blend(self, other, alpha, copy=True):
        r = blend(self, other, alpha)
        if copy: return r

        self[:] = r
        return self

    def get(self, at=Ellipsis):
        return self[at]
    def set(self, data, at=Ellipsis, fill=0):
        l = numpy.shape(data)
        if not l:
            # fill with data
            self[at] = data
        else:
            l = l[-1]
            self[at,:l] = data
            self[at,l:] = fill
        return self
    def setPart(self, data, at=Ellipsis):
        l = numpy.shape(data)
        if not l:
            # fill with data
            self[at] = data
        else:
            self[at,:l] = data
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _formatXformMap = {}
    def fillFrom(self, value=GLArrayBase.useDefault):
        value = self._fillValueFrom(value)
        self.xformFrom(value)

    def xformAs(self, other, at=Ellipsis):
        if isinstance(other, (basestring, tuple)):
            other = self.fromShape(self[at].shape[:-1], other, value=None, completeShape=False)
        elif isinstance(other, numpy.dtype):
            other = self.fromShape(self[at].shape, other, value=None, completeShape=True)
        else:
            other = self.fromShape(other.shape[-1:], other.dtype, value=None, completeShape=True)

        return other.xformFrom(self, at)

    def xformFrom(self, other, at=Ellipsis):
        schar = self.dtype.base.char; sshape = self.shape[-1]
        ochar = other.dtype.base.char; oshape = other.shape[-1]
        mshape = min(sshape, oshape)

        scale = self._formatXformMap.get((ochar, schar), 1)
        if scale != 1:
            if self.itemsize > other.itemsize:
                self[at, :mshape] = other[at, :mshape]
                self[at, :mshape] *= scale
            else:
                self[at, :mshape] = scale * other[at, :mshape]
        else:
            self[at, :mshape] = other[at, :mshape]

        if sshape > oshape:
            vmax = self._formatXformMap['f', other.dtype.base.char]
            self[at, oshape:] = vmax

        return self[at]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexArray(DataArrayBase):
    default = numpy.array([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (2,3,4), default='3f')
    glinfo = gldtype.arrayInfoFor('vertex')

class TextureCoordArray(DataArrayBase):
    default = numpy.array([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (1,2,3,4), default='3f')
    glinfo = gldtype.arrayInfoFor('texture_coord')

class MultiTextureCoordArray(TextureCoordArray):
    gldtype = TextureCoordArray.gldtype.copy()
    glinfo = gldtype.arrayInfoFor('multi_texture_coord')

class NormalArray(DataArrayBase):
    default = numpy.array([0, 0, 1], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('bhlifd', (3,), default='3f')
    glinfo = gldtype.arrayInfoFor('normal')

class ColorArray(DataArrayBase):
    default = numpy.array([1., 1., 1., 1.], 'f')

    _formatXformMap = {
        ('f', 'f'): 1., ('d', 'd'): 1.,
        ('f', 'd'): 1., ('d', 'f'): 1.,

        ('f', 'B'): 0xff, ('f', 'b'): 0x7f,
        ('d', 'B'): 0xff, ('d', 'b'): 0x7f,
        ('f', 'H'): 0xffff, ('f', 'h'): 0x7fff,
        ('d', 'H'): 0xffff, ('d', 'h'): 0x7fff,
        ('f', 'L'): 0xffffffff, ('f', 'l'): 0x7fffffff,
        ('f', 'I'): 0xffffffff, ('f', 'i'): 0x7fffffff,
        ('d', 'L'): 0xffffffff, ('d', 'l'): 0x7fffffff,
        ('d', 'I'): 0xffffffff, ('d', 'i'): 0x7fffffff,

        ('B', 'b'): 0.5, ('b', 'B'): 2,
        ('H', 'h'): 0.5, ('h', 'H'): 2,
        ('L', 'l'): 0.5, ('l', 'L'): 2,
        ('L', 'i'): 0.5, ('i', 'L'): 2,
        ('I', 'i'): 0.5, ('i', 'I'): 2,

        ('l', 'i'): 1, ('i', 'l'): 1,
        ('L', 'I'): 1, ('I', 'L'): 1,

        ('b', 'h'): 0x0101,     ('h', 'b'): 1./0x0100,
        ('b', 'l'): 0x01010101, ('l', 'b'): 1./0x01000000,
        ('b', 'i'): 0x01010101, ('i', 'b'): 1./0x01000000,
        ('h', 'l'): 0x00010001, ('l', 'h'): 1./0x00010000,
        ('h', 'i'): 0x00010001, ('i', 'h'): 1./0x00010000,

        ('B', 'H'): 0x0101,     ('H', 'B'): 1./0x0100,
        ('B', 'L'): 0x01010101, ('L', 'B'): 1./0x01000000,
        ('B', 'I'): 0x01010101, ('I', 'B'): 1./0x01000000,
        ('H', 'L'): 0x00010001, ('L', 'H'): 1./0x00010000,
        ('H', 'I'): 0x00010001, ('I', 'H'): 1./0x00010000,

        ('b', 'H'): 0x0101<<1,     ('H', 'b'): 1./(0x0100<<1),
        ('b', 'L'): 0x01010101<<1, ('L', 'b'): 1./(0x01000000<<1),
        ('b', 'I'): 0x01010101<<1, ('I', 'b'): 1./(0x01000000<<1),
        ('h', 'L'): 0x00010001<<1, ('L', 'h'): 1./(0x00010000<<1),
        ('h', 'I'): 0x00010001<<1, ('I', 'h'): 1./(0x00010000<<1),

        ('B', 'h'): 0x0101>>1,     ('h', 'B'): 1./(0x0100<<1),
        ('B', 'l'): 0x01010101>>1, ('l', 'B'): 1./(0x01000000<<1),
        ('B', 'i'): 0x01010101>>1, ('i', 'B'): 1./(0x01000000<<1),
        ('H', 'l'): 0x00010001>>1, ('l', 'H'): 1./(0x00010000<<1),
        ('H', 'i'): 0x00010001>>1, ('i', 'H'): 1./(0x00010000<<1),
        }
    for (l,r), v in _formatXformMap.items():
        if (r,l) not in _formatXformMap:
            _formatXformMap[(r,l)] = 1./v

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,4), default='4f')
    glinfo = gldtype.arrayInfoFor('color')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    hexFormatMap = {}
    for n in range(1, 5):
        hexFormatMap[0, n] = (n, 1)
        hexFormatMap[n, n] = (n, 1)
        hexFormatMap[n, 2*n] = (n, 2)
        hexFormatMap[0, 2*n] = (n, 2)

    @classmethod
    def fromHex(klass, value, hexFormatMap=hexFormatMap):
        if value[:1] == '#':
            value = value[1:]
        else: raise ValueError("Unsure of the color value string format: %r" % (value,))

        value = value.strip().replace(' ', ':').replace(',', ':')
        components = value.count(':') 
        if components:
            # add one if no trailing : is found
            components += (not value.endswith(':')) 
            value = value.replace(':', '')
        else: components = 0

        n, f = hexFormatMap[components, len(value)]
        if f == 2: 
            result = tuple(int(e, 16) for e in (value[0:2], value[2:4], value[4:6], value[6:8]) if e)
        elif f == 1: 
            result = tuple(int(e+e, 16) for e in (value[0:1], value[1:2], value[2:3], value[3:4]) if e)
        else: 
            raise ValueError("Do not know how to interpret %r as a color with %r components" % (value, components))

        if len(result) == 1:
            result = result * 4
        elif len(result) == 2:
            result = result[:-1] * 3 + result[-1:]

        return klass.fromDataRaw(result, '%dB' % len(result))
    del hexFormatMap

    #~ remap some setters to be able to understand hex values

    def set(self, data, at=Ellipsis, fill=0):
        if isinstance(data, basestring):
            data = self.fromHex(data).xformAs(self)
        return DataArrayBase.set(self, data, at, fill=0)
    def setPart(self, data, at=Ellipsis):
        if isinstance(data, basestring):
            data = self.fromHex(data).xformAs(self)
        return DataArrayBase.setPart(self, data, at)
    def __setslice__(self, i, j, value):
        if isinstance(value, basestring):
            value = self.fromHex(value).xformAs(self)
        return DataArrayBase.__setslice__(self, i, j, value)
    def __setitem__(self, index, value):
        if isinstance(value, basestring):
            value = self.fromHex(value).xformAs(self)
        return DataArrayBase.__setitem__(self, index, value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SecondaryColorArray(ColorArray):
    default = numpy.array([1., 1., 1.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,), default='3f')
    glinfo = gldtype.arrayInfoFor('secondary_color')

class ColorIndexArray(DataArrayBase):
    default = numpy.array([0], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('Bhlifd', (1,), default='1B')
    glinfo = gldtype.arrayInfoFor('color_index')

class FogCoordArray(DataArrayBase):
    default = numpy.array([0.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('fd', (1,), default='1f')
    glinfo = gldtype.arrayInfoFor('fog_coord')

class EdgeFlagArray(DataArrayBase):
    default = numpy.array([1], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('B', (1,), default='1B')
    glinfo = gldtype.arrayInfoFor('edge_flag')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = sorted(name for name, value in vars().items() if isinstance(value, type) and issubclass(value, GLArrayBase))
__all__.append('blend')

