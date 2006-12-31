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
from numpy import ndarray
from .glArrayBase import GLArrayBase

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorFormatMixin(object):
    def __setslice__(self, i, j, value):
        value = self._valueFrom(value, self.edtype)
        ndarray.__setslice__(self, i, j, value)
    def __setitem__(self, index, value):
        value = self._valueFrom(value, self.edtype)
        ndarray.__setitem__(self, index, value)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    @classmethod
    def xform(klass, src, dst):
        if isinstance(dst, (basestring, tuple, numpy.dtype)):
            dst = klass.fromShape(src.shape[:-1], dst)
        elif not isinstance(dst, ndarray):
            raise TypeError("Dst parameter expected to be an ndarray or dtype")

        srcFormatChar = src.dtype.base.char
        srcEShape = src.shape[-1]

        dstFormatChar = dst.dtype.base.char
        dstEShape = dst.shape[-1]

        minEShape = min(dstEShape, srcEShape)

        scale = klass._formatXformMap.get((srcFormatChar, dstFormatChar), 1)

        tmpDst = dst.view(ndarray)
        tmpSrc = src.view(ndarray)
        if scale != 1:
            if tmpDst.itemsize > tmpSrc.itemsize:
                tmpDst[..., :minEShape] = tmpSrc[..., :minEShape]
                tmpDst[..., :minEShape] *= scale
            else:
                tmpDst[..., :minEShape] = scale * tmpSrc[..., :minEShape]
        else:
            tmpDst[..., :minEShape] = tmpSrc[..., :minEShape]

        if dstEShape > srcEShape:
            # fill in the last value
            vmax = klass._formatXformMap['f', srcFormatChar]
            tmpDst[..., srcEShape:] = vmax

        return dst

    def xformAs(self, other):
        return self.xform(self, other)

    def xformFrom(self, other):
        return self.xform(other, self)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def _valueFrom(klass, value, edtype):
        if isinstance(value, ndarray):
            if value.dtype != edtype[0]:
                return klass.xform(value, edtype)
            else:
                return value

        if isinstance(value, basestring):
            return klass.fromHex(value, edtype[0], edtype[1])

        if (isinstance(value, (list, tuple)) and value and 
                (isinstance(value[0], basestring))):
            return klass.fromHex(value, edtype[0], edtype[1])

        value = super(ColorFormatMixin, klass)._valueFrom(value, edtype)
        return value

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromHex(klass, hexData, dtype=None, shape=None):
        colors = klass.fromHexRaw(hexData)

        if dtype is None and not shape:
            return colors

        colors = colors.xformAs((dtype, shape))
        return colors

    #~ raw color transformations ~~~~~~~~~~~~~~~~~~~~~~~~

    hexFormatMap = {}
    for n in range(1, 5):
        hexFormatMap[0, n] = (n, 1)
        hexFormatMap[n, n] = (n, 1)
        hexFormatMap[n, 2*n] = (n, 2)
        hexFormatMap[0, 2*n] = (n, 2)
    hexFormatMap[0, 2] = (1, 2)

    remapNto4 = {
        1: (lambda r: r*3+(0xff,)),
        2: (lambda r: r[:-1]*3 + r[-1:]),
        3: (lambda r: r+(0xff,)),
        4: (lambda r: r),
        }
    hexComponentRemap = (lambda r, remapNto4=remapNto4: remapNto4[len(r)](r))
    del remapNto4

    @classmethod
    def fromHexRaw(klass, hexColorData, hexFormatMap=hexFormatMap, hexComponentRemap=hexComponentRemap):
        if isinstance(hexColorData, basestring):
            hexColorData = [hexColorData]

        colorResult = klass.fromShape((len(hexColorData),-1), '4B')
        for i in xrange(len(colorResult)):
            value = hexColorData[i]

            if value[:1] == '#':
                value = value[1:]
            else: raise ValueError("Unsure of the color value string format: %r" % (value,))

            value = value.strip().replace(' ', ':').replace(',', ':')
            components = value.count(':') 
            if components:
                # add one if no trailing : is found
                components += (not value.endswith(':')) 
                value = value.split(':')
                totalSize = sum(len(e) for e in value)
            else: 
                components = 0
                totalSize = len(value)

            n, f = hexFormatMap[components, totalSize]
            if not components:
                if f == 2: value = (value[0:2], value[2:4], value[4:6], value[6:8])
                elif f == 1: value = (value[0:1], value[1:2], value[2:3], value[3:4])
                else: raise ValueError("Do not know how to interpret %r as a color with %r components" % (value, components))

            if f == 2: 
                result = tuple(int(e, 16) for e in value if e)
            elif f == 1: 
                result = tuple(int(e, 16)*17 for e in value if e)

            result = hexComponentRemap(result)
            colorResult[i] = result
        return klass._normalized(colorResult)

    del hexFormatMap
    del hexComponentRemap

