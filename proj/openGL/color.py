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

from functools import partial 
from .data.vertexArrays import ColorArray
from .data.arrayViews import ColorArrayView
from raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

glColorNoOp = object # fast noop

class ColorObject(object):
    glColorFnMap = {None: glColorNoOp, 0: glColorNoOp, 3: gl.glColor3fv, 4: gl.glColor4fv}
    select = None # replaced with a fn from glColorFnMap to "select" the color

    def __init__(self, value=None, dataFormat=None):
        self.setValue(value, dataFormat)

    _value = None
    def getValue(self):
        return self._value
    def setValue(self, value, dataFormat=None):
        if value is None:
            glColorFn = self.glColorFnMap[None]
            self.select = glColorFn
            self._value = value
            return
        elif isinstance(value, tuple):
            if len(value) == 2:
                value = self.convertValue(*value)
        else:
            value = self.convertValue(value, dataFormat)

        glColorFn = self.glColorFnMap[len(value)]
        value = asarray(value, float32)
        self._value = value
        self.select = partial(glColorFn, value.ctypes.data_as(glColorFn.api.argtypes[-1]))
    def delValue(self):
        self.setValue(None)
    value = property(getValue, setValue, delValue)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorFormats(object):
    ColorArray = ColorArray

    @classmethod
    def colorFrom(klass, value, components=0):
        if isinstance(value, (list, tuple)):
            return klass.colorArrayFrom(value)[:components or None]

        elif isinstance(value, (long, int)):
            return klass.colorFromInt(value, components)[:components or None]

        elif isinstance(value, float):
            return klass.colorFromFloat(value, components)[:components or None]
        
        elif isinstance(value, basestring):
            return klass.colorFromHex(value, components)[:components or None]

        return klass.colorArrayFrom(value)[:components or None]

    @classmethod
    def colorFromInt(klass, value, components=0):
        if components in (-1,0,3):
            result = ((value >> 16) & 0xff, (value >> 8) & 0xff, (value >> 0) & 0xff)
        elif components == 4:
            result = ((value >> 32) & 0xff, (value >> 16) & 0xff, (value >> 8) & 0xff, (value >> 0) & 0xff)
        else: 
            raise ValueError("Do not know how to interpret %r as a color with %r components" % (value, components))

        return klass.colorArrayFrom(result, 'B')

    @classmethod
    def colorFromFloat(klass, value, components=0):
        result = (value,)*max(1, components)
        return klass.colorArrayFrom(result, 'f')

    hexFormatMap = {}
    for n in range(1, 5):
        hexFormatMap[0, n] = (n, 1)
        hexFormatMap[n, n] = (n, 1)
        hexFormatMap[n, 2*n] = (n, 2)
        hexFormatMap[0, 2*n] = (n, 2)

    @classmethod
    def colorFromHex(klass, value, components=0, hexFormatMap=hexFormatMap):
        if value[:1] == '#':
            value = value[1:]
        elif value[:1].isdigit() and value[1:2] == '#':
            components = int(value[0])
            value = value[2:]
        else: raise ValueError("Unsure of the color value string format: %r" % (value,))

        value = value.strip().replace(' ', ':')
        if ':' in value:
            components = value.count(':') + (not value.endswith(':')) # add one if no trailing : is found
            value = value.replace(':', '')

        n, f = hexFormatMap[max(0, components), len(value)]
        if f == 2: 
            result = tuple(int(e, 16) for e in (value[0:2], value[2:4], value[4:6], value[6:8]) if e)
        elif f == 1: 
            result = tuple(int(e+e, 16) for e in (value[0:1], value[1:2], value[2:3], value[3:4]) if e)
        else: 
            raise ValueError("Do not know how to interpret %r as a color with %r components" % (value, components))

        return klass.colorArrayFrom(result, 'B')
    del hexFormatMap

    @classmethod
    def colorArrayFrom(klass, result, dtype=None):
        if len(result) == 1:
            result = result * 4
        elif len(result) == 2:
            result = result[:-1] * 3 + result[-1:]

        return klass.ColorArray(result, dtype)

color = ColorFormats.colorFrom
hexcolor = ColorFormats.colorFromHex

