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
from numpy import asarray, float32
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
        self.select = partial(glColorFn, value.ctypes.data_as(glColorFn.api.argtypes[0]))
    def delValue(self):
        self.setValue(None)
    value = property(getValue, setValue, delValue)

    def convertValue(self, value, dataFormat):
        if isinstance(value, (long, int)):
            if dataFormat in ('h', 'hex', 'rgb'):
                value = (v/255. for v in ((value >> 16) & 0xff, (value >> 8) & 0xff, (value) & 0xff))
            elif dataFormat in ('ha', 'hexalpha', 'hexa', 'rgba'):
                value = (v/255. for v in ((value >> 32) & 0xff, (value >> 16) & 0xff, (value >> 8) & 0xff, (value) & 0xff))
            else: 
                if value > 255:
                    raise ValueError("Do not know how to interpret %r as a color with format %r" % (value, dataFormat))
                if value > 1:
                    value /= 255.
                value = (value, value, value)

        elif isinstance(value, float):
            if dataFormat in ('i', 'I', 'la', 'LA', 'rgba', 'RGBA', 'a', 'A'):
                value = (value, value, value, value)
            elif dataFormat in ('l', 'L', 'rgb', 'RGB'):
                value = (value, value, value)
            else: 
                value = (value, value, value)

        elif isinstance(value, basestring):
            value = value.replace(' ', '')
            if value.startswith('#'):
                value = value[1:]
            elif value.startswith('0x'):
                value = value[2:]

            if len(value) > 4:
                # 6 or 8 length hex string
                value = tuple(int(e, 16)/255. for e in (value[0:2], value[2:4], value[4:6], value[6:]) if e)
            elif len(value) > 3:
                # 3 or 4 length hex string
                value = tuple(int(e, 16)/15. for e in (value[0:1], value[1:2], value[2:3], value[3:]) if e)
            else:
                raise ValueError("Do not know how to interpret %r as a color with format %r" % (value, dataFormat))

        return value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorProperty(object):
    ColorFactory = ColorObject

    def __init__(self, *argsColor, **kwColor):
        self.argsColor = argsColor
        self.kwColor = kwColor

    def __get__(self, obj, klass):
        if obj is None:
            return self

        try:
            return obj.__value
        except AttributeError:
            return self.create(obj, *self.argsColor, **self.kwColor)

    def __set__(self, obj, value):
        try:
            colorObj = obj.__value
        except AttributeError:
            colorObj = self.create(obj, value)
        else:
            colorObj.setValue(value)

    def __delete__(self, obj):
        try:
            colorObj = obj.__value
        except AttributeError:
            pass
        else:
            colorObj.delValue()

    def create(self, obj, *args, **kw):
        colorObj = self.ColorFactory(*args, **kw)
        obj.__value = colorObj
        return colorObj

