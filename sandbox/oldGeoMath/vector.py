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

import sys
import math
import numpy

from TG.common.properties import CopyProperty, PropertyFactory

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dot = numpy.dot
cross = numpy.cross

def crossMag(v0, v1):
    cp = cross(v0, v1) 
    mag = numpy.sqrt(dot(cp, cp))
    return cp, 

def dotCross(v0, v1):
    return numpy.dot(v0, v1), numpy.cross(v0, v1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def linearMapping((lowFrom, hiFrom), (lowTo, hiTo), aslambda=True):
    """
    >>> linearMapping((10.,20.), (-1.,1.), False)
    (0.20000000000000001, -3.0)
    >>> linearMapping((-1.,1.), (10.,20.), False)
    (5.0, 15.0)
    """
    wr = (hiTo - lowTo)/(hiFrom - lowFrom)
    tr = lowTo - lowFrom*wr
    if aslambda:
        return lambda r: wr*(r-lowFrom)+lowTo
    else: 
        return (wr, lowTo - lowFrom*wr)

def linearDimMapping((lowFrom, dimFrom), (lowTo, dimTo), aslambda=True):
    """
    >>> linearDimMapping((10.,10.), (-1.,2.), False)
    (0.20000000000000001, -3.0)
    >>> linearDimMapping((-1.,2.), (10.,10.), False)
    (5.0, 15.0)
    """
    wr = dimTo/dimFrom
    if aslambda:
        return lambda r: wr*(r-lowFrom)+lowTo
    else: 
        return (wr, lowTo - lowFrom*wr)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class IdxProp(object):
    def __init__(self, idx):
        self.idx = idx
        
    def __get__(self, obj, klass):
        if obj is not None:
            return obj[self.idx]
        else: return klass

    def __set__(self, obj, value):
        obj[self.idx] = value

    def __delete__(self, obj):
        del obj[self.idx]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Vector definition
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VectorProperty(CopyProperty):
    def __init__(self, PropClass, *args, **kw):
        self.PropClass = PropClass
        self.kw = kw
        value = PropClass(*args, **kw)
        CopyProperty.__init__(self, value)

    def __set__(self, obj, value):
        if not isinstance(value, self.PropClass):
            value = self.PropClass(value, **self.kw)
            value.matchLen(self.startValue)
        CopyProperty.__set__(self, obj, value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Vector(PropertyFactory):
    """
    >>> x = Vector(1., 0., 0.); x
    <1.0, 0.0, 0.0>
    >>> y = Vector(0., 1., 0.); y
    <0.0, 1.0, 0.0>
    >>> z = Vector(0., 0., 1.); z
    <0.0, 0.0, 1.0>
    >>> xyz = Vector(x + y + z); xyz
    <1.0, 1.0, 1.0>
    >>> Vector.cross(x,y)
    <0.0, 0.0, 1.0>
    >>> Vector.cross(x,z)
    <0.0, -1.0, 0.0>
    >>> Vector.cross(y,z)
    <1.0, 0.0, 0.0>
    >>> Vector.cross(y,x)
    <0.0, 0.0, -1.0>
    >>> Vector.cross(z,y)
    <-1.0, 0.0, 0.0>
    >>> Vector.cross(z,z)
    <0.0, 0.0, 0.0>
    >>> Vector.cross(z,x)
    <0.0, 1.0, 0.0>
    >>> Vector.cross(x,x)
    <0.0, 0.0, 0.0>
    >>> Vector.cross(y,y)
    <0.0, 0.0, 0.0>
    >>> '%1.8f' % x.degreesTo(y)
    '90.00000250'
    >>> '%1.8f' % x.degreesTo(z)
    '90.00000250'
    >>> '%1.8f' % y.degreesTo(z)
    '90.00000250'
    >>> '%1.8f' % x.angleTo(z)
    '1.57079637'
    >>> '%1.8f' % x.angleTo(y)
    '1.57079637'
    >>> '%1.8f' % z.angleTo(y)
    '1.57079637'
    >>> x * 4
    <4.0, 0.0, 0.0>
    >>> x / 4.
    <0.25, 0.0, 0.0>
    >>> x + 1.
    <2.0, 1.0, 1.0>
    >>> x - 1.
    <0.0, -1.0, -1.0>
    >>> x + y
    <1.0, 1.0, 0.0>
    >>> x - y
    <1.0, -1.0, 0.0>
    >>> x * y
    0.0
    >>> x * y.cross(z)
    1.0
    >>> x * z.cross(y)
    -1.0
    >>> abs(4*x), abs(3*y), abs(2*z)
    (4.0, 3.0, 2.0)
    >>> -x
    <-1.0, -0.0, -0.0>
    >>> x = Vector(1.,1.,1.); x
    <1.0, 1.0, 1.0>
    >>> x.normalize(); x
    <0.577350, 0.577350, 0.577350>
    >>> x = Vector(2.,3.,4.); x
    <2.0, 3.0, 4.0>
    >>> x.get()
    array([ 2.,  3.,  4.], dtype=float32)
    >>> x.getH()
    array([ 2.,  3.,  4.,  1.], dtype=float32)
    """

    PropertyClass = VectorProperty

    x = IdxProp(0)
    y = IdxProp(1)
    z = IdxProp(2)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = numpy.Float32
    _array = numpy.asarray([], atype)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Special 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, *args, **kw):
        atype = kw.pop('atype', None)
        if atype is not None and atype != self.atype: 
            self.atype = atype

        if not args:
            pass
        elif isinstance(args[0], Vector):
            self.set(args[0])
        elif isinstance(args[0], (tuple, list, numpy.ArrayType)):
            self.set(args[0])
        else:
            self.set(args)

    def __repr__(self):
        return "<%s>" % ', '.join([str(f)[:8] for f in self.tolist()])

    def fromV(self, vector):
        klass = self.__class__
        return klass(vector, atype=self.atype)

    def copy(self):
        return self.fromV(self)

    def get(self):
        return self._array
    def set(self, other):
        self._array = self._simplify(other)

    def getH(self):
        return numpy.r_[self.get(), 1]
    def setH(self, otherH):
        self._array = otherH[:-1]
        self._array /= otherH[-1]

    def matchLen(self, other):
        lo = len(other)  
        value = self._array
        lv = len(value)

        value = numpy.r_[value[:lo], other[lv:]]
        self.set(value)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __len__(self): 
        return len(self._array)
    def __getitem__(self, key): 
        return self._array[key]
    def __setitem__(self, key, value): 
        self._array[key] = value

    def __pos__(self): 
        return self
    def __neg__(self): 
        return self.fromV(-self._array)
    def __abs__(self):
        return numpy.sqrt(numpy.dot(self._array, self._array))

    def __iadd__(self, other): 
        if isinstance(other, Vector): other = other._array
        self.set(self._array + other)
        return self
    def __add__(self, other): 
        if isinstance(other, Vector): other = other._array
        return self.fromV(self._array + other)
    def __radd__(self, other): 
        if isinstance(other, Vector): other = other._array
        return self.fromV(other + self._array)
    def __isub__(self, other): 
        if isinstance(other, Vector): other = other._array
        self.set(self._array - other)
        return self
    def __sub__(self, other): 
        if isinstance(other, Vector): other = other._array
        return self.fromV(self._array - other)
    def __rsub__(self, other): 
        if isinstance(other, Vector): other = other._array
        return self.fromV(other - self._array)
        
    def __idiv__(self, other):
        if isinstance(other, Vector):
            raise TypeError, "can't divide a Vector by a Vector"
        else:
            self.set(self._array / other)
        return self
    def __div__(self, other):
        if isinstance(other, Vector):
            raise TypeError, "can't divide a Vector by a Vector"
        else: return self.fromV(self._array / other)

    def __imul__(self, other):
        if isinstance(other, Vector):
            raise TypeError, "can't multiply by a Vector in-place"
        else:
            self.set(self._array * other)
        return self

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot(other)
        else: return self.fromV(self._array * other)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            return other.dot(self)
        else: return self.fromV(other * self._array)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def asarray(self, *args, **kw):
        if args or kw: return numpy.asarray(self._array, *args, **kw)
        else: return self._array

    def asarrayH(self, *args, **kw):
        return numpy.r_[self.asarray(*args, **kw), 1]

    def tolist(self):
        return self._array.tolist()

    def dot(self, other):
        return numpy.dot(self._array, other._array)

    def squared(self):
        return numpy.dot(self._array, self._array)

    def crossUpdate(self, other):
        self[:] = numpy.cross(self._array, other._array)

    def cross(self, other):
        return self.fromV(numpy.cross(self._array, other._array))

    def crossMag(self, v1):
        cp = numpy.cross(self._array, other._array) 
        mag = numpy.sqrt(numpy.dot(cp, cp))
        return cp, mag

    def dotCross(self, other):
        v0 = self._array; v1 = other._array
        return numpy.dot(v0, v1), numpy.cross(v0, v1)

    def normalize(self):
        self /= abs(self)
    def normalized(self):
        return self / abs(self)

    def degreesTo(self, other):
        return math.degrees(self.angleTo(other))

    def angleTo(self, other):
        return numpy.arccos(self.dot(other))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Protected Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _simplify(self, value):
        r = numpy.asarray(value, self.atype)
        return r

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class UnitVector(Vector):
    def __setitem__(self, key, value): 
        a = self._array
        a[key] = value
        a /= a[-1:]
        a /= numpy.sqrt(numpy.dot(a, a))

    def _simplify(self, value):
        r = numpy.asarray(value, self.atype)
        r /= numpy.sqrt(numpy.dot(r, r))
        return r

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorVector(Vector):
    atype = numpy.UInt8

    r = IdxProp(0)
    g = IdxProp(1)
    b = IdxProp(2)
    a = IdxProp(3)

class ColorFloatVector(ColorVector):
    """
    >>> a = ColorVector(1,2,3,4); a
    <1, 2, 3, 4>
    >>> a.r, a.g, a.b, a.a
    (1, 2, 3, 4)
    """

    atype = numpy.Float32

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Testing 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    print "Testing..."
    import doctest, vector
    doctest.testmod(vector)
    print "Test complete."

