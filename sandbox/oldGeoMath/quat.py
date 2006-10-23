##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

"""
Derived from Quaternions text and sources: 
    Jav Savarovsky, "Game Programming Gems", Section 2.7, (c) 2000
    Jason Shankel, "Game Programming Gems", Section 2.8, (c) 2000 
    Jason Shankel, "Game Programming Gems", Section 2.9, (c) 2000 
    Stan Melax, "Game Programming Gems", Section 2.10, (c) 2000 
    Donald Hearn & M. Pauline Baker, "Computer Graphics", 2nd ed., (c) 1994
    David H. Eberly, "3D Game Engine Design", (c) 2001
    Konrad Hinsen <hinsen@cnrs-orleans.fr> from his Scientific.Geometry.Quaternion package
    OpenGL.quaternion

"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import math
import numpy

from xform import Transform3dh
from vector import Vector, dotCross

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Quaternion(Transform3dh):
    """
    >>> q = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); q
    <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
    >>> r = Quaternion.fromDegreesAxis(60., (0., 1., 0.)); r
    <Quaternion: 0.866025403784 + 0.0i + 0.5j + 0.0k>
    >>> s = Quaternion(q + r); s
    <Quaternion: 1.83195123007 + 0.0i + 0.5j + 0.258819045103k>
    >>> q[0], q[1], q[2], q[3]
    (0.96592582628906831, 0.0, 0.0, 0.25881904510252074)
    >>> r[0], r[1], r[2], r[3]
    (0.86602540378443871, 0.0, 0.5, 0.0)

    Algebraic expressions quaternion OP value
    >>> q + 2.
    <Quaternion: 2.96592582629 + 2.0i + 2.0j + 2.2588190451k>
    >>> q - 2.
    <Quaternion: -1.03407417371 + -2.0i + -2.0j + -1.7411809549k>
    >>> q / 2.
    <Quaternion: 0.482962913145 + 0.0i + 0.0j + 0.129409522551k>
    >>> q * 2.
    <Quaternion: 1.93185165258 + 0.0i + 0.0j + 0.517638090205k>
    >>> 2. + q
    <Quaternion: 2.96592582629 + 2.0i + 2.0j + 2.2588190451k>
    >>> 2. - q
    <Quaternion: 1.03407417371 + 2.0i + 2.0j + 1.7411809549k>
    >>> 2. * q
    <Quaternion: 1.93185165258 + 0.0i + 0.0j + 0.517638090205k>
    >>> 2. / q
    Traceback (most recent call last):
    TypeError: unsupported operand type(s) for /: 'float' and 'Quaternion'

    Algebraic expressions quaternion OP quaternion
    >>> q + r
    <Quaternion: 1.83195123007 + 0.0i + 0.5j + 0.258819045103k>
    >>> r + q
    <Quaternion: 1.83195123007 + 0.0i + 0.5j + 0.258819045103k>
    >>> q - r
    <Quaternion: 0.0999004225046 + 0.0i + -0.5j + 0.258819045103k>
    >>> r - q
    <Quaternion: -0.0999004225046 + 0.0i + 0.5j + -0.258819045103k>
    >>> q * r
    <Quaternion: 0.836516303738 + -0.129409522551i + 0.482962913145j + 0.224143868042k>
    >>> r * q
    <Quaternion: 0.836516303738 + 0.129409522551i + 0.482962913145j + 0.224143868042k>
    >>> r / q
    Traceback (most recent call last):
    TypeError: can't divide a Quaternion by a Quaternion

    >>> +q
    <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
    >>> -q
    <Quaternion: -0.965925826289 + -0.0i + -0.0j + -0.258819045103k>
    >>> abs(q), abs(r)
    (1.0, 1.0)
    >>> abs(q + r)
    1.9165157467330176

    >>> from xform import Rotate
    >>> data = [[1,0,0],[0,1,0],[0,0,1]]
    >>> r = Rotate(20., [0,0,1]);
    >>> q = Quaternion.fromDegreesAxis(20., [0,0,1])
    >>> (abs((r*data) - (q*data)) < 1e-6).all()
    True
    >>> r = Rotate(20., [0,1,0]);
    >>> q = Quaternion.fromDegreesAxis(20., [0,1,0])
    >>> (abs((r*data) - (q*data)) < 1e-6).all()
    True
    >>> r = Rotate(20., [1,0,0]);
    >>> q = Quaternion.fromDegreesAxis(20., [1,0,0])
    >>> (abs((r*data) - (q*data)) < 1e-6).all()
    True
    >>> r = Rotate(55., [1,1,-1]);
    >>> q = Quaternion.fromDegreesAxis(55., [1,1,-1])
    >>> (abs((r*data) - (q*data)) < 1e-6).all()
    True
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = numpy.Float
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Special 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, *args):
        if not args:
            self._array = numpy.asarray((1., 0., 0., 0.), self.atype)
        elif isinstance(args[0], Quaternion):
            self._array = args[0]._array.copy()
        elif isinstance(args[0], (tuple, list, numpy.ArrayType)):
            self._array = numpy.asarray(args[0], self.atype)
        elif isinstance(args[1], (tuple, list, numpy.ArrayType)):
            self._array = numpy.asarray([args[0]] + list(args[1]), self.atype)
        else:
            self._array = numpy.asarray(args[:4], self.atype)

    def __repr__(self):
        return "<Quaternion: %s + %si + %sj + %sk>" % tuple(self._array)

    @classmethod
    def fromV(klass, *args):
        return klass(*args)

    def copy(self):
        return self.fromV(self)

    def __len__(self): return 4
    def __getitem__(self, key): return self._array[key]
    def __setitem__(self, key, value): self._array[key] = value

    def __pos__(self): 
        return self
    def __neg__(self): 
        return self.fromV(-self._array)
    def __abs__(self):
        return numpy.sqrt(numpy.dot(self._array, self._array))

    def __iadd__(self, other): 
        if isinstance(other, Quaternion): other = other._array
        self._array += other
        return self
    def __add__(self, other): 
        if isinstance(other, Quaternion): other = other._array
        return self.fromV(self._array + other)
    def __radd__(self, other): 
        if isinstance(other, Quaternion): other = other._array
        return self.fromV(other + self._array)
    def __isub__(self, other): 
        if isinstance(other, Quaternion): other = other._array
        self._array -= other
        return self
    def __sub__(self, other): 
        if isinstance(other, Quaternion): other = other._array
        return self.fromV(self._array - other)
    def __rsub__(self, other): 
        if isinstance(other, Quaternion): other = other._array
        return self.fromV(other - self._array)
        
    def __imul__(self, other):
        if isinstance(other, Quaternion):
            self._array = self._quatMult(other._array)
        elif isinstance(other, (int, float, long)):
            self._array *= other
        else: 
            self._array *= other
        return self

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            m = self._quatMult(other._array)
            return self.fromV(m)
        elif isinstance(other, Transform3dh):
            return self.composite(other)
        elif isinstance(other, (int, float, long)):
            return self.fromV(other*self._array)
        else:
            return self.xform(other)

    def __rmul__(self, other):
        if isinstance(other, Transform3dh):
            return self.rcomposite(other)
        elif isinstance(other, (int, float, long)):
            return self.fromV(other*self._array)
        else: 
            return NotImplemented

    def __div__(self, other):
        if isinstance(other, Quaternion):
            raise TypeError, "can't divide a Quaternion by a Quaternion"
        else: return self.fromV(self._array / other)
    def __idiv__(self, other):
        if isinstance(other, Quaternion):
            raise TypeError, "can't divide a Quaternion by a Quaternion"
        elif isinstance(other, (int, float, long)):
            self._array /= other
        else:
            return NotImplemented
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _quatMult(self, other):
        s0, a0, b0, c0 = self._array
        s1, a1, b1, c1 = other
        return numpy.asarray([
            s0*s1 - a0*a1 - b0*b1 - c0*c1,
            s0*a1 + a0*s1 + b0*c1 - c0*b1,
            s0*b1 - a0*c1 + b0*s1 + c0*a1,
            s0*c1 + a0*b1 - b0*a1 + c0*s1], self.atype)

    def _quatConjugate(self):
        s, a, b, c = self._array
        return numpy.asarray([s, -a, -b, -c], self.atype)

    def _quatNorm(self):
        return numpy.sqrt(numpy.dot(self._array, self._array))

    def _quatInverse(self):
        inv = -self._array
        inv[0] = -inv[0]
        inv /= numpy.dot(inv, inv)
        return inv

    def asArray4x4(self):
        """
        >>> Quaternion().asArray4x4()
        array([[ 1.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.],
               [ 0.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  1.]])
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); aa
        <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
        >>> aa.asArray4x4()
        array([[ 0.8660254, -0.5      ,  0.       ,  0.       ],
               [ 0.5      ,  0.8660254,  0.       ,  0.       ],
               [ 0.       ,  0.       ,  1.       ,  0.       ],
               [ 0.       ,  0.       ,  0.       ,  1.       ]])
        >>> aa.inversed().asArray4x4()
        array([[ 0.8660254,  0.5      ,  0.       ,  0.       ],
               [-0.5      ,  0.8660254,  0.       ,  0.       ],
               [ 0.       ,  0.       ,  1.       ,  0.       ],
               [ 0.       ,  0.       ,  0.       ,  1.       ]])
        >>> a2 = Quaternion.fromDegreesAxis(45., (1., 1., 1.)); a2
        <Quaternion: 0.923879532511 + 0.22094238269i + 0.22094238269j + 0.22094238269k>
        >>> a2.asArray4x4()
        array([[ 0.80473785, -0.31061722,  0.50587936,  0.        ],
               [ 0.50587936,  0.80473785, -0.31061722,  0.        ],
               [-0.31061722,  0.50587936,  0.80473785,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        >>> a2.inversed().asArray4x4()
        array([[ 0.80473785,  0.50587936, -0.31061722,  0.        ],
               [-0.31061722,  0.80473785,  0.50587936,  0.        ],
               [ 0.50587936, -0.31061722,  0.80473785,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        >>> a2.asInverse4x4()
        array([[ 0.80473785,  0.50587936, -0.31061722,  0.        ],
               [-0.31061722,  0.80473785,  0.50587936,  0.        ],
               [ 0.50587936, -0.31061722,  0.80473785,  0.        ],
               [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """
        s, a, b, c = self._array
        r = 2. * numpy.asarray([
                    [.5-b*b - c*c,  a*b - c*s,      c*a + b*s,      0.],
                    [ a*b + c*s,    .5-c*c - a*a,   b*c - a*s,      0.],
                    [ a*c - b*s,    b*c + a*s,      .5-b*b - a*a,   0.],
                    [0.,            0.,             0.,             .5]], self.atype)
        return r

    def asInverse4x4(self):
        return self.inversed().asArray4x4()

    def degreeAxis(self):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.))
        >>> (30 - aa.degreeAxis()[0]) < 1e-6
        True
        >>> aa.degreeAxis()[1]
        array([ 0.,  0.,  1.])
        """
        radian, axis = self.angleAxis()
        return math.degrees(radian), axis

    def angleAxis(self):
        """
        >>> ra = Quaternion.fromAngleAxis(numpy.pi/6., (0., 0., 1.))
        >>> (0.5235987 - ra.angleAxis()[0]) < 1e-6
        True
        >>> ra.angleAxis()[1]
        array([ 0.,  0.,  1.])
        """
        s = self._array[0]
        if s < 1.:
            f = 1./numpy.sqrt(1.-self._array[0]**2)
            result = 2.*numpy.arccos(s), f * self._array[1:]
        else:
            #result = 0., self._array[1:]
            result = 0., numpy.asarray([0., 0., 1.], self.atype)
        return result

    def squared(self):
        return numpy.dot(self._array, self._array)

    def normalize(self):
        """
        >>> q = Quaternion(2., 3., 4., 5.); q
        <Quaternion: 2.0 + 3.0i + 4.0j + 5.0k>
        >>> abs(q)
        7.3484692283495345
        >>> qn = q.normalized(); qn
        <Quaternion: 0.272165526976 + 0.408248290464i + 0.544331053952j + 0.68041381744k>
        >>> abs(1.0 - abs(qn)) < 1e-6
        True
        """
        self /= abs(self)
        return self
    def normalized(self):
        return self/abs(self)

    def inverse(self):
        """
        >>> Quaternion() # Identity
        <Quaternion: 1.0 + 0.0i + 0.0j + 0.0k>
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.))
        >>> aa * aa.inversed()
        <Quaternion: 1.0 + 0.0i + 0.0j + 0.0k>
        >>> aa.inversed() * aa
        <Quaternion: 1.0 + 0.0i + 0.0j + 0.0k>
        """
        self._array = self._quatInverse()
        return self
    def inversed(self):
        return self.fromV(self._quatInverse())

    def degreesTo(self, other):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); aa
        <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
        >>> a2 = Quaternion.fromDegreesAxis(60., (0., 0., 1.))
        >>> a2
        <Quaternion: 0.866025403784 + 0.0i + 0.0j + 0.5k>
        >>> (30 - aa.degreesTo(a2)) < 1e-6
        True
        >>> (30 - a2.degreesTo(aa)) < 1e-6
        True
        """
        return math.degrees(self.angleTo(other))

    def angleTo(self, other):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.))
        >>> a2 = Quaternion.fromDegreesAxis(60., (0., 0., 1.))
        >>> (0.5235987 - aa.angleTo(a2)) < 1e-6
        True
        >>> (0.5235987 - a2.angleTo(aa)) < 1e-6
        True
        """
        return 2. * numpy.arccos(numpy.dot(self._array, other._array))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getDegrees(self):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.))
        >>> (30 - aa.degrees) < 1e-6
        True
        """
        return math.degrees(self.getAngle())
    def setDegrees(self, value):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); aa
        <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
        >>> aa.degrees = 60; aa
        <Quaternion: 0.866025403784 + 0.0i + 0.0j + 0.5k>
        >>> (60 - aa.degrees) < 1e-6
        True
        """
        self.setAngle(math.radians(value))
    degree = degrees = property(getDegrees, setDegrees)

    def getAngle(self):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.))
        >>> (0.5235987755 - aa.angle) < 1e-6
        True
        """
        return 2. * numpy.arccos(self._array[0])
    def setAngle(self, value):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); aa
        <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
        >>> aa.angle = numpy.pi/4.; aa
        <Quaternion: 0.923879532511 + 0.0i + 0.0j + 0.382683432365k>
        >>> aa.angle
        0.78539816339744839
        """
        self._array = self._setFromAngleAxis(value, self.axis)
    angle = property(getAngle, setAngle)

    def getAxis(self):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); aa.axis
        array([ 0.,  0.,  1.])
        """
        return self.angleAxis()[-1]
    def setAxis(self, value):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); aa
        <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
        >>> aa.axis = 0., 1., 0.; aa
        <Quaternion: 0.965925826289 + 0.0i + 0.258819045103j + 0.0k>
        >>> aa.axis
        array([ 0.,  1.,  0.])
        """
        self._array = self._setFromAngleAxis(self.getAngle(), value)
    axis = property(getAxis, setAxis)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Conversions
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def asarray(self, *args):
        if args: return numpy.asarray(self._array, *args)
        else: return self._array

    def tolist(self):
        return self._array

    @classmethod
    def _setFromAngleAxis(klass, angle, axis):
        angle *= .5
        cosR, sinR = numpy.cos(angle), numpy.sin(angle)
        quat = numpy.asarray([cosR] + list(axis), klass.atype)
        unitNorm = numpy.sqrt(numpy.dot(quat[1:], quat[1:]))
        quat[1:] *= sinR / unitNorm
        return quat 

    @classmethod
    def fromDegreesAxis(klass, degrees, axis):
        """
        >>> aa = Quaternion.fromDegreesAxis(30., (0., 0., 1.)); aa
        <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
        """
        return klass(klass._setFromAngleAxis(math.radians(degrees), axis))

    @classmethod
    def fromAngleAxis(klass, angle, axis):
        """
        >>> ra = Quaternion.fromAngleAxis(numpy.pi/6., (0., 0., 1.)); ra
        <Quaternion: 0.965925826289 + 0.0i + 0.0j + 0.258819045103k>
        """
        return klass(klass._setFromAngleAxis(angle, axis))

    @classmethod
    def fromUnitRotationArc(klass, v0, v1):
        """
        >>> urarc = Quaternion.fromUnitRotationArc((0., 0., 1.), (1., 0., 0.)); urarc
        <Quaternion: 0.707106781187 + 0.0i + 0.707106781187j + 0.0k>
        >>> urarc.degrees
        90.0
        """
        cosR, axis = dotCross(v0, v1)
        s = numpy.sqrt(2.*(cosR+1))
        quat = numpy.asarray((s/2.,) + tuple(axis), klass.atype)
        quat[1:] /= s
        return klass(quat)
    fromUnitArc = fromUnitRotationArc

    @classmethod
    def fromRotationArc(klass, v0, v1):
        """
        >>> rarc = Quaternion.fromUnitRotationArc((0., 0., 3.), (9., 0., 0.)); rarc
        <Quaternion: 0.707106781187 + 0.0i + 19.091883092j + 0.0k>
        >>> rarc.degrees
        90.0
        >>> rarc = Quaternion.fromRotationArc((0., 1., 0.), (0., 0.99999, 0.)); rarc
        <Quaternion: 1.0 + 0.0i + 0.0j + 0.0k>
        >>> rarc.angleAxis()
        (0.0, array([ 0.,  0.,  1.]))
        """
        v0 = numpy.asarray(v0, klass.atype)
        v0 /= numpy.sqrt(numpy.dot(v0, v0))
        v1 = numpy.asarray(v1, klass.atype)
        v1 /= numpy.sqrt(numpy.dot(v1, v1))
        return klass.fromUnitRotationArc(v0, v1)
    fromArc = fromRotationArc

Q = Quat = Quaternion

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Testing 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    print "Testing..."
    import doctest
    import quat as _test
    doctest.testmod(_test)
    print "Test complete."

