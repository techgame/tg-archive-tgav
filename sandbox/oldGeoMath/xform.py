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

import math
import numpy
from numpy import linalg

from vector import Vector, UnitVector, linearMapping

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Transform3dh(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = numpy.Float

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def asArray4x4(self):
        """Returns the transformation in 4x4 numpy array form.
        """
        return numpy.identity(4)

    def asInverse4x4(self):
        """Returns the inverse transformation in 4x4 numpy array form.
        """
        return linalg.inverse(self.asArray4x4())
    def asLinAlgInverse4x4(self):
        """Returns the inverse transformation in 4x4 numpy array form.
        """
        return linalg.inverse(self.asArray4x4())

    def __mul__(self, other):
        if isinstance(other, Transform3dh):
            return self.composite(other)
        else:
            return self.xform(other)

    def __rmul__(self, other):
        if isinstance(other, Transform3dh):
            return self.rcomposite(other)
        else: 
            return other * self.asArray4x4()

    def asPoints(self, points, includeDimRestore=False):
        dims = len(points[0]) 
        if dims not in (2,3,4): 
            raise ValueError, "Points are not of right dimension -- must be 2, 3 or 4, but found %d" % dims
        if dims == 2:
            ones = numpy.ones((len(points), 1), self.atype)
            zeros = numpy.zeros((len(points), 1), self.atype)
            points = numpy.concatenate((points, zeros, ones), 1)
            correctdims = lambda pts: pts[:,:-2] # trim the z and homogenous coordinates -- assumes that they are all still 1.
        elif dims == 3:
            # add the homogeneous coordinate
            ones = numpy.ones((len(points), 1), self.atype)
            points = numpy.concatenate((points, ones), 1)
            correctdims = lambda pts: pts[:,:-1] # trim the homogenous coordinate -- assumes that they are all still 1.
        else:
            correctdims = lambda pts: pts

        if includeDimRestore:
            return points, correctdims
        else: return points

    def xform(self, points, bInverse=False):
        if bInverse:
            matrix = self.asInverse4x4()
        else:
            matrix = self.asArray4x4()
        
        points, correctdims = self.asPoints(points, True)
        result = numpy.transpose(numpy.dot(matrix, numpy.transpose(points)))
        return correctdims(result)

    def collapse(self):
        return Matrix(self.asArray3x3())

    def composite(self, other):
        #Composite([self, other])
        return Matrix(numpy.dot(self.asArray4x4(), other.asArray4x4()))
    def rcomposite(self, other):
        #Composite([other, self])
        return Matrix(numpy.dot(other.asArray4x4(), self.asArray4x4()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Composite(Transform3dh):
    """
    # Non-sensical test -- detects changes more than anything
    >>> from projections import *
    >>> c = Composite()
    >>> n = c.add(Identity())
    >>> n = c.add(Translate((1,2,3)))
    >>> n = c.add(Scale((2,3,4)))
    >>> n = c.add(Rotate(23, (7,11,13)))
    >>> c
    <Composite: [<Identity>, <Translation <1.0, 2.0, 3.0>>, <Scale <2.0, 3.0, 4.0>>, <Rotation 23.0 <0.380187, 0.597437, 0.706063>>] >
    >>> c.inverse()
    <Composite: [<Rotation 23.0 <-0.38018, -0.59743, -0.70606>>, <Scale <0.5, 0.333333, 0.25>>, <Translation <-1.0, -2.0, -3.0>>, <Identity>] >
    >>> abs((numpy.identity(4,'f') - (c * c.inverse()).asArray4x4()) < 1e-6).all()
    True
    >>> n = c.add(Skew(2,3,4,5,6,7))
    >>> n = c.add(Shear(10,20,30,40,50,60))
    >>> n = c.add(Orthographic())
    >>> n = c.add(Frustum())
    >>> n = c.add(Perspective())
    >>> abs(44.296370346028688 - c.asArray4x4()[0][0]) < 1e-6
    True
    >>> abs(-0.010173620849422499 - c.asInverse4x4()[0][0]) < 1e-6
    True
    """

    def __init__(self, collection=[]):
        self.collection = collection[:]

    def __repr__(self):
        return "<Composite: %s >" % self.collection

    def __contains__(self, *args, **kw):
        return self.collection.__contains__(self, *args, **kw)
    def __iter__(self):
        return iter(self.collection)
    def __len__(self):
        return len(self.collection)
    def __getitem__(self, *args, **kw):
        return self.collection.__getitem__(*args, **kw)
    def __setitem__(self, *args, **kw):
        return self.collection.__setitem__(*args, **kw)
    def __delitem__(self, *args, **kw):
        return self.collection.__delitem__(*args, **kw)

    def __imul__(self, other):
        if isinstance(other, Transform3dh):
            if isinstance(other, Composite):
                self.collection.extend(other.collection)
            else:
                self.collection.append(other)
            return self
        else:
            raise TypeError, "Can only inline multiply with other Transform3dh"

    def __mul__(self, other):
        if isinstance(other, Transform3dh):
            if isinstance(other, Composite):
                return Composite(self.collection + other.collection)
            else:
                return Composite(self.collection + [other])
        else: Transform3dh.__mul__(self, other)

    def __rmul__(self, other):
        if isinstance(other, Transform3dh):
            if isinstance(other, Composite):
                return Composite(other.collection + self.collection)
            else:
                return Composite([other] + self.collection)
        else: Transform3dh.__rmul__(self, other)

    def add(self, Transform):
        assert isinstance(Transform, Transform3dh)
        self.collection.append(Transform)
        return self.collection[-1]
    append=add

    def insert(self, idx, Transform):
        self.collection.insert(idx, Transform)
        return self.collection[idx]

    def clear(self):
        self.collection[:] = []
    
    def asArray4x4(self):
        """Returns the transformation in 4x4 numpy array form"""
        r = numpy.identity(4, self.atype)
        for xform in self.collection:
            r = numpy.dot(r, xform.asArray4x4())
        return r

    def inverse(self):
        result = self.__class__()
        result.collection = [x.inverse() for x in self.collection]
        result.collection.reverse()
        return result

    def composite(self, other):
        return Composite([self, other])
    def rcomposite(self, other):
        return Composite([other, self])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Matrix(Transform3dh):
    """
    >>> a = numpy.array([[2.0, 0.0, 0.0, 20.0], \
                         [0.0, 0.5, 2.0, 15.0], \
                         [0.0, 0.5, 1.5, -5.0], \
                         [0.0, 0.0, 0.0,  1.0]])
    >>> m = Matrix(a); m
    <Matrix: [[2.0, 0.0, 0.0, 20.0], [0.0, 0.5, 2.0, 15.0], [0.0, 0.5, 1.5, -5.0], [0.0, 0.0, 0.0, 1.0]] >
    >>> m.asArray4x4()
    array([[  2. ,   0. ,   0. ,  20. ],
           [  0. ,   0.5,   2. ,  15. ],
           [  0. ,   0.5,   1.5,  -5. ],
           [  0. ,   0. ,   0. ,   1. ]])
    >>> numpy.dot(m.asArray4x4(), m.asInverse4x4())
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    
    """

    def __init__(self, matrix=None):
        if matrix is None: 
            self.matrix = numpy.identity(4, self.atype)
        else: 
            self.matrix = numpy.asarray(matrix, self.atype)
            assert self.matrix.shape == (4,4)

    def __repr__(self):
        return "<Matrix: %s >" % self.matrix.tolist()

    def __imul__(self, other):
        if isinstance(other, Transform3dh):
            self.matrix = numpy.dot(self.matrix, other.asArray4x4())
            return self
        else:
            raise TypeError, "Can only inline multiply with other Transform3dh"

    def asArray4x4(self):
        return self.matrix

    def inverse(self):
        return self.__class__(self.asInverse4x4())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Identity(Transform3dh):
    """
    >>> i = Identity()
    >>> i
    <Identity>
    >>> i.asArray4x4()
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> i.asInverse4x4()
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

    >>> i.xform([(2., 3., 4., 1.)])
    array([[ 2.,  3.,  4.,  1.]])
    >>> i.xform([(2., 3., 4., 1.)], True)
    array([[ 2.,  3.,  4.,  1.]])

    >>> i.xform([(2., 3., 4.)])
    array([[ 2.,  3.,  4.]])
    >>> i.xform([(2., 3., 4.)], True)
    array([[ 2.,  3.,  4.]])

    >>> i.xform([(2., 3.)])
    array([[ 2.,  3.]])
    >>> i.xform([(2., 3.)], True)
    array([[ 2.,  3.]])
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "<Identity>"

    def asArray4x4(self):
        return numpy.identity(4, self.atype)

    def asInverse4x4(self):
        return numpy.identity(4, self.atype)

    def inverse(self):
        return self.__class__()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Translate(Transform3dh):
    """
    >>> t = Translate([-1,-2,-3]); t
    <Translation <-1.0, -2.0, -3.0>>
    >>> t.asArray4x4()
    array([[ 1.,  0.,  0., -1.],
           [ 0.,  1.,  0., -2.],
           [ 0.,  0.,  1., -3.],
           [ 0.,  0.,  0.,  1.]])
    >>> t.asInverse4x4()
    array([[ 1.,  0.,  0.,  1.],
           [ 0.,  1.,  0.,  2.],
           [ 0.,  0.,  1.,  3.],
           [ 0.,  0.,  0.,  1.]])
    """

    atype = Transform3dh.atype
    direction = Vector.property((0., 0., 0.), atype=atype)

    def __init__(self, direction=(0,0,0)):
        self.direction = direction

    def __repr__(self):
        return "<Translation %s>" % (self.direction,)

    def asArray4x4(self):
        result = numpy.identity(4, self.atype)
        result[:-1, 3] = self.direction
        return result

    def asInverse4x4(self):
        result = numpy.identity(4, self.atype)
        result[:-1, 3] = -self.direction
        return result

    def inverse(self):
        return self.__class__(-self.direction)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Scale(Transform3dh):
    """
    >>> s = Scale([2,3,4]);s
    <Scale <2.0, 3.0, 4.0>>
    >>> s.asArray4x4()
    array([[ 2.,  0.,  0.,  0.],
           [ 0.,  3.,  0.,  0.],
           [ 0.,  0.,  4.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> s.asInverse4x4()
    array([[ 0.5       ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.33333333,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.25      ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """

    atype = Transform3dh.atype
    scale = Vector.property((1.,1.,1.), atype=atype)

    def __init__(self, scale=1.0):
        if not isinstance(scale, (tuple,list)):
            # uniform scaling
            self.scale = (scale,) * 3
        else:
            # make sure it is length 3
            self.scale = scale

    def __repr__(self):
        return "<Scale %r>" % (self.scale,)

    def asArray4x4(self):
        result = numpy.identity(4, self.atype)
        for idx in range(4-1): result[idx,idx] = self.scale[idx]
        return result

    def asInverse4x4(self):
        result = numpy.identity(4, self.atype)
        for idx in range(4-1): result[idx,idx] = 1./self.scale[idx]
        return result

    def inverse(self):
        return self.__class__([1./x for x in self.scale])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Rotate(Transform3dh):
    """
    >>> Rotate(30).asArray4x4()
    array([[ 0.8660254, -0.5      ,  0.       ,  0.       ],
           [ 0.5      ,  0.8660254,  0.       ,  0.       ],
           [ 0.       ,  0.       ,  1.       ,  0.       ],
           [ 0.       ,  0.       ,  0.       ,  1.       ]])
    >>> Rotate(30, [1, 0, 0]).asArray4x4()
    array([[ 1.       ,  0.       ,  0.       ,  0.       ],
           [ 0.       ,  0.8660254, -0.5      ,  0.       ],
           [ 0.       ,  0.5      ,  0.8660254,  0.       ],
           [ 0.       ,  0.       ,  0.       ,  1.       ]])
    >>> Rotate(30, [0, 1, 0]).asArray4x4()
    array([[ 0.8660254,  0.       ,  0.5      ,  0.       ],
           [ 0.       ,  1.       ,  0.       ,  0.       ],
           [-0.5      ,  0.       ,  0.8660254,  0.       ],
           [ 0.       ,  0.       ,  0.       ,  1.       ]])
    >>> Rotate(30, [0, 0, 1]).asArray4x4()
    array([[ 0.8660254, -0.5      ,  0.       ,  0.       ],
           [ 0.5      ,  0.8660254,  0.       ,  0.       ],
           [ 0.       ,  0.       ,  1.       ,  0.       ],
           [ 0.       ,  0.       ,  0.       ,  1.       ]])
    >>> Rotate(37, [2, 3, 5]).asArray4x4()
    array([[ 0.81983177, -0.45634205,  0.34587252,  0.        ],
           [ 0.51993083,  0.8463271 , -0.11576859,  0.        ],
           [-0.23989121,  0.27474056,  0.93111215,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    >>> Rotate(37, [2, 3, 5]).asInverse4x4()
    array([[ 0.81983177,  0.51993083, -0.23989121,  0.        ],
           [-0.45634205,  0.8463271 ,  0.27474056,  0.        ],
           [ 0.34587252, -0.11576859,  0.93111215,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    >>> Rotate(37, [1., 2., 3.]).axis
    <0.267261, 0.534522, 0.801783>
    >>> abs(30.-Rotate(30, [1., 2., 3.]).degrees) < 1e-6
    True
    >>> abs(0.52359877559829882 - Rotate(30, [1., 2., 3.]).angle) < 1e-6
    True
    >>> Rotate(30, [1., 2., 3.])
    <Rotation 29.999999999999996 <0.267261, 0.534522, 0.801783>>
    >>> a = Rotate(30, [1., 2., 3.]); a
    <Rotation 29.999999999999996 <0.267261, 0.534522, 0.801783>>
    >>> ai = a.inverse(); ai
    <Rotation 29.999999999999996 <-0.26726, -0.53452, -0.80178>>
    >>> (((a * ai).asArray4x4() - numpy.identity(4, 'f')) < 1e-6).any()
    True
    >>> (((ai * a).asArray4x4() - numpy.identity(4, 'f')) < 1e-6).any()
    True
    """
    atype = Transform3dh.atype
    axis = UnitVector.property((0.,0.,1.), atype=atype)

    def __init__(self, degrees=0.0, axis=(0.0, 0.0, 1.0)):
        self.setDegrees(degrees)
        self.axis = axis

    def __repr__(self):
        return "<Rotation %r %r>" % (self.degrees, self.axis)

    def inverse(self):
        return self.__class__(self.degrees, -self.axis)

    def getDegrees(self):
        return math.degrees(self.getAngle())
    def setDegrees(self, degrees):
        self.setAngle(math.radians(degrees))
    degree = degrees = property(getDegrees, setDegrees)

    _angle = 0.
    def getAngle(self):
        return self._angle
    def setAngle(self, value):
        self._angle = value
    angle = property(getAngle, setAngle)

    def asArray4x4(self):
        angle = self.angle
        u = self.axis
        uut = numpy.outerproduct(u, u)
        M = numpy.identity(3, self.atype) - uut
        S = numpy.asarray([[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0.]], self.atype)
        R = uut + numpy.cos(angle) * M + numpy.sin(angle) * S
        result = numpy.identity(4, self.atype)
        result[:3,:3] = R
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LookAt(Transform3dh):
    """
    >>> l = LookAt((13.,-12.,11.), (1.,2.,3.), (1.,1.,1.)); l
    <LookAt <1.0, 2.0, 3.0> eye:<13.0, -12.0, 11.0> up:<0.577350, 0.577350, 0.577350>>
    >>> l.asArray4x4()
    array([[  0.64153303,   0.11664237,  -0.7581754 ,   1.39970842],
           [  0.4816635 ,   0.70798732,   0.51648255,  -3.4470858 ],
           [  0.59702231,  -0.69652603,   0.39801488, -20.49776612],
           [  0.        ,   0.        ,   0.        ,   1.        ]])
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = Transform3dh.atype
    eye = Vector.property((0.,0.,1.), atype=atype)
    center = Vector.property((0.,0.,0.), atype=atype)
    up = UnitVector.property((0.,1.,0.), atype=atype)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$

    def __init__(self, eye=None, center=None, up=None):
        Transform3dh.__init__(self)
        if eye is not None: self.eye = eye
        if center is not None: self.center = center
        if up is not None: self.up = up

    def __repr__(self):
        return "<LookAt %s eye:%s up:%s>" % (self.center, self.eye, self.up)

    def asArray4x4(self):
        e = self.eye
        c = self.center
        u = self.up
        l = c - e
        l.normalize()
        s = l.cross(u)
        s.normalize()
        u_ = s.cross(l)

        result = numpy.identity(4, self.atype)
        result[0,:-1] = s
        result[1,:-1] = u_
        result[2,:-1] = -l
        xlate = numpy.identity(4, self.atype)
        xlate[:-1,3] = -e
        return numpy.dot(result, xlate)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SphericalLookAt(LookAt):
    """
    >>> s = SphericalLookAt((10.,45.,45.), center=[0,0,1]); s
    <SphericalLookAt <0.0, 0.0, 1.0> row:10.0 theta:45.0 phi:45.0 up:<0.0, 1.0, 0.0>>
    >>> s.spherical
    <5.0, 7.071067, 5.0>
    >>> s.eye
    <5.0, 7.071067, 6.0>
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = Transform3dh.atype

    center = Vector.property((0.,0.,0.), atype=atype)
    spherical = Vector.property((0.,0.,1.), atype=atype)
    up = UnitVector.property((0.,1.,0.), atype=atype)

    _rhoThetaPhi = (1., 0., 90)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, rhoThetaPhiDegrees=None, center=None, up=None):
        LookAt.__init__(self, None, center=center, up=up)
        if rhoThetaPhiDegrees is not None: 
            self.rhoThetaPhiDegrees = rhoThetaPhiDegrees

    def __repr__(self):
        rtp = "row:%.1f theta:%.1f phi:%.1f" % self.getRhoThetaPhiDegrees()
        return "<SphericalLookAt %r %s up:%r>" % (self.center, rtp, self.up)

    def getRhoThetaPhi(self):
        return self._rhoThetaPhi
    def setRhoThetaPhi(self, (row,theta,fi)):
        self._rhoThetaPhi = row,theta,fi
        rowsinfi = row*numpy.sin(fi)
        x,y,z = (rowsinfi*numpy.cos(theta),row*numpy.cos(fi),rowsinfi*numpy.sin(theta))
        self.spherical = x,y,z
    rhoThetaPhi = property(getRhoThetaPhi, setRhoThetaPhi)

    def getRhoThetaPhiDegrees(self):
        row, theta, fi = self.rhoThetaPhi
        return row, math.degrees(theta), math.degrees(fi)
    def setRhoThetaPhiDegrees(self, (row,theta,fi)):
        self.setRhoThetaPhi((row, math.radians(theta), math.radians(fi)))
    rhoThetaPhiDegrees = property(getRhoThetaPhiDegrees, setRhoThetaPhiDegrees)

    def getEye(self):
        return self.center + self.spherical
    eye = property(getEye)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Shear(Matrix):
    """
    >>> s = Shear(1,2,3,4,5,6)
    >>> s
    <Shear: [[1.0, 1.0, 2.0, 0.0], [3.0, 1.0, 4.0, 0.0], [5.0, 6.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] >
    >>> s.asArray4x4()
    array([[ 1.,  1.,  2.,  0.],
           [ 3.,  1.,  4.,  0.],
           [ 5.,  6.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> s.asInverse4x4()
    array([[-1.15,  0.55,  0.1 ,  0.  ],
           [ 0.85, -0.45,  0.1 , -0.  ],
           [ 0.65, -0.05, -0.1 ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ]])
    >>> s.asLinAlgInverse4x4()
    array([[-1.15,  0.55,  0.1 ,  0.  ],
           [ 0.85, -0.45,  0.1 , -0.  ],
           [ 0.65, -0.05, -0.1 ,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  1.  ]])
    """

    def __init__(self, xy=0.0, xz=0.0, yx=0.0, yz = 0.0, zx=0.0, zy=0.0):
        Matrix.__init__(self)
        self.matrix[0, 1] = xy
        self.matrix[0, 2] = xz
        self.matrix[1, 0] = yx
        self.matrix[1, 2] = yz
        self.matrix[2, 0] = zx
        self.matrix[2, 1] = zy

    def __repr__(self):
        return "<Shear: %s >" % self.matrix.tolist()

    def inverse(self):
        return Matrix(self.asInverse4x4())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Skew(Shear):
    """
    >>> s = Skew(10,20,30,40,50,60)
    >>> s
    <Skew: [[1.0, 0.17632698070846498, 0.36397023426620234, 0.0], [0.57735026918962573, 1.0, 0.83909963117727993, 0.0], [1.19175359259421, 1.7320508075688767, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] >
    >>> s.asArray4x4()
    array([[ 1.        ,  0.17632698,  0.36397023,  0.        ],
           [ 0.57735027,  1.        ,  0.83909963,  0.        ],
           [ 1.19175359,  1.73205081,  1.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    >>> s.asInverse4x4()
    array([[ 1.01054753, -1.01216303,  0.4814964 ,  0.        ],
           [-0.94208715, -1.26214385,  1.40195612, -0.        ],
           [ 0.42741917,  3.39234621, -2.00208431,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    >>> s.asLinAlgInverse4x4()
    array([[ 1.01054753, -1.01216303,  0.4814964 ,  0.        ],
           [-0.94208715, -1.26214385,  1.40195612, -0.        ],
           [ 0.42741917,  3.39234621, -2.00208431,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """

    def __init__(self, xy=0.0, xz=0.0, yx=0.0, yz=0.0, zx=0.0, zy=0.0):
        kw = self._convertArgs(xy=xy, xz=xz, yx=yx, yz=yz, zx=zx, zy=zy)
        Shear.__init__(self, **kw)

    def __repr__(self):
        return "<Skew: %s >" % self.matrix.tolist()

    def _convertArgs(self, **kw):
        result = {}
        for key,value in kw.iteritems(): 
            result[key] = numpy.tan(math.radians(value))
        return result

    def inverse(self):
        return Matrix(self.asInverse4x4())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LinearMappingMatrix(Matrix):
    """
    >>> M = LinearMappingMatrix()
    >>> M.toX = [0., 100.]
    >>> M.toY = [10., 20.]
    >>> M.fromZ = [10., 20.]
    >>> M
    <LinearMappingMatrix: [([-1.0, 1.0], [0.0, 100.0]), ([-1.0, 1.0], [10.0, 20.0]), ([10.0, 20.0], [-1.0, 1.0])] >
    >>> M.asArray4x4()
    array([[ 50. ,   0. ,   0. ,  50. ],
           [  0. ,   5. ,   0. ,  15. ],
           [  0. ,   0. ,   0.2,  -3. ],
           [  0. ,   0. ,   0. ,   1. ]])
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Definitions 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fromX = [-1., 1.]
    fromY = [-1., 1.]
    fromZ = [-1., 1.]

    toX = [-1., 1.]
    toY = [-1., 1.]
    toZ = [-1., 1.]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    linearMapping = staticmethod(linearMapping)

    def __repr__(self):
        return "<LinearMappingMatrix: %s >" % ([(self.fromX, self.toX), (self.fromY, self.toY), (self.fromZ, self.toZ)],)

    def getFromRect(self):
        return numpy.asarray([self.fromX, self.fromY, self.fromZ], self.atype)
    def setFromRect(self, rect):
        self.fromX = rect[:2,0]
        self.fromY = rect[:2,1]
        self.fromZ = rect[:2,2]

    def getToRect(self):
        return numpy.asarray([self.toX, self.toY, self.toZ], self.atype)
    def setToRect(self, rect):
        self.toX = rect[:2,0]
        self.toY = rect[:2,1]
        self.toZ = rect[:2,2]

    def asArray4x4(self):
        Xs,Xt = self.linearMapping(self.fromX, self.toX, False)
        Ys,Yt = self.linearMapping(self.fromY, self.toY, False)
        Zs,Zt = self.linearMapping(self.fromZ, self.toZ, False)
        result = numpy.asarray([
            [Xs, 0, 0, Xt],
            [0, Ys, 0, Yt],
            [0, 0, Zs, Zt],
            [0, 0, 0, 1]], 
            self.atype)
        return result

    def asInverse4x4(self):
        Xs,Xt = self.linearMapping(self.toX, self.fromX, False)
        Ys,Yt = self.linearMapping(self.toY, self.fromY, False)
        Zs,Zt = self.linearMapping(self.toZ, self.fromZ, False)
        result = numpy.asarray([
            [Xs, 0, 0, Xt],
            [0, Ys, 0, Yt],
            [0, 0, Zs, Zt],
            [0, 0, 0, 1]], 
            self.atype)
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Testing 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    print "Testing..."
    import doctest, xform
    doctest.testmod(xform)
    print "Test complete."


