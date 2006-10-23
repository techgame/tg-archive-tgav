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

from vector import UnitVector, linearMapping

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def from3x3to4x4(matrix):
    r = numpy.identity(4, matrix.dtype)
    r[:2,:2] = matrix[:2,:2]
    r[ 3,:2] = matrix[ 2,:2]
    r[:2, 3] = matrix[:2, 2]
    r[ 3, 3] = matrix[ 3, 2]
    return r

class Transform2dh(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = numpy.Float

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def asArray3x3(self):
        """Returns the transformation in 3x3 numpy array form."""
        return numpy.identity(3, self.atype)

    def asInverse3x3(self):
        """Returns the inverse transformation in 3x3 numpy array form."""
        return linalg.inverse(self.asArray3x3())

    from3x3to4x4 = staticmethod(from3x3to4x4)

    def asArray4x4(self):
        """Returns the transformation in 4x4 numpy array form."""
        return from3x3to4x4(self.asArray3x3())

    def asInverse4x4(self):
        """Returns the inverse transformation in 4x4 numpy array form."""
        return from3x3to4x4(self.asArray3x3())

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
        if dims == 3:
            points = numpy.asarray(points, self.atype)
            dimRestore = lambda pts: pts
        elif dims == 2:
            # add the homogeneous coordinate
            #points = numpy.asarray(points, self.atype)
            ones = numpy.ones((len(points), 1), self.atype)
            points = numpy.concatenate((points, ones), 1)
            dimRestore = lambda pts: pts[:,:-1]
        else: 
            raise ValueError, "Points are not of right dimension -- must be 2 or 3, but found %d" % dims

        if includeDimRestore:
            return points, dimRestore
        else: return points

    def xform(self, points, bInverse=False):
        if bInverse:
            matrix = self.asInverse3x3()
        else:
            matrix = self.asArray3x3()
        
        points, dimRestore = self.asPoints(points, True)
        newPoints = numpy.dot(points, matrix.transpose())
        return dimRestore(newPoints)

    def collapse(self):
        return Matrix(self.asArray3x3())

    def composite(self, other):
        #Composite([self, other])
        return Matrix(numpy.dot(self.asArray4x4(), other.asArray4x4()))
    def rcomposite(self, other):
        #Composite([other, self])
        return Matrix(numpy.dot(other.asArray4x4(), self.asArray4x4()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Composite(Transform2dh):
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
        if isinstance(other, Transform2dh):
            if isinstance(other, Composite):
                self.collection.extend(other.collection)
            else:
                self.collection.append(other)
            return self
        else:
            raise TypeError, "Can only inline multiply with other Transform2dh"

    def __mul__(self, other):
        if isinstance(other, Transform2dh):
            if isinstance(other, Composite):
                return Composite(self.collection + other.collection)
            else:
                return Composite(self.collection + [other])
        else: 
            return Transform2dh.__mul__(self, other)

    def __rmul__(self, other):
        if isinstance(other, Transform2dh):
            if isinstance(other, Composite):
                return Composite(other.collection + self.collection)
            else:
                return Composite([other] + self.collection)
        else: 
            return Transform2dh.__rmul__(self, other)

    def add(self, Transform):
        self.collection.append(Transform)
        return self.collection[-1]
    append=add

    def insert(self, idx, Transform):
        self.collection.insert(idx, Transform)
        return self.collection[idx]

    def clear(self):
        self.collection[:] = []
    
    def asArray3x3(self):
        """Returns the transformation in 3x3 numpy array form"""
        coll = self.collection
        if coll:
            r = coll[0].asArray3x3()
            for xf in coll[1:]:
                r = numpy.dot(r, xf.asArray3x3())
            return r
        else:
            return numpy.identity(3, self.atype)

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

class Matrix(Transform2dh):
    def __init__(self, matrix=None):
        if matrix: 
            self.matrix = numpy.asarray(matrix, self.atype)
            self.matrix.reshape((3,3))
        else: 
            self.matrix = numpy.identity(3, self.atype)

    def __repr__(self):
        return "<Matrix: %s >" % self.matrix.tolist()

    def __imul__(self, other):
        if isinstance(other, Transform2dh):
            self.matrix = numpy.dot(self.matrix, other.asArray3x3())
            return self
        else:
            raise TypeError, "Can only inline multiply with other Transform2dh"

    def asArray3x3(self):
        return self.matrix

    def inverse(self):
        return self.__class__(self.asInverse3x3())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Identity(Transform2dh):
    """
    >>> i = Identity()
    >>> i
    <Identity>
    >>> i.asArray3x3()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> i.asInverse3x3()
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    >>> i.xform([(2., 3., 1.)])
    array([[ 2.,  3.,  1.]])
    >>> i.xform([(2., 3., 1.)], True)
    array([[ 2.,  3.,  1.]])

    >>> i.xform([(2., 3.)])
    array([[ 2.,  3.]])
    >>> i.xform([(2., 3.)], True)
    array([[ 2.,  3.]])
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "<Identity>"

    def asArray3x3(self):
        return numpy.identity(3, self.atype)

    def asInverse3x3(self):
        return numpy.identity(3, self.atype)

    def inverse(self):
        return self.__class__()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Translate(Transform2dh):
    def __init__(self, direction=(0,0)):
        self.direction = direction

    def __repr__(self):
        return "<Translation: %r>" % (self.direction,)

    def asArray3x3(self):
        result = numpy.identity(3, self.atype)
        result[:-1, 2] = self.direction
        return result

    def asInverse3x3(self):
        result = numpy.identity(3, self.atype)
        result[:-1, 2] = -self.direction
        return result

    def inverse(self):
        return self.__class__(-self.direction)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Scale(Transform2dh):
    def __init__(self, Scale=1.0):
        if not isinstance(Scale, (tuple,list)):
            # uniform scaling
            self.Scale = (Scale, Scale, 1.)
        elif len(Scale) == 1:
            self.Scale = (Scale[0], Scale[0], 1.)
        elif len(Scale) > 1:
            self.Scale = (Scale[0], Scale[1], 1.)
        else:
            raise ValueError, "Expecting scalar or tuple of length 1 or 2"

    def __repr__(self):
        return "<Scale: %r>" % (self.Scale,)

    def asArray3x3(self):
        result = numpy.identity(3, self.atype)
        for idx in range(3-1): result[idx,idx] = self.Scale[idx]
        return result

    def asInverse3x3(self):
        result = numpy.identity(3, self.atype)
        for idx in range(3-1): result[idx,idx] = 1./self.Scale[idx]
        return result

    def inverse(self):
        return self.__class__([1./x for x in self.Scale])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Rotate(Transform2dh):
    atype = Transform2dh.atype
    axis = UnitVector.property(atype=atype)

    def __init__(self, degrees=0.0, axis=(0.0, 0.0, 1.0)):
        self.degrees = float(degrees)
        self.axis = axis

    def __repr__(self):
        return "<Rotation: Angle=%s, axis=%s>" % (self.Angle, self.axis.tolist(),)

    _degrees = None
    def getDegrees(self):
        return math.degrees(self.angle)
    def setDegrees(self, degrees):
        self.setAngle(math.radians(degrees))
    degree = degrees = property(getDegrees, setDegrees)

    _angle = None
    def getAngle(self):
        if self._angle is None:
            pass #self.setAngle(None)
        return self._angle
    def setAngle(self, angle):
        self._angle = angle
    angle = property(getAngle, setAngle)

    def asArray3x3(self):
        u = self.axis[:-1]
        angle = self.angle
        atype = self.atype
        uut = numpy.outerproduct(u, u)
        m = numpy.identity(3, atype) - uut
        s = numpy.asarray([[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0.]], atype)
        r = uut + numpy.cos(angle) * m + numpy.sin(angle) * s
        result = numpy.identity(3, atype)
        result[:2,:2] = r[:2,:2]
        return result

    def inverse(self):
        return self.__class__(self.Angle, -self.axis)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Shear(Matrix):
    def __init__(self, xy=0.0, yx=0.0):
        Matrix.__init__(self)
        self.matrix[0, 1] = xy
        self.matrix[1, 0] = yx

    def __repr__(self):
        return "<Shear: %s >" % self.matrix.tolist()

    def inverse(self):
        result = Matrix(self.asInverse3x3())
        result.__class__ = self.__class__ # a little black magic never hurt too much ;)
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Skew(Shear):
    def __init__(self, xy=0.0, yx=0.0, inRadians=False):
        kw = self._convertArgs(inRadians, xy=xy, yx=yx)
        Shear.__init__(self, **kw)

    def __repr__(self):
        return "<Skew: %s >" % self.matrix.tolist()

    def _convertArgs(self, inRadians=False, **kw):
        result = {}
        if inRadians:
            for key,value in kw.iteritems(): 
                result[key] = numpy.tan(value)
        else:
            for key,value in kw.iteritems(): 
                result[key] = numpy.tan(math.radians(value))
        return result

    def inverse(self):
        result = Matrix(self.asInverse3x3())
        result.__class__ = self.__class__ # a little black magic never hurt too much ;)
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LinearMappingMatrix(Matrix):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Definitions 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fromX = [-1., 1.]
    fromY = [-1., 1.]

    toX = [-1., 1.]
    toY = [-1., 1.]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    linearMapping = staticmethod(linearMapping)

    def __repr__(self):
        return "<LinearMappingMatrix: %s >" % ([(self.fromX, self.toX), (self.fromY, self.toY)],)

    def getFromRect(self):
        return numpy.asarray([self.fromX, self.fromY], self.atype)
    def setFromRect(self, rect):
        self.fromX = rect[:2,0]
        self.fromY = rect[:2,1]

    def getToRect(self):
        return numpy.asarray([self.toX, self.toY], self.atype)
    def setToRect(self, rect):
        self.toX = rect[:2,0]
        self.toY = rect[:2,1]

    def asArray3x3(self):
        Xs,Xt = self.linearMapping(self.fromX, self.toX, False)
        Ys,Yt = self.linearMapping(self.fromY, self.toY, False)
        result = numpy.asarray([
            [Xs,  0, Xt],
            [ 0, Ys, Yt],
            [ 0,  0,  1]], 
            self.atype)
        return result

    def asInverse3x3(self):
        Xs,Xt = self.linearMapping(self.toX, self.fromX, False)
        Ys,Yt = self.linearMapping(self.toY, self.fromY, False)
        result = numpy.asarray([
            [Xs,  0, Xt],
            [ 0, Ys, Yt],
            [ 0,  0,  1]], 
            self.atype)
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Testing 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    print "Testing..."
    import doctest, xform2d
    doctest.testmod(xform2d)
    print "Test complete."


