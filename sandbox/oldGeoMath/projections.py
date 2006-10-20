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
from xform import Transform3dh

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Projection(Transform3dh):
    """
    >>> proj = Projection(-2, 3, 5,8, -9, -2)
    >>> proj.width
    5.0
    >>> proj.height
    3.0
    >>> proj.deprojh
    7.0
    >>> proj.center
    (0.5, 6.5, -5.5)
    >>> proj.dimensions
    (5.0, 3.0, 7.0)
    >>> proj.hCenter
    0.5
    >>> proj.vCenter
    6.5
    >>> proj.dCenter
    -5.5
    """
    def __init__(self, left=-1.0, right=1.0, bottom=-1.0, top=1.0, near=-1.0, far=1.0):
        self.left = float(left)
        self.right = float(right)
        self.bottom = float(bottom)
        self.top = float(top)
        self.near = float(near)
        self.far = float(far)

    def setCorners(self, Lower, Higher):
        self.left, self.bottom, self.near = Lower
        self.right, self.top, self.far = Higher

    def getWidth(self):
        return self.right - self.left
    def setWidth(self, value):
        delta = (self.width - value) * 0.5
        self.right -= delta
        self.left += delta
    width = property(getWidth, setWidth)
    def getHCenter(self, mult=0.5):
        return (self.left + self.right) * mult
    def setHCenter(self, value):
        hw = self.width * 0.5
        self.right = value + hw
        self.left = value - hw
    hCenter = property(getHCenter, setHCenter)

    def getHeight(self):
        return self.top - self.bottom
    def setHeight(self, value):
        delta = (self.height - value) * 0.5
        self.top -= delta
        self.bottom += delta
    height = property(getHeight, setHeight)
    def getVCenter(self, mult=0.5):
        return (self.top + self.bottom) * mult
    def setVCenter(self, value):
        hw = self.height * 0.5
        self.top = value + hw
        self.bottom = value - hw
    vCenter = property(getVCenter, setVCenter)

    def getDeprojh(self):
        return self.far - self.near
    def setDeprojh(self, value):
        delta = (self.deprojh - value) * 0.5
        self.far -= delta
        self.near += delta
    deprojh = property(getDeprojh, setDeprojh)
    def getDCenter(self, mult=0.5):
        return (self.far + self.near) * mult
    def setDCenter(self, value):
        hw = self.deprojh * 0.5
        self.far = value + hw
        self.near = value - hw
    dCenter = property(getDCenter, setDCenter)

    def getCenter(self):
        return (self.hCenter, self.vCenter, self.dCenter)
    def setCenter(self, value):
        self.hCenter, self.vCenter, self.dCenter = value
    center = property(getCenter, setCenter)
    def getDimensions(self):
        return (self.width, self.height, self.deprojh)
    def setDimensions(self, value):
        self.width, self.height, self.deprojh = value
    dimensions = property(getDimensions, setDimensions)

    def getAspectRatio(self):
        return self.height / self.width
    def setAspectRatio(self, value, byWidth=None):
        self.setDimensionsAspectRatio(self.width, self.height, value)
    aspect = property(getAspectRatio, setAspectRatio)

    def setDimensionsAspectRatio(self, width, height, aspectYX=1., largest=True):
        width, height, aspectYX = map(float, (width, height, aspectYX))
        if largest: # scale by larger dimension
            if height/width > aspectYX: # height is greater -- scale width
                width = height/aspectYX
            else: # width is greater -- scale height
                height = width*aspectYX
        else: # scale by smaller dimension
            if height/width < aspectYX: # width is greater -- scale height
                width = height/aspectYX
            else: # height is greater -- scale width
                height = width*aspectYX
        self.width, self.height = width, height
        return width, height

    def getRect(self):
        return [[self.left, self.right], [self.bottom, self.top], [self.near, self.far]]
    def setRect(self, value):
        ((self.left, self.right), (self.bottom, self.top), (self.near, self.far)) = value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Orthographic(Projection):
    """
    >>> proj = Orthographic(-2, 3, 5,8, -9, -2); proj
    <Orthographic: [[-2.0, 3.0], [5.0, 8.0], [-9.0, -2.0]]>
    >>> proj.asArray4x4()
    array([[ 0.4       ,  0.        ,  0.        ,  0.2       ],
           [ 0.        ,  0.66666667,  0.        ,  4.33333333],
           [ 0.        ,  0.        , -0.28571429, -1.57142857],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """

    def asArray4x4(self):
        """As defined by the OpenGL Red Bock"""
        width = self.width
        height = self.height
        deprojh = self.deprojh
        xoff = self.getHCenter(1.) / width
        yoff = self.getVCenter(1.) / height
        zoff = self.getDCenter(1.) / deprojh

        result = numpy.asarray([
            [2./width, 0., 0., xoff],
            [0., 2./height, 0., yoff],
            [0., 0., -2./deprojh, zoff],
            [0., 0., 0., 1.]],
            self.atype)
        return result

    def __repr__(self):
        return "<Orthographic: %s>" % (self.getRect(),)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Viewport(Orthographic):
    """
    >>> proj = Viewport(0, 10, 0, 10, -1, 1); proj
    <Viewport: [[0.0, 10.0], [0.0, 10.0], [-1.0, 1.0]]>
    >>> proj.asArray4x4()
    array([[ 5.,  0.,  0.,  5.],
           [ 0.,  5.,  0.,  5.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> proj.asInverse4x4()
    array([[ 0.2,  0. ,  0. , -1. ],
           [ 0. ,  0.2,  0. , -1. ],
           [ 0. ,  0. ,  1. ,  0. ],
           [ 0. ,  0. ,  0. ,  1. ]])
    >>> # Create "screen" coordinates"
    >>> v = numpy.transpose(numpy.asarray([(4.,4.,0.,1.),(6.,6.,0.,1.)])); v
    array([[ 4.,  6.],
           [ 4.,  6.],
           [ 0.,  0.],
           [ 1.,  1.]])
    >>> # Convert to v to [-1,1] nominal coordinates
    >>> vn = numpy.dot(proj.asInverse4x4(), v); vn
    array([[-0.2,  0.2],
           [-0.2,  0.2],
           [ 0. ,  0. ],
           [ 1. ,  1. ]])
    >>> # Convert back to "screen" coordinates
    >>> numpy.dot(proj.asArray4x4(), vn)
    array([[ 4.,  6.],
           [ 4.,  6.],
           [ 0.,  0.],
           [ 1.,  1.]])
    """

    def __init__(self, left=0., right=1., bottom=0., top=1., near=-1., far=1.):
        self.left = float(left)
        self.right = float(right)
        self.bottom = float(bottom)
        self.top = float(top)
        self.near = float(near)
        self.far = float(far)

    def asArray4x4(self):
        """As defined by the OpenGL Red Bock"""
        halfwidth = 0.5 * self.width
        halfheight = 0.5 * self.height
        halfdeprojh = 0.5 * self.deprojh
        xoff = self.getHCenter()
        yoff = self.getVCenter()
        zoff = self.getDCenter()

        result = numpy.asarray([
            [halfwidth, 0, 0, xoff],
            [0, halfheight, 0, yoff],
            [0, 0, halfdeprojh, zoff],
            [0, 0, 0, 1]])
        return result

    def __repr__(self):
        return "<Viewport: %s>" % (self.getRect(),)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Frustum(Projection):
    """
    >>> proj = Frustum(-2, 3, 5,8, -9, -2); proj
    <Frustm: [[-2.0, 3.0], [5.0, 8.0], [-9.0, -2.0]]>
    >>> proj.asArray4x4()
    array([[-3.6       ,  0.        ,  0.2       ,  0.        ],
           [ 0.        , -6.        ,  4.33333333,  0.        ],
           [ 0.        ,  0.        ,  1.57142857, -2.57142857],
           [ 0.        ,  0.        , -1.        ,  0.        ]])
    """

    def __init__(self, left=-1.0, right=1.0, bottom=-1.0, top=1.0, near=1., far=10.):
        Projection.__init__(self, left, right, bottom, top, near, far)

    def asArray4x4(self):
        """As defined by the OpenGL Red Bock"""
        width = self.width
        height = self.height
        deprojh = self.deprojh
        xoff = self.getHCenter(1.) / width
        yoff = self.getVCenter(1.) / height
        zoff = self.getDCenter(1.) / deprojh
        near2 = self.near * 2.
        ugly = -self.near * self.far / deprojh

        result = numpy.asarray([
            [near2/width, 0., xoff, 0.],
            [0., near2/height, yoff, 0.],
            [0., 0., -zoff, ugly],
            [0., 0., -1., 0.]],
            self.atype)
        return result

    def __repr__(self):
        return "<Frustm: %s>" % (self.getRect(),)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Perspective(Frustum):
    """
    >>> proj = Perspective(30., 1.234, 0.5, 23.2); proj
    <Persepctive degrees:30.0 aspect=1, n=0.5, f=23.2>
    >>> proj.asArray4x4()
    array([[ 4.6053507 ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  3.73205081,  0.        ,  0.        ],
           [ 0.        ,  0.        , -1.04405286, -0.51101322],
           [ 0.        ,  0.        , -1.        ,  0.        ]])
    >>> (1.234 - proj.aspect) < 1e-6
    True
    >>> abs(30 - proj.degrees) < 1e-6
    True
    >>> abs(0.52359877 - proj.angle) < 1e-6
    True
    """

    aspect = 1.
    near = 1.
    far = 1.

    def __init__(self, degrees=45., aspect=1, near=1., far=10.):
        """Derived from OpenGL Red Book"""
        Frustum.__init__(self,-1.,1.,-1.,1.,near,far)
        self.setAspectDegrees(aspect, degrees)

    def __repr__(self):
        return "<Persepctive degrees:%.1f aspect=%.0f, n=%.1f, f=%.1f>" % (self.degrees, self.aspect, self.near, self.far)

    def getAngle(self):
        return 2.*math.atan(self.height/(2.*self.near))
    def setAngle(self, angle):
        self.setAspectAngle(self.aspect, angle)
    angle = property(getAngle, setAngle)

    def getDegrees(self):
        return math.degrees(self.getAngle())
    def setDegrees(self, value):
        self.setAngle(math.radians(value))
    degrees = property(getDegrees, setDegrees)

    def getAspectAngle(self):
        return self.aspect, self.getAngle()
    def setAspectAngle(self, aspect, angle):
        self.aspect = aspect
        self.height = math.tan(0.5*angle) * 2. * self.near
        self.width = self.height / aspect
    aspectAngle = property(getAspectAngle, setAspectAngle)
    
    def getAspectDegrees(self):
        return self.aspect, self.getDegrees()
    def setAspectDegrees(self, aspect, degrees):
        self.setAspectAngle(aspect, math.radians(degrees))
    aspectDegrees = property(getAspectDegrees, setAspectDegrees)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Testing 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    print "Testing..."
    import doctest, projections
    doctest.testmod(projections)
    print "Test complete."

