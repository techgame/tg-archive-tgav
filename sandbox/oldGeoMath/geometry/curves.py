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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_2pi = 2 * numpy.pi

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CurveBase(object):
    _steps = 16
    numpyType = numpy.Float

    def __init__(self, steps=None):
        if steps is not None:
            self.setSteps(steps)

    def getSteps(self):
        return self._steps
    def setSteps(self, steps):
        if steps != self._steps:
            self._steps = steps
            self._invalidate()

    def asCurve(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.asCurve())

    def _invalidate(self):
        pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Bezier Based Curves
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BezierBase(CurveBase):
    def __init__(self, *points, **kw):
        if points:
            self.setControlPoints(*points)
        CurveBase.__init__(self, **kw)

    def _createUVector(self):
        return self._buildUVector(numpy.arrayrange(0., 1., 1./(self.getSteps()-1), self.numpyType), True)

    def _buildUVector(self, u, addend=False):
        result = [u, numpy.ones(len(u), self.numpyType)]
        count = self.getDimension()+1
        for i in xrange(2, count):
            result.insert(0, u*result[0])
        if addend:
            result = numpy.concatenate((result, numpy.ones((count, 1), self.numpyType)), 1)
        return numpy.transpose(result)

    _uvector = ()
    def getUVector(self):
        result = self._uvector
        if len(result) != self.getSteps():
            result = self._uvector = self._createUVector()
        return result

    def setControlPoints(self, *points):
        assert len(points) == self.getDimension()+1
        self.points = numpy.asarray(points)
    def getControlPoints(self):
        return self.points

    def curveFromUVector(self, uvector):
        ubezier = numpy.dot(uvector, self.getBezierMatrix())
        return numpy.dot(ubezier, self.getControlPoints())

    def curveAtPoints(self, *points):
        uvector = self._buildUVector(numpy.asarray(points, self.numpyType))
        return self.curveFromUVector(uvector)

    def curveRange(self, start=0., stop=1., stepsize=None):
        stepsize = stepsize or 1./(self.getSteps()-1)
        uvector = self._buildUVector(numpy.arrayrange(start, stop, stepsize, self.numpyType))
        return self.curveFromUVector(uvector)

    def asCurve(self):
        return self.curveFromUVector(self.getUVector())

    def getDimension(self):
        return len(self.getBezierMatrix())-1
    def getBezierMatrix(self):
        raise NotImplementedError

    def _invalidate(self):
        self._uvector = ()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LinearBezier(BezierBase):
    _matrix = numpy.asarray([
        [-1.,  1.],
        [ 1.,  0.]], 
        BezierBase.numpyType)

    def getBezierMatrix(self):
        return self._matrix

    def getDimension(self):
        return 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QuadraticBezier(BezierBase):
    _matrix = numpy.asarray([
        [ 1., -2.,  1.],
        [-2.,  2.,  0.],
        [ 1.,  0.,  0.]], 
        BezierBase.numpyType)

    def getBezierMatrix(self):
        return self._matrix

    def getDimension(self):
        return 2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CubicBezier(BezierBase):
    _matrix = numpy.asarray([
        [-1.,  3., -3.,  1.],
        [ 3., -6.,  3.,  0.],
        [-3.,  3.,  0.,  0.],
        [ 1.,  0.,  0.,  0.]], 
        BezierBase.numpyType)

    def getBezierMatrix(self):
        return self._matrix

    def getDimension(self):
        return 3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Ellipse Based Curves
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Ellipse(CurveBase):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    center = (0., 0.)
    rx = 1.
    ry = 1.
    sweep = (0., _2pi)
    xrotation = 0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, center=(0., 0.), rx=1., ry=1., startangle=0, endangle=360, xrotation=0, inDegrees=True, **kw):
        CurveBase.__init__(self, **kw)
        self.setCenter(center)
        self.setRadii(rx, ry)
        if inDegrees:
            self.setSweepAngles(startangle, endangle)
            self.setXRotationAngle(xrotation)
        else: 
            self.setSweep(startangle, endangle)
            self.setXRotation(xrotation)

    @classmethod
    def fromArc(klass, fromxy, toxy, rx=1., ry=1., xrotation=0, largeArcFlag=False, sweepFlag=False, inDegrees=True, **kw):
        """
        Derived from SVG Specification:
            http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
        """
        if inDegrees:
            xrotation = math.radians(xrotation)
        fromxy = numpy.asarray(fromxy, klass.numpyType)
        toxy = numpy.asarray(toxy, klass.numpyType)
        if xrotation:
            x, y = numpy.dot(klass._rotationMatrix(-xrotation), 0.5*(fromxy-toxy))
        else:
            x, y = 0.5*(fromxy-toxy)
        x2,y2 = x**2, y**2

        rx2, ry2 = rx**2, ry**2
        radiscale = x2/rx2 + y2/ry2
        if radiscale > 1:
            # ellipse is not big enough... scale it
            radiscale = numpy.sqrt(radiscale)
            rx, ry = radiscale*rx, radiscale*ry
            rx2, ry2 = rx**2, ry**2

        tmp = rx2*y2 + ry2*x2
        radicand = (rx2*ry2-tmp)/tmp
        if radicand > 0:
            sign = (largeArcFlag != sweepFlag) and 1 or -1
            scale = sign * numpy.sqrt(radicand)
            cxP, cyP = scale*y*rx/ry, scale*x*-ry/rx
        else:
            cxP, cyP = 0., 0.

        if xrotation:
            rot = klass._rotTranMatrix(xrotation, (0.5*(fromxy+toxy)))
            center = numpy.dot(rot, numpy.asarray([cxP, cyP, 1.], klass.numpyType))
        else:
            center = numpy.asarray([cxP, cyP]) + (0.5*(fromxy+toxy))
        center = center[:2]
        
        startpt = ((x-cxP)/rx, (y-cyP)/ry)
        endpt = ((-x-cxP)/rx, (-y-cyP)/ry)
        start = klass._findAngle((1., 0.), startpt)
        sweep = klass._findAngle(startpt, endpt) % _2pi
        if sweepFlag:
            if sweep < 0:
                sweep += _2pi
        else:
            if sweep > 0:
                sweep -= _2pi

        end = start + sweep
        return klass(center, rx, ry, start, end, xrotation, inDegrees=False, **kw)

    def setCenter(self, center):
        self.center = center
    def getCenter(self):
        return self.center

    def setRadius(self, radius):
        self.setRadii(radius, radius)
    def setRadii(self, rx, ry=None):
        if ry is None: rx, ry = rx
        self.rx = rx
        self.ry = ry
    def getRadii(self):
        return self.rx, self.ry

    def setSweepAngles(self, startangle, endangle):
        self.setSweep(math.radians(startangle), math.radians(endangle))
    def setSweep(self, start, end):
        self.sweep = [start, end]
    def getSweep(self):
        return self.sweep

    def setXRotationAngle(self, xrotationAngle):
        self.setXRotation(math.radians(xrotationAngle))
    def setXRotation(self, xrotation):
        self.xrotation = xrotation
    def getXRotation(self):
        return self.xrotation

    def asCurve(self):
        sweep = self._getSweepVector()
        rx, ry = self.getRadii()
        xrotation = self.getXRotation()
        if xrotation:
            ellipse = numpy.asarray([rx*numpy.cos(sweep), ry*numpy.sin(sweep), numpy.ones(len(sweep), self.numpyType)], self.numpyType)
            rot = self._rotTranMatrix(xrotation, self.getCenter())
            ellipse = numpy.dot(rot, ellipse)
            return numpy.transpose(ellipse[:-1,:])
        else:
            cx, cy = self.getCenter()
            ellipse = numpy.asarray([rx*numpy.cos(sweep)+cx, ry*numpy.sin(sweep)+cy], self.numpyType)
            return numpy.transpose(ellipse)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _sweepvector = ()
    def _getSweepVector(self):
        sweep = self._sweepvector
        if len(sweep) != self.getSteps():
            sweep = self._sweepvector = self._createSweepVector()
        return sweep

    def _createSweepVector(self):
        steps = self.getSteps()-1 # we append the last answer so it is always end-correct... even if the last step is a little off
        startrad, endrad = self.getSweep()
        sweep = numpy.arrayrange(startrad, endrad, (endrad-startrad)/steps, self.numpyType)
        sweep = numpy.concatenate((sweep, [endrad]), 0)
        return sweep

    @classmethod
    def _rotationMatrix(klass, radians):
        """Count clockwise rotation"""
        cosr = numpy.cos(radians)
        sinr = numpy.sin(radians)
        return numpy.asarray([[cosr, -sinr],[sinr, cosr]], klass.numpyType)

    @classmethod
    def _rotTranMatrix(klass, radians, (tx, ty)=(0.,0.)):
        """Count clockwise rotation"""
        cosr = numpy.cos(radians)
        sinr = numpy.sin(radians)
        return numpy.asarray([[cosr, -sinr, tx],[sinr, cosr, ty], [0., 0., 1.]], klass.numpyType)

    @classmethod
    def _findAngle(klass, u, v):
        cosangle = numpy.dot(u,v)/numpy.sqrt(numpy.dot(u,u)*numpy.dot(v,v))
        sign = numpy.sign(u[0]*v[1]-u[1]*v[0]) or 1. # 0 is neither poistive or negative... but just make it positive for argument's sake...
        return sign*numpy.arccos(cosangle)

    def _invalidate(self):
        self._sweepvector = ()

