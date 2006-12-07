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

from numpy import ndarray, array, vander, dot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Bezier(ndarray):
    order = {}

    def at(self, u, pts=None):
        uvec = vander(array(u, copy=False, ndmin=1), len(self))
        r = dot(uvec, self)
        return r

    def dot(self, u, pts):
        return dot(self.at(u), pts)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for Mb in [
        array([[-1,  1], [ 1,  0]], 'b'),
        array([[ 1, -2,  1], [-2,  2,  0], [ 1,  0,  0]], 'b'),
        array([[-1,  3, -3,  1], [ 3, -6,  3,  0], [-3,  3,  0,  0], [ 1,  0,  0,  0]], 'b'),
        ]:

    Bezier.order[len(Mb)-1] = Bezier(Mb.shape, Mb.dtype, Mb)

Bezier.linear = Bezier.order[1]
Bezier.quadratic = Bezier.order[2]
Bezier.cubic = Bezier.order[3]

