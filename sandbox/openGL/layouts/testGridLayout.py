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

import time

from TG.openGL.layouts.cells import *
from TG.openGL.layouts.gridLayout import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

box = Rect.fromPosSize((0,0), (1000, 1000))

def runGridLayout(nRows=2, nCols=4, excess=4):

    if 1: gl = FlexGridLayout(nRows, nCols)
    else: gl = GridLayout(nRows, nCols)

    if 1: cells = [Cell((i%2, (i//4)%2), (100, 100)) for i in xrange(nRows*nCols+excess)]
    else: cells = [Cell() for i in xrange(nRows*nCols+excess)]

    if 1:
        gl.inside.set(10)
        gl.outside.set((50, 50, 0))

    for p in xrange(2):
        lb = gl.layout(cells, box, not p%2)
        print
        print 'box:', box
        print '  layout:', lb
        for i, c in enumerate(cells):
            print '    cell %s:' % i, c.box
        print

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def runGridTiming(n=100):
    box = Rect.fromPosSize((0,0), (1000, 1000))
    box.size *= (5, 2)

    nRows, nCols = 10, 8
    excess = 0
    if 0: gl = FlexGridLayout(nRows, nCols)
    else: gl = GridLayout(nRows, nCols)

    if 1: cells = [Cell((i%2, (i//4)%2), (100, 100)) for i in xrange(nRows*nCols+excess)]
    else: cells = [Cell() for i in xrange(nRows*nCols+excess)]

    cn = max(1, len(cells)*n)

    if 1:
        s = time.time()
        for p in xrange(n):
            gl.layout(cells, box, False)
        dt = time.time() - s
        print '%r time: %5s cn/s: %5s pass/s: %5s' % ((n, nRows, nCols, cn), dt, cn/dt, n/dt)

    if 1:
        s = time.time()
        for p in xrange(n):
            gl.layout(cells, box, True)
        dt = time.time() - s
        print '%r time: %5s cn/s: %5s pass/s: %5s' % ((n, nRows, nCols, cn), dt, cn/dt, n/dt)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    if 1:
        runGridLayout()

    # timing analysis
    if 1:
        runGridTiming()

