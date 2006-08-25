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

from bisect import insort, bisect_left, bisect_right

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Block(object):
    x = y = w = h = 0
    key = None

    def __cmp__(self, other):
        if (other is True): 
            return -1
        else: 
            return cmp(self.key, other.key)

    @classmethod
    def fromSize(klass, size, key=None):
        self = klass()
        self.size = size
        if key is not None:
            self.key = key
        return self

    @classmethod
    def fromPosSize(klass, pos, size, key=None):
        self = klass()
        self.pos = pos
        self.size = size
        if key is not None:
            self.key = key
        return self

    def getPos(self):
        return self.x, self.y
    def setPos(self, pos):
        self.x, self.y = pos
        assert self.x >= 0
        assert self.y >= 0
    pos = property(getPos, setPos)

    def getSize(self):
        return self.w, self.h
    def setSize(self, size):
        self.w, self.h = size
        assert self.w >= 0
        assert self.h >= 0
    size = property(getSize, setSize)

    def getArea(self, borders=0):
        return (self.w + borders*2)*(self.h + borders*2)
    area = property(getArea)

    unused = None
    def addUnusedRgn(self, pos, size, key=None):
        if not min(size): 
            return None

        if self.unused is None:
            self.unused = []

        anUnusedRgn = self.fromPosSize(pos, size, key)
        self.unused.append(anUnusedRgn)
        return anUnusedRgn

    def printUnused(self, exclude=(), title='Unused:'):
        print title
        total = 0
        for r in self.unused:
            if r.key in exclude: continue
            total += r.area
            print '  %5s <+ %5s' % (total, r.area), ':', r.key, 's:', r.size, 'p:', r.pos
        print '-- unused :', total, 'ratio:', total/float(self.area) 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BlockMosaicAlg(object):
    BlockFactory = Block
    borders = 1

    def __init__(self):
        self.blocks = []

    _maxSize = None
    def getMaxSize(self):
        return self._maxSize
    def setMaxSize(self, maxSize):
        self._maxSize = maxSize
    maxSize = property(getMaxSize, setMaxSize)

    @property
    def maxArea(self):
        w,h = self.getMaxSize()
        return w*h

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def addBlock(self, size, key=None):
        self._checkModify()

        aBlock = self.BlockFactory.fromSize(size, key)
        self.blocks.append(aBlock)
        return aBlock

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
    def _orderBlocksByHeight(self, blocks):
        blocksByHeight = {}
        for block in blocks:
            w, h = block.size
            widthMap = blocksByHeight.setdefault(h, [])
            insort(widthMap, (w, block))
        return blocksByHeight

    def _calcSizeStats(self):
        self._lockModify()

        blocks = self.blocks
        blocksByHeight = self._orderBlocksByHeight(blocks)
        heights = sorted(blocksByHeight.keys())
        self._heightGrps = [list(g) for g in self._avgDeltaGroups(heights)]

        self._blockArea = sum(b.getArea(self.borders) for b in blocks)
        self._blockAreaP2 = self._nextPowerOf2(self._blockArea)

        self._blockWMaxP2 = self._nextPowerOf2(max(b.w for b in blocks))
        self._blockHMaxP2 = self._nextPowerOf2(max(b.h for b in blocks))

        if self._blockAreaP2 > self.maxArea:
            raise Exception("Layout not possible. Required layout area is larger than maximum area available")

        width, height = self._blockWMaxP2, self._blockHMaxP2

        widthList = [sum(w for w, b in widthMap) for h, widthMap in blocksByHeight.iteritems()]
        widthP2List = map(self._nextPowerOf2, widthList)

        if 1:
            self._wWidest = max(widthP2List)
            self._wWidestP2 = self._nextPowerOf2(self._wWidest)
            self._hWidestP2 = self._blockAreaP2 / self._wWidestP2

            if (self._wWidestP2 > width) and (self._hWidestP2 > height):
                return (self._wWidestP2, self._hWidestP2)

        if 1:
            self._wWide = sum(widthP2List)/len(widthP2List) 
            self._wWideP2 = self._nextPowerOf2(self._wWide)
            self._hWideP2 = self._blockAreaP2 / self._wWideP2

            if (self._wWideP2 > width) and (self._hWideP2 > height):
                return (self._wWideP2, self._hWideP2)

        if 0:
            self._wNarrow = sum(widthList)/len(widthList)
            self._wNarrowP2 = self._nextPowerOf2(self._wNarrow)
            self._hNarrowP2 = self._blockAreaP2 / self._wNarrowP2

            if (self._wNarrowP2 > width) and (self._hNarrowP2 > height):
                return (self._wNarrowP2, self._hNarrowP2)

        if 0:
            self._wMaxWidth = self.maxSize[0]
            self._wMaxWidthP2 = self._nextPowerOf2(self._wMaxWidth)
            self._hMaxWidthP2 = self._blockAreaP2 / self._wMaxWidthP2

            if (self._wMaxWidthP2 > width) and (self._hMaxWidthP2 > height):
                return (self._wMaxWidthP2, self._hMaxWidthP2)

        width <<= 3
        return width, (self._blockAreaP2 / width)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def layout(self):
        self._lockModify()
        size = self._calcSizeStats()
        return self.layoutSize(size)

    def layoutSize(self, size):
        rgn = self.BlockFactory.fromSize(size)
        return rgn, self.iterLayoutRegion(rgn)

    def iterLayoutRegion(self, rgn):
        heightMap = self._orderBlocksByHeight(self.blocks)

        bOutOfRoom = False
        hMax = 0
        x = y = self.borders
        for heightGrp in self._heightGrps:
            for elemH in heightGrp:
                widthMap = heightMap[elemH]
                if not widthMap:
                    continue

                if elemH > hMax:
                    # mark the height delta as unused
                    heightRgn = rgn.addUnusedRgn((x, y), (x, (elemH-hMax)), key='HeightGrow')
                    hMax = elemH
                elif elemH < hMax:
                    raise Exception("Heights are not in increasing order")

                bOutOfRoom = (y+hMax > rgn.h)
                if bOutOfRoom: 
                    raise Exception("Layout Error: no more room")

                rowWidth = rgn.w - x
                while widthMap:
                    # Use bisect to find a block that is smaller than the remaining row width
                    i = bisect_right(widthMap, (rowWidth, True)) - 1

                    if i >= 0:
                        # we found one, pop it off
                        elemW, block = widthMap.pop(i)

                        # adjust it's layout position
                        block.pos = (x, y)
                        # and yield it
                        yield block

                        # advance horizontally
                        x += elemW + self.borders*2
                        rowWidth = rgn.w - x

                        bNewRow = rowWidth < (elemW >> 1)
                    else:
                        # there are no blocks small enough to fit on this row
                        bNewRow = True


                    if bNewRow:
                        # mark the rest of this row as unused
                        if rowWidth > 0:
                            endRng = rgn.addUnusedRgn((x, y), (rowWidth, hMax), key='Endcap')
                        else: 
                            assert rowWidth >= -2*self.borders, rowWidth

                        # advance to the beginning of the next row
                        y += hMax + self.borders*2
                        hMax = elemH
                        x = self.borders
                        rowWidth = rgn.w - x

                        bOutOfRoom = (y+hMax > rgn.h)
                        if bOutOfRoom: 
                            if widthMap:
                                raise Exception("Layout Error: no more room")
                            else: break

        if rowWidth > 0:
            lastRowRgn = rgn.addUnusedRgn((x, y), (rowWidth, hMax), key='LastRow')
        else: 
            assert rowWidth >= -2*self.borders, rowWidth

        if hMax > 0:
            bottomRgn = rgn.addUnusedRgn((x, y), (rowWidth, hMax), key='Bottom')
        else: 
            assert hMax >= -2*self.borders, hMax

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Utility Functions
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _powersOfTwo = [1<<s for s in xrange(31)]

    @staticmethod
    def _idxNextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return bisect_left(powersOfTwo, v)
    @staticmethod
    def _nextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return powersOfTwo[bisect_left(powersOfTwo, v)]
    @staticmethod
    def _avgDeltaGroups(v):
        dv = [e1-e0 for e0, e1 in zip(v[:-1], v[1:])] + [0]
        dAvg = sum(dv, 0.)/len(dv)

        p = v[0]
        r = [p]
        for n in v[1:]:
            if n-p > dAvg:
                yield r
                r = []
            r.append(n)
            p = n

        if r: yield r

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Modification control
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _modifyLocked = False
    def _checkModify(self):
        if self._modifyLocked:
            raise Exception("Algorithm modification locked")
    def _lockModify(self, lock=True):
        self._modifyLocked = lock
    def _unlockModify(self):
        self._lockModify(False)

