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
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

maxSentinal = object()

class LayoutException(Exception): 
    pass
class LayoutRoomException(LayoutException): 
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Block object
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Block(object):
    x = y = w = h = 0
    key = None

    maxSentinal = maxSentinal
    def __cmp__(self, other):
        if (other is self.maxSentinal): 
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getPos(self):
        return self.x, self.y
    def setPos(self, pos):
        self.x, self.y = pos
    pos = property(getPos, setPos)

    def offset(self, pos, fromRgn=None):
        if fromRgn is not None:
            self.x = fromRgn.x + pos[0]
            self.y = fromRgn.y + pos[1]
        else:
            self.pos = pos

    def getSize(self):
        return self.w, self.h
    def setSize(self, size):
        self.w, self.h = size
    size = property(getSize, setSize)

    def getArea(self, borders=0):
        return (self.w + borders*2)*(self.h + borders*2)
    area = property(getArea)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Block Mosic Algorithm
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BlockMosaicAlg(object):
    BlockFactory = Block
    LayoutRegionFactory = None
    borders = 1

    def __init__(self, maxSize=None):
        self._layoutRgn = self.LayoutRegionFactory()
        if maxSize:
            self.maxSize = maxSize

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def addBlock(self, size, key=None):
        block = self.BlockFactory.fromSize(size, key)
        self._layoutRgn.addBlock(block)
        return block

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getMaxWidth(self):
        return self._layoutRgn.getMaxWidth()
    def setMaxWidth(self, maxWidth):
        return self._layoutRgn.setMaxWidth(maxWidth)
    maxWidth = property(getMaxWidth, setMaxWidth)

    def getMaxHeight(self):
        return self._layoutRgn.getMaxHeight()
    def setMaxHeight(self, maxHeight):
        return self._layoutRgn.setMaxHeight(maxHeight)
    maxHeight = property(getMaxHeight, setMaxHeight)

    def getMaxSize(self):
        return self._layoutRgn.getMaxSize()
    def setMaxSize(self, maxSize):
        return self._layoutRgn.setMaxSize(maxSize)
    maxSize = property(getMaxSize, setMaxSize)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def layout(self, borders=NotImplemented):
        return self.layoutSize(self.getMaxSize(), borders)

    def layoutSize(self, size, borders=NotImplemented):
        return self._layoutRegion(self._layoutRgn, size, borders)

    def _layoutRegion(self, rgn, size, borders=NotImplemented):
        if borders is NotImplemented:
            borders = self.borders

        rgn.setRgnFromSize(size)
        return rgn, rgn.iterLayout(borders)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Layout Regions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LayoutRegion(object):
    maxSentinal = maxSentinal

    RegionFactory = Block
    UnusedRegionFactory = None
    NarrowRegionFactory = RegionFactory
    BlockRegionFactory = RegionFactory

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def addBlock(self, block):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def iterLayout(self, borders=1):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _maxWidth = None
    def getMaxWidth(self):
        return self._maxWidth
    def setMaxWidth(self, maxWidth):
        self._maxWidth = maxWidth
    maxWidth = property(getMaxWidth, setMaxWidth)

    _maxHeight = None
    def getMaxHeight(self):
        return self._maxHeight or self.getMaxWidth()
    def setMaxHeight(self, maxHeight):
        self._maxHeight = maxHeight
    maxHeight = property(getMaxHeight, setMaxHeight)

    def getMaxSize(self):
        return (self.maxWidth, self.maxHeight)
    def setMaxSize(self, maxSize):
        if isinstance(maxSize, tuple):
            (self.maxWidth, self.maxHeight) = maxSize
        else:
            self.maxWidth, self.maxHeight = maxSize
    maxSize = property(getMaxSize, setMaxSize)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _rgn = None
    def getRgn(self):
        if self._rgn is None:
            self.setRgnFromSize()
        return self._rgn
    def setRgn(self, rgn):
        self._rgn = rgn
    rgn = property(getRgn, setRgn)

    def setRgnFromSize(self, size=None):
        if size is None: 
            size = self.getMaxSize()
        self.rgn = self.RegionFactory.fromSize(size)
    def setRgnFromPosSize(self, pos, size):
        self.rgn = self.RegionFactory.fromPosSize(pos, size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    minKeepDim = 10
    unused = None

    def addUnusedGrowRgn(self, pos, size, key=None):
        if self.UnusedRegionFactory is not None:
            rgn = self.UnusedRegionFactory.fromPosSize(pos, size, key)
            return self.addUnusedRgn(rgn, None)

    def addUnusedNarrowRgn(self, pos, size, key=None):
        rgn = self.NarrowRegionFactory.fromPosSize(pos, size, key)
        return self.addUnusedRgn(rgn, False)

    def addUnusedBlockRgn(self, pos, size, key=None):
        rgn = self.BlockRegionFactory.fromPosSize(pos, size, key)
        return self.addUnusedRgn(rgn, True)

    def addUnusedRgn(self, rgn, rgnIsBlock):
        if min(rgn.size) < self.minKeepDim: 
            return None

        if self.unused is None:
            self.unused = []

        self.unused.append(rgn)
        return rgn

    def printUnused(self, exclude=()):
        print "Font Texture size:", self.rgn.size
        print '  Unused Regions:'
        total = 0
        for r in self.unused:
            if r.key in exclude: continue
            total += r.area
            print '    %5s <+ %5s' % (total, r.area), ':', r.key, 's:', r.size, 'p:', r.pos
        print '  -- unused :', total, 'ratio:', total/float(self.rgn.area) 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

class VerticalLayoutRegion(LayoutRegion):
    def __init__(self):
        self._blocksByHeight = {}

    def addBlock(self, block):
        widthMap = self._blocksByHeight.setdefault(block.h, [])
        insort(widthMap, (block.w, block))
        return block

    def _avgDeltaHeights(self):
        return self._avgDeltaGroups(sorted(self._blocksByHeight.keys()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VerticalBlockLayoutRegion(VerticalLayoutRegion):
    bBreakGroups = True

    def iterLayout(self, borders=1):
        rgn = self.getRgn()

        heightMap = self._blocksByHeight
        heightGrps = self._avgDeltaHeights()

        maxSentinal = self.maxSentinal
        bOutOfRoom = False
        hMax = 0
        x = y = borders
        for heightGrp in heightGrps:
            for elemH in heightGrp:
                widthMap = heightMap[elemH]
                if not widthMap:
                    continue

                if elemH > hMax:
                    # mark the height delta as unused
                    heightRgn = self.addUnusedGrowRgn((x, y), (x, (elemH-hMax)))
                    hMax = elemH
                elif elemH < hMax:
                    raise LayoutException("Heights are not in increasing order")

                bOutOfRoom = (y+hMax > rgn.h)
                if bOutOfRoom: 
                    raise LayoutRoomException("Layout Error: no more room")

                rowWidth = rgn.w - x
                while widthMap:
                    # Use bisect to find a block that is smaller than the remaining row width
                    i = bisect_right(widthMap, (rowWidth, maxSentinal)) - 1

                    if i >= 0:
                        # we found one, pop it off
                        elemW, block = widthMap.pop(i)

                        # adjust it's layout position
                        block.offset((x, y), fromRgn=rgn)

                        # and yield it
                        yield block

                        # advance horizontally
                        x += elemW + borders*2
                        rowWidth = rgn.w - x

                        bNewRow = rowWidth < (elemW >> 1)
                    else:
                        # there are no blocks small enough to fit on this row
                        bNewRow = True


                    if bNewRow:
                        # mark the rest of this row as unused
                        endRng = self.addUnusedNarrowRgn((x, y), (rowWidth, hMax), key='Cap')

                        # advance to the beginning of the next row
                        y += hMax + borders*2
                        hMax = elemH
                        x = borders
                        rowWidth = rgn.w - x

                        bOutOfRoom = (y+hMax > rgn.h)
                        if bOutOfRoom: 
                            if widthMap:
                                raise LayoutRoomException("Layout Error: no more room")
                            else: break

            if self.bBreakGroups and (0 < rowWidth < x):
                # mark the rest of this row as unused
                endRng = self.addUnusedNarrowRgn((x, y), (rowWidth, hMax), key='Group')

                # advance to the beginning of the next row
                y += hMax + borders*2
                hMax = 0
                x = borders
                rowWidth = rgn.w - x

        if x > borders:
            lastRowRgn = self.addUnusedNarrowRgn((x, y), (rowWidth, hMax), key='Last')

        x = 0; y += hMax
        bottomRgn = self.addUnusedBlockRgn((x, y), (rgn.w - x, rgn.h - y), key='Bottom')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BlockMosaicAlg.LayoutRegionFactory = VerticalBlockLayoutRegion

