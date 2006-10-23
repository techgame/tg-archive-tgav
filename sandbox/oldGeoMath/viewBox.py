#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy
from vector import linearMapping
from xform import Transform3dh
from xform2d import Transform2dh

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ViewBox(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = numpy.Float

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, pos0=(-1., -1.), pos1=(1., 1.)):
        self.setPts(pos0, pos1)

    def __repr__(self):
        (x0, y0), (x1, y1) = self.getBox()
        return '<%s [%r, %r]>' % (self.__class__.__name__, (x0, y0), (x1, y1))

    def copy(self):
        return self.fromViewBox(self)

    def blend(self, factor, viewbox):
        return self.fromBox((1-factor)*self.getBox()+factor*viewbox.getBox())

    def updateProjection(self, proj):
        _box = self.getBox()
        proj.Left, proj.Bottom = _box[0]
        proj.Right, proj.Top = _box[1]

    def fromViewBox(klass, viewbox):
        result = ViewBox.__new__(ViewBox) # avoid constructor call
        result.setBox(viewbox.getBox())
        return result
    fromViewBox = classmethod(fromViewBox)
    def fromBox(klass, box):
        result = ViewBox.__new__(ViewBox) # avoid constructor call
        result.setBox(box)
        return result
    fromBox = classmethod(fromBox)
    def fromPts(klass, pos0, pos1):
        return klass(pos0, pos1)
    fromPts = classmethod(fromPts)
    def fromRectangle(klass, pos0, dim):
        pos0, dim = klass._ascoords(pos0, dim)
        return klass(pos0, pos0+dim)
    fromRectangle = classmethod(fromRectangle)
    def fromCenter(klass, center, dim):
        center, dim = klass._ascoords(center, dim)
        dim = 0.5*dim
        return klass(center-dim, center+dim)
    fromCenter = classmethod(fromCenter)

    def __iadd__(self, value):
        if isinstance(value, ViewBox):
            return self.setBox(self.getBox()+value.getBox())
        else:
            return self.setBox(self.getBox()+value)
    def __add__(self, value):
        if isinstance(value, ViewBox):
            return self.fromBox(self.getBox()+value.getBox())
        else:
            return self.fromBox(self.getBox()+value)
    def __radd__(self, value):
        if isinstance(value, ViewBox):
            return self.fromBox(value.getBox()+self.getBox())
        else:
            return self.fromBox(value+self.getBox())

    def __isub__(self, value):
        if isinstance(value, ViewBox):
            return self.setBox(self.getBox()-value.getBox())
        else:
            return self.setBox(self.getBox()-value)
    def __sub__(self, value):
        if isinstance(value, ViewBox):
            return self.fromBox(self.getBox()-value.getBox())
        else:
            return self.fromBox(self.getBox()-value)
    def __rsub__(self, value):
        if isinstance(value, ViewBox):
            return self.fromBox(value.getBox()-self.getBox())
        else:
            return self.fromBox(value-self.getBox())

    def __imul__(self, value):
        return self.setBox(self.getBox()*value)
    def __mul__(self, value):
        return self.fromBox(self.getBox()*value)
    def __rmul__(self, value):
        return self.fromBox(value*self.getBox())

    def __idiv__(self, value):
        return self.setBox(self.getBox()/value)
    def __div__(self, value):
        return self.fromBox(self.getBox()/value)

    def getP0(self):
        return self.getBox()[0]
    def getP1(self):
        return self.getBox()[1]
    def getBox(self):
        return self._box
    def getRect(self):
        _box = self.getBox()
        return _box[0], _box[1]-_box[0]
    def getRectangle(self):
        _box = self.getBox()
        (x, y), (w, h) = _box[0], _box[1]-_box[0]
        return x, y, w, h
    def getPts(self):
        return self.getBox()
    def getCenter(self):
        _box = self.getBox()
        return 0.5*(_box[0]+_box[1])
    def getSize(self):
        _box = self.getBox()
        return _box[1]-_box[0]
    def getAspectRatio(self):
        size = self.getSize()
        return size[1]/size[0]
    def getWidth(self):
        return self.getSize()[0]
    def getHeight(self):
        return self.getSize()[1]
    def getXSpan(self):
        box = self.getBox()
        return (box[0][0], box[1][0])
    def getYSpan(self):
        box = self.getBox()
        return (box[0][1], box[1][1])
    def getXYSpan(self):
        return numpy.transpose(self.getBox())

    def setP0(self, p0):
        return self.setBox((p0, self.getP1()))
    def setP1(self, p1):
        return self.setBox((self.getP0(), p1))
    def setBox(self, box):
        self._box = numpy.asarray(box, self.atype)
        return self._box
    def setRect(self, pos, dim):
        pos, dim = self._ascoords(pos, dim)
        return self.setPts(pos, pos+dim)
    def setRectangle(self, rect):
        pos, dim = self._ascoords(rect[:2], rect[2:4])
        return self.setPts(pos, pos+dim)
    def setPts(self, pos0, pos1):
        pos0, pos1 = self._ascoords(pos0, pos1)
        pos0, pos1 = numpy.minimum(pos0, pos1), numpy.maximum(pos0, pos1)
        return self.setBox((pos0, pos1))
    def setCenterAndSize(self, center, dim):
        center, dim = self._ascoords(center, dim)
        dim = 0.5*dim
        return self.setBox((center-dim, center+dim))
    def setSize(self, dim):
        dim, = self._ascoords(dim)
        _box = self.getBox()
        return self.setBox((_box[0], _box[0] + dim))
    def setWidth(self, width):
        return self.setSize((width, self.getHeight()))
    def setHeight(self, height):
        return self.setSize((self.getWidth(), height))
        
    def getXYWH(self):
        raise NotImplementedError()

    def setAspectRatio(self, aspectYX=1.0, largest=True):
        return self.viewBoxSize(self.getSize(), aspectYX, largest)

    def zoomCenter(self, factor, center=(0., 0.), relative=True):
        center, = self._ascoords(center)
        halfsize = 0.5*self.getSize()/factor
        if relative: center += self.getCenter()
        return self.setPts(center-halfsize, center+halfsize)

    def move(self, pos, relative=True):
        _box = self.getBox()
        pos, = self._ascoords(pos)
        if not relative:
            pos = pos-_box[0] # make it relative
        return self.setPts(_box[0]+pos, _box[1]+pos)

    def moveHorizontal(self, pos, relative=True):
        if relative:
            return self.move((pos, 0.), True)
        else:
            return self.move((pos, self.getBox()[0][1]), False)

    def moveVertical(self, pos, relative=True):
        if relative:
            return self.move((0., pos), True)
        else:
            return self.move((self.getBox()[0][0], pos), False)

    def viewBoxSize(self, dim, aspectYX=None, largest=True):
        size = self._getViewBoxSize(dim, aspectYX, largest)
        return self.setCenterAndSize(self.getCenter(), size)

    def viewBox(self, pos, dim, aspectYX=None, largest=True):
        size = self._getViewBoxSize(dim, aspectYX, largest)
        return self.setCenterAndSize(pos, size)

    def viewBoxPts(self, pos0, pos1, aspectYX=None, largest=True):
        pos0, pos1 = self._ascoords(pos0, pos1)
        pos0, pos1 = numpy.minimum(pos0, pos1), numpy.maximum(pos0, pos1)
        size = self._getViewBoxSize(abs(pos1-pos0), aspectYX, largest)
        return self.setCenterAndSize(0.5*(pos0+pos1), size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def mapPointFrom0to1(self, posIn0to1):
        blend, = self._ascoords(posIn0to1)
        _box = self.getBox()
        result = (1-blend)*_box[0]+blend*_box[1]
        return result

    def mapPointTo0to1(self, pos):
        pos, = self._ascoords(pos)
        _box = self.getBox()
        return (pos-_box[0])/(_box[1]-_box[0])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def mapPointTo(self, point, *args, **kw):
        return self.mapPointsTo([point], *args, **kw)[0]

    def mapPointsTo(self, points, xspan=(-1., 1.), yspan=None, flipx=False, flipy=False):
        xspan, yspan = xspan or yspan, yspan or xspan
        if flipx: xspan = xspan[1], xspan[0]
        if flipy: yspan = yspan[1], yspan[0]
        span = numpy.transpose(numpy.asarray([xspan, yspan], self.atype))
        linearfn = linearMapping(self.getBox(), span)
        points = numpy.asarray(points, self.atype)
        return numpy.asarray(map(linearfn, points), self.atype)

    def mapPointFrom(self, point, *args, **kw):
        return self.mapPointsFrom([point], *args, **kw)[0]

    def mapPointsFrom(self, points, xspan=(-1., 1.), yspan=None, flipx=False, flipy=False):
        xspan, yspan = xspan or yspan, yspan or xspan
        if flipx: xspan = xspan[1], xspan[0]
        if flipy: yspan = yspan[1], yspan[0]
        span = numpy.transpose(numpy.asarray([xspan, yspan], self.atype))
        linearfn = linearMapping(span, self.getBox())
        points = numpy.asarray(points, self.atype)
        return numpy.asarray(map(linearfn, points), self.atype)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Protected Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _ascoords(klass, *args):
        return numpy.asarray(args, klass.atype)
    _ascoords = classmethod(_ascoords)

    def _getViewBoxSize(self, dim, aspectYX=None, largest=True):
        if aspectYX is None: aspectYX = self.getAspectRatio()

        width, height = map(float, dim)
        if largest: # scale by larger dimension
            if height/width > aspectYX: # height is greater -- scale height
                width = height/aspectYX
            else: # width is greater -- scale width
                height = width*aspectYX
        else:
            if height/width < aspectYX: # width is greater -- scale height
                width = height/aspectYX
            else: # height is greater -- scale width
                height = width*aspectYX
        return width, height

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BlendedViewBox(ViewBox):
    def __init__(self, viewbox0, viewbox1, blend=0.):
        self.setupBlend(viewbox0, viewbox1, blend)

    def setupBlend(self, viewbox0, viewbox1, blend=0.):
        self.blend = blend
        self.viewbox0 = viewbox0
        self.viewbox1 = viewbox1

    def getBlend(self):
        return self.blend

    def setBlend(self, blend):
        self.blend = blend

    def getBox(self):
        b = self.blend
        if b == 0.:
            return self.viewbox0.getBox()
        elif b == 1.:
            return self.viewbox1.getBox()
        else:
            return (1-b)*self.viewbox0.getBox()+b*self.viewbox1.getBox()

    def setBox(self, box):
        raise NotImplementedError

    def reduce(self):
        b = self.blend
        if b == 0.: return self.viewbox0
        elif b == 1.: return self.viewbox1
        else: return self

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ ViewBox
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ViewBoxBase(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _fromrect = ViewBox()
    _torect = ViewBox()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, fromrect=None, torect=None):
        if isinstance(fromrect, (list, tuple)):
            self._fromrect = ViewBox(fromrect)
        elif fromrect is not None:
            self._fromrect = fromrect
        if isinstance(torect, (list, tuple)):
            self._torect = ViewBox(torect)
        elif torect is not None:
            self._torect = torect

    def __repr__(self):
        return "<%s from:%r to:%r>" % (self.__class__.__name__, self.mapFrom, self.mapTo)

    def getMapTo(self):
        return self._torect
    def setMapTo(self, torect):
        self._torect = torect
    def delMapTo(self):
        del self._torect
    mapTo = property(getMapTo, setMapTo, delMapTo)

    def getMapFrom(self):
        return self._fromrect
    def setMapFrom(self, fromrect):
        self._fromrect = fromrect
    def delMapFrom(self):
        del self._fromrect
    mapFrom = property(getMapFrom, setMapFrom, delMapFrom)

    def _getMapping(self, frombox, tobox):
        (x0f, y0f), (x1f, y1f) = frombox
        (x0t, y0t), (x1t, y1t) = tobox
        sx, tx = linearMapping((x0f, x1f), (x0t, x1t), False)
        sy, ty = linearMapping((y0f, y1f), (y0t, y1t), False)
        return (sx, tx), (sy,ty)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ViewBox3dh(ViewBoxBase, Transform3dh):
    def asArray4x4(self):
        (sx, tx), (sy, ty) = self._getMapping(self.getMapFrom().getBox(), self.getMapTo().getBox())
        result = numpy.asarray([
            [sx,  0,  0, tx],
            [ 0, sy,  0, ty],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]], self.atype)
        return result

    def asInverse4x4_(self):
        (sx, tx), (sy, ty) = self._getMapping(self.getMapTo().getBox(), self.getMapFrom().getBox())
        result = numpy.asarray([
            [sx,  0,  0, tx],
            [ 0, sy,  0, ty],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]], self.atype)
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ViewBox2dh(ViewBoxBase, Transform2dh):
    def asArray4x4(self):
        (sx, tx), (sy, ty) = self._getMapping(self.getMapFrom().getBox(), self.getMapTo().getBox())
        result = numpy.asarray([
            [sx,  0, tx],
            [ 0, sy, ty],
            [ 0,  0,  1]], self.atype)
        return result

    def asInverse4x4_(self):
        (sx, tx), (sy, ty) = self._getMapping(self.getMapTo().getBox(), self.getMapFrom().getBox())
        result = numpy.asarray([
            [sx,  0, tx],
            [ 0, sy, ty],
            [ 0,  0,  1]], self.atype)
        return result

