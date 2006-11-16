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

from numpy import asarray, float32

from PIL import Image

from .raw import gl, glext
from .texture import Texture
from .data import interleavedArrays

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageGeometryArray(interleavedArrays.InterleavedArrays):
    drawMode = gl.GL_QUADS
    dataFormat = gl.GL_T2F_V3F

    @classmethod
    def fromCount(klass, count):
        return klass.fromFormat(count * 4, klass.dataFormat)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTextureBase(Texture):
    GeometryFactory = ImageGeometryArray

    texParams = Texture.texParams.copy()
    texParams.update(
            wrap=gl.GL_CLAMP_TO_EDGE,

            genMipmaps=True,
            magFilter=gl.GL_LINEAR,
            minFilter=gl.GL_LINEAR_MIPMAP_LINEAR,
            )

    imgModeFormatMap = {
        'RGBA': (gl.GL_RGBA, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
        'RGB': (gl.GL_RGB, gl.GL_RGB, gl.GL_UNSIGNED_BYTE),
        'LA': (gl.GL_LUMINANCE_ALPHA, gl.GL_LUMINANCE_ALPHA, gl.GL_UNSIGNED_BYTE),
        'L': (gl.GL_LUMINANCE, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE),
        }

    def create(self, image=None, **kwattrs):
        Texture.create(self, **kwattrs)

        if image is not None:
            self.loadImage(image)

    def load(self, image, format=True):
        if isinstance(image, basestring):
            self.loadFilename(image, format)
        else:
            self.loadImage(image, format)
    def loadFilename(self, filename, format=True):
        image = Image.open(filename)
        return self.loadImage(image, format)

    def loadImage(self, image, format=True):
        impliedFormat, dataFormat, dataType = self.imgModeFormatMap[image.mode]
        if format is True:
            self.format = impliedFormat
        elif format:
            self.format = format

        self.imageSize = image.size + (0.,)
        size = self.validSizeForTarget(image.size)
        data = self.data2d(size=size, format=dataFormat, dataType=dataType)
        data.setImageOn(self)

        data.texString(image.tostring(), dict(alignment=1,))
        data.setSubImageOn(self, size=image.size)
    imageSize = (0., 0., 0.)

    _geo = None
    def geometry(self, geo=None):
        if geo is None:
            geo = self._geo
        if geo is None:
            geo = self.GeometryFactory.fromCount(1)
            self._geo = geo

        geo['v'] = self.verticies()
        geo['t'] = self.texCoords()
        return geo

    def verticies(self, geov=None):
        iw, ih, id = self.imageSize
        return [[0., 0., id], [iw, 0., id], [iw, ih, id], [0., ih, id]]

    def texCoords(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTexture2d(ImageTextureBase):
    texParams = ImageTextureBase.texParams.copy()
    texParams.update(target=gl.GL_TEXTURE_2D)

    def texCoords(self):
        w, h = self.size
        iw, ih, id = self.imageSize
        iw = float(iw)/w; ih = float(ih)/h
        return [[0., ih], [iw, ih], [iw, 0.], [0., 0.]]

class ImageTextureRect(ImageTextureBase):
    texParams = ImageTextureBase.texParams.copy()
    texParams.update(target=glext.GL_TEXTURE_RECTANGLE_ARB)

    def texCoords(self):
        iw, ih, id = self.imageSize
        return [[0., ih], [iw, ih], [iw, 0.], [0., 0.]]

ImageTexture = ImageTextureRect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageDisplay(object):
    def __init__(self, *args, **kw):
        if args or kw:
            self.update(*args, **kw)

    def update(self, imageObj, imageTexture, geometry):
        self.texture = imageTexture
        self.geometry = geometry

    def render(self):
        self.texture.select()
        self.geometry.draw()
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageObject(object):
    ImageTextureFactory = ImageTextureRect
    DisplayFactory = ImageDisplay

    def __init__(self, image=None, format=True, **kwattr):
        if image is not None:
            self.load(image, format)

        self.set(kwattr)

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _pos = None
    def getPos(self):
        pos = self._pos
        if pos is None:
            self.setPos((0., 0., 0.))
            pos = self._pos
        return pos
    def setPos(self, pos, doUpdate=False):
        self._pos = asarray(pos, float32)
        if doUpdate:
            self.update()
    pos = property(getPos, setPos)

    _size = None
    def getSize(self):
        size = self._size
        if size is None:
            self.setSize((0., 0., 0.))
            size = self._size
        return size
    def setSize(self, size, doUpdate=True):
        self._size = asarray(size, float32)
        if doUpdate:
            self.update()
    size = property(getSize, setSize)

    _align = None
    def getAlign(self):
        align = self._align
        if align is None:
            self.setAlign((0., 0., 0.))
            align = self._align
        return align
    def setAlign(self, align, doUpdate=True):
        if isinstance(align, (int, long, float)):
            align = (align, align, align)
        self._align = asarray(align, float32)
        if doUpdate:
            self.update()
    align = property(getAlign, setAlign)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update(self):
        image = self.image
        geo = image.geometry()

        size = self.size
        size[:] = image.imageSize
        off = self.pos - (self.align*self.size)

        geo['v'] += off

        self.display.update(self, image, geo)
        return True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load(self, *args, **kw):
        self.image.load(*args, **kw)
        self.update()
    def loadImage(self, *args, **kw):
        self.image.loadImage(*args, **kw)
        self.update()
    def loadFilename(self, *args, **kw):
        self.image.loadFilename(*args, **kw)
        self.update()

    _image = None
    def getImage(self):
        image = self._image
        if image is None:
            self.createImage()
        return self._image
    def setImage(self, image, doUpdate=True):
        self._image = image
        if doUpdate:
            self.update()
    image = property(getImage, setImage)

    def createImage(self):
        image = self.ImageTextureFactory()
        self.setImage(image, False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _display = None
    def getDisplay(self):
        display = self._display
        if display is None:
            display = self.DisplayFactory()
            self._display = display
        return display
    def setDisplay(self, display, doUpdate=False):
        self._display = display
        if doUpdate:
            self.update()
    display = property(getDisplay, setDisplay)

    def render(self):
        self.display.render()

