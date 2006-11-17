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

from .shapes import PositionalObject
from .texture import Texture
from .data import interleavedArrays

from .raw import gl, glext

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageGeometryArray(interleavedArrays.InterleavedArrays):
    drawMode = gl.GL_QUADS
    dataFormat = gl.GL_T2F_V3F

    @classmethod
    def fromCount(klass, count):
        return klass.fromFormat((count, 4), klass.dataFormat)

    @classmethod
    def fromSingle(klass):
        return klass.fromFormat(4, klass.dataFormat)

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

    def geometry(self, geo=None):
        if geo is None:
            geo = self.GeometryFactory.fromSingle()

        geo['v'] = self.verticies()
        geo['t'] = self.texCoords()
        return geo

    def verticies(self):
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

class ImageObject(PositionalObject):
    ImageTextureFactory = ImageTextureRect

    def __init__(self, image=None, format=True, **kwattr):
        if kwattr: 
            self.set(kwattr)
        if image is not None:
            self.load(image, format)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update(self, **kwattr):
        if kwattr: 
            self.set(kwattr)

        image = self.image
        geo = image.geometry()

        size = self.size
        size[:] = image.imageSize
        off = self.pos - (self.align*size)

        if self.roundValues:
            geo['v'] += off.round()
        else:
            geo['v'] += off

        self.geometry = geo
        return True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load(self, image=None, format=True, **kwattr):
        self.image.load(image, format)
        if kwattr: 
            self.set(kwattr)
        self.update()
    def loadImage(self, image=None, format=True, **kwattr):
        self.image.loadImage(image, format)
        if kwattr: 
            self.set(kwattr)
        self.update()
    def loadFilename(self, image=None, format=True, **kwattr):
        self.image.loadFilename(image, format)
        if kwattr: 
            self.set(kwattr)
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

    def render(self):
        self.image.select()
        self.color.select()
        self.geometry.draw()

