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

    target = None
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

    def create(self, image, geometry=True, **kwattrs):
        Texture.create(self, **kwattrs)

        if image is not None:
            if isinstance(image, basestring):
                self.loadFilename(image)
            else:
                self.loadImage(image)

        if geometry:
            self.geometry()

    def loadFilename(self, filename, format=True):
        image = Image.open(filename)
        return self.loadImage(image, format)

    def loadImage(self, image, texFormat=True):
        impliedFormat, dataFormat, dataType = self.imgModeFormatMap[image.mode]
        if texFormat is True:
            self.format = impliedFormat
        elif texFormat:
            self.format = texFormat

        self.imageSize = image.size
        size = self.validSizeForTarget(image.size)
        data = self.data2d(size=size, format=dataFormat, dataType=dataType)
        data.setImageOn(self)

        data.texString(image.tostring(), dict(alignment=1,))
        data.setSubImageOn(self, size=image.size)
        self.recompile()
    imageSize = (0, 0)

    _geo = None
    def geometry(self, geo=None):
        if geo is None:
            geo = self._geo
        if geo is None:
            geo = self.GeometryFactory.fromCount(1)
            self._geo = geo

        self.verticies(geo['v'])
        self.texCoords(geo['t'])
        return geo

    def verticies(self, geov):
        iw, ih = self.imageSize
        geov[:] = [[0., 0., 0.],
                   [iw, 0., 0.],
                   [iw, ih, 0.],
                   [0., ih, 0.]]

    def texCoords(self, geot):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTexture2d(ImageTextureBase):
    target = glext.GL_TEXTURE_2D

    def texCoords(self, geot):
        w, h = self.size
        iw, ih = self.imageSize
        iw = float(iw)/w; ih = float(ih)/h
        geot[:] = [[0., ih], [iw, ih], [iw, 0.], [0., 0.]]

class ImageTextureRect(ImageTextureBase):
    target = glext.GL_TEXTURE_RECTANGLE_ARB

    def texCoords(self, geot):
        iw, ih = self.imageSize
        geot[:] = [[0., ih], [iw, ih], [iw, 0.], [0., 0.]]

ImageTexture = ImageTextureRect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageObject(object):
    def __init__(self, image, texFormat=True):
        pass
    def render(self):
        self.select()
        self._geo.draw()

class ImageDisplay(object):
    def __init__(self, *args, **kw):
        if args or kw:
            self.update(*args, **kw)

    def update(self, imageObj, imageTexture, geometry):
        self.geometry = geometry
        self.texture = imageTexture

    def render(self):
        self.texture.select()
        self.geometry.draw()
    __call__ = render

