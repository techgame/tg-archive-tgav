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

    def loadFilename(self, filename, changeFormat=True):
        image = Image.open(filename)
        return self.loadImage(image, changeFormat)

    def loadImage(self, image, changeFormat=True):
        format, dataFormat, dataType = self.imgModeFormatMap[image.mode]
        if changeFormat:
            self.format = format

        self.imageSize = image.size
        size = self.validSizeForTarget(image.size)
        data = self.data2d(size=size, format=dataFormat, dataType=dataType)
        data.setImageOn(self)

        data.texString(image.tostring(), dict(alignment=1,))
        data.setSubImageOn(self, size=image.size)
        self.recompile()
    imageSize = (0, 0)

    def recompile(self):
        if self._geo is not None:
            if self._geo['v'].mean():
                self.geometry(self._geo)
            else:
                self.centerGeometry(self._geo)

    def centerGeometry(self, geo=None):
        geo = self.geometry(geo)
        geo['v'] = geo['v'] - geo['v'].mean(0)

    _geo = None
    def geometry(self, geo=None):
        if geo is None:
            geo = self._geo
        if geo is None:
            geo = self.GeometryFactory.fromCount(1)
            self._geo = geo

        self._setVerticies(geo['v'])
        self._setTexCoords(geo['t'])
        return geo

    def _setVerticies(self, geov):
        iw, ih = self.imageSize
        geov[:] = [[0., 0., 0.],
                   [iw, 0., 0.],
                   [iw, ih, 0.],
                   [0., ih, 0.]]

    def _setTexCoords(self, geot):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def render(self):
        self.select()
        self._geo.draw()
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTexture2d(ImageTextureBase):
    target = glext.GL_TEXTURE_2D

    def _setTexCoords(self, geot):
        w, h = self.size
        iw, ih = self.imageSize
        iw = float(iw)/w
        ih = float(ih)/h

        geot[:] = [[0., ih],
                   [iw, ih],
                   [iw, 0.],
                   [0., 0.]]

class ImageTextureRect(ImageTextureBase):
    target = glext.GL_TEXTURE_RECTANGLE_ARB

    def _setTexCoords(self, geot):
        iw, ih = self.imageSize
        geot[:] = [[0., ih],
                   [iw, ih],
                   [iw, 0.],
                   [0., 0.]]

ImageTexture = ImageTextureRect

