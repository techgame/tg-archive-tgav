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

from ..raw import gl, glext

from .texture import Texture
from .singleArrays import Vector
from .interleavedArrays import InterleavedArray

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageGeometryArray(InterleavedArray):
    drawMode = gl.GL_QUADS
    gldtype = InterleavedArray.gldtype.copy()
    gldtype.setDefaultFormat(gl.GL_T2F_V3F)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTextureBase(Texture):
    GeometryFactory = ImageGeometryArray
    premultiply = False

    texParams = Texture.texParams + [
            ('wrap', gl.GL_CLAMP_TO_EDGE),

            ('genMipmaps', True),
            ('magFilter', gl.GL_LINEAR),
            ('minFilter', gl.GL_LINEAR_MIPMAP_LINEAR),
            ]

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

    openImage = staticmethod(Image.open)
    def loadFilename(self, filename, format=True):
        image = self.openImage(filename)
        return self.loadImage(image, format)

    def loadImage(self, image, format=True):
        impliedFormat, dataFormat, dataType = self.imgModeFormatMap[image.mode]
        if format is True:
            self.format = impliedFormat
        elif format:
            self.format = format

        image = self._preRenderTexture(image)
        self.imageSize = image.size + (0.,)
        size = self.validSizeForTarget(image.size)
        data = self.data2d(size=size, format=dataFormat, dataType=dataType)
        data.setImageOn(self)

        data.texString(image.tostring(), dict(alignment=1,))
        data.setSubImageOn(self, size=image.size)
    imageSize = (0., 0., 0.)

    def _preRenderTexture(self, image):
        if self.premultiply:
            bands = image.getbands()
            assert bands[-1] == 'A', bands

            imageData = image.getdata()

            a = imageData.getband(len(bands)-1)
            
            for idx in xrange(len(bands)-1):
                premult = a.chop_multiply(imageData.getband(idx))
                imageData.putband(premult, idx)
        return image

    def geometry(self, geo=None):
        if geo is None:
            geo = self.GeometryFactory.fromShape((1,4,-1))

        geo.v = self.verticies()
        geo.t = self.texCoords()
        return geo

    def verticies(self):
        iw, ih, id = self.imageSize
        return [[0., 0., id], [iw, 0., id], [iw, ih, id], [0., ih, id]]

    def texCoords(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTexture2d(ImageTextureBase):
    texParams = ImageTextureBase.texParams + [
                    ('target', gl.GL_TEXTURE_2D), ]

    def texCoords(self, flip=True):
        w, h = self.size
        iw, ih, id = self.imageSize
        iw = float(iw)/w; ih = float(ih)/h
        if flip:
            return [[0., ih], [iw, ih], [iw, 0.], [0., 0.]]
        else:
            return [[0., 0.], [iw, 0.], [iw, ih], [0., ih]]

class ImageTextureRect(ImageTextureBase):
    texParams = ImageTextureBase.texParams + [
                    ('target', glext.GL_TEXTURE_RECTANGLE_ARB), ]

    def texCoords(self, flip=True):
        iw, ih, id = self.imageSize
        if flip:
            return [[0., ih], [iw, ih], [iw, 0.], [0., 0.]]
        else:
            return [[0., 0.], [iw, 0.], [iw, ih], [0., ih]]

ImageTexture = ImageTextureRect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageObject(object):
    ImageTextureFactory = ImageTextureRect

    roundValues = True

    def __init__(self, image=None, format=True, **kwattr):
        self._pos = self._pos.copy()
        self._size = self._size.copy()
        self._align = self._align.copy()

        if kwattr: 
            self.set(kwattr)
        if image is not None:
            self.load(image, format)

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _pos = Vector([0., 0., 0.], 'f')
    def getPos(self): return self._pos
    def setPos(self, pos): self._pos.set(pos)
    pos = property(getPos, setPos)

    _size = Vector([0., 0., 0.], 'f')
    def getSize(self): return self._size
    def setSize(self, size): self._size.set(size)
    size = property(getSize, setSize)

    _align = Vector([0., 0., 0.], 'f')
    def getAlign(self): return self._align
    def setAlign(self, align): self._align.set(align)
    align = property(getAlign, setAlign)

    def update(self, **kwattr):
        if kwattr: 
            self.set(kwattr)

        image = self.image
        geo = image.geometry()

        size = self.size
        size[:] = image.imageSize
        off = self.pos - (self.align*size)

        geo.v += (off.round() if self.roundValues else off)

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

    def getPremultiply(self):
        return self.image.premultiply
    def setPremultiply(self, premultiply):
        self.image.premultiply = premultiply
    premultiply = property(getPremultiply, setPremultiply)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def render(self):
        self.image.select()
        #self.color.select()
        geom = self.geometry
        gl.glInterleavedArrays(geom.glTypeId, 0, geom.ctypes)
        gl.glDrawArrays(geom.drawMode, 0, geom.size)

