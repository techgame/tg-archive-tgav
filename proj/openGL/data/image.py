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
from .texture import Texture, TextureCoord, TextureCoordArray, VertexArray

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTextureBase(Texture):
    texParams = Texture.texParams + [
            ('wrap', gl.GL_CLAMP_TO_EDGE),

            ('genMipmaps', True),
            ('magFilter', gl.GL_LINEAR),
            ('minFilter', gl.GL_LINEAR),#_MIPMAP_LINEAR),
            ]

    imgModeFormatMap = {
        'RGBA': (gl.GL_RGBA, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
        'RGB': (gl.GL_RGB, gl.GL_RGB, gl.GL_UNSIGNED_BYTE),
        'LA': (gl.GL_LUMINANCE_ALPHA, gl.GL_LUMINANCE_ALPHA, gl.GL_UNSIGNED_BYTE),
        'L': (gl.GL_LUMINANCE, gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE),
        }

    vertexScale = {
        2: VertexArray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], '2f'),
        3: VertexArray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], '3f'),
        }
    texCoordScale = {
        (2, 'normal'): TextureCoordArray([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], '2f'),
        (3, 'normal'): TextureCoordArray([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], '3f'),

        (2, 'flip'): TextureCoordArray([[0., 1.], [1., 1.], [1., 0.], [0., 0.]], '2f'),
        (3, 'flip'): TextureCoordArray([[0., 1., 0.], [1., 1., 0.], [1., 0., 0.], [0., 0., 0.]], '3f'),
        }

    imageSize = TextureCoord.property([0., 0., 0.])

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
    imageFilename = property(fset=loadFilename)

    def loadImage(self, image, format=True):
        impliedFormat, dataFormat, dataType = self.imgModeFormatMap[image.mode]
        if format is True:
            self.format = impliedFormat
        elif format:
            self.format = format

        self.select()
        size = self.validSizeForTarget(image.size+(0,))
        data = self.data2d(size=size, format=dataFormat, dataType=dataType)
        assert data.pos.max() == 0, data.pos
        assert (data.size == size).all(), data.size
        data.setImageOn(self)

        data.texString(image.tostring(), dict(alignment=1,))
        data.setSubImageOn(self, size=image.size+(0,))
        self.imageSize = data.size.copy()
    image = property(fset=loadImage)

    def verticesForImage(self, components=3):
        scale = self.vertexScale[components]
        return self.imageSize[:components] * scale

    def texCoordsForImage(self, components=2, key='flip'):
        scale = self.texCoordScale[components, key]
        adjSize = self.imageSize[:components]
        if self._rstNormalizeTargets.get(self.target, True):
            adjSize -= 1
        return self.texCoordsFor(adjSize*scale)

    def texCoordsForRect(self, rect, components=2, key='flip'):
        return self.texCoordsForPosSize(rect.pos, rect.size, components, key)
    def texCoordsForPosSize(self, pos, size, components=2, key='flip'):
        scale = self.texCoordScale[components, key]
        result = size[:components]*scale + pos[:components]
        result = self.texCoordsFor(result)
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTexture2d(ImageTextureBase):
    texParams = ImageTextureBase.texParams + [
                    ('target', gl.GL_TEXTURE_2D), ]

class ImageTextureRect(ImageTextureBase):
    texParams = ImageTextureBase.texParams + [
                    ('target', glext.GL_TEXTURE_RECTANGLE_ARB), ]

