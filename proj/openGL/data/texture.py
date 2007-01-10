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

import sys
from bisect import bisect_left
from numpy import asarray
from ctypes import cast, byref, c_void_p

from ..raw import gl, glext
from ..raw.errors import  GLError

from .vertexArrays import TextureCoordArray, VertexArray
from .singleArrays import TextureCoord, Vertex

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class glTexParamProperty(object):
    _as_parameter_ = None
        
    def __init__(self, propertyEnum):
        self._as_parameter_ = gl.GLenum(propertyEnum)

    def __get__(self, obj, klass):
        if obj is None: 
            return self

        cValue = self.GLParamType()
        self.glGetTexParameter(obj.target, self, cValue)
        return cValue.value

    def __set__(self, obj, value):
        if not isinstance(value, (list, tuple)):
            value = (value,)
        cValue = self.GLParamType(*value)
        self.glSetTexParameter(obj.target, self, cValue)

class glTexParamProperty_i(glTexParamProperty):
    GLParamType = (gl.GLint*1)
    glGetTexParameter = property(lambda self: gl.glGetTexParameteriv)
    glSetTexParameter = property(lambda self: gl.glTexParameteriv)

class glTexParamProperty_f(glTexParamProperty):
    GLParamType = (gl.GLfloat*1)
    glGetTexParameter = property(lambda self: gl.glGetTexParameteriv)
    glSetTexParameter = property(lambda self: gl.glTexParameteriv)

class glTexParamProperty_4f(glTexParamProperty):
    GLParamType = (gl.GLfloat*4)
    glGetTexParameter = property(lambda self: gl.glGetTexParameterfv)
    glSetTexParameter = property(lambda self: gl.glTexParameterfv)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PixelStore(object):
    alignment = 4
    swapBytes = False
    lsbFirst = False

    rowLength = imgHeight = 0 # 0 means use for row length to image height
    skipPixels = skipRows = skipImages = 0

    fmtPNames = {
        'alignment': (gl.GL_UNPACK_ALIGNMENT, gl.GL_PACK_ALIGNMENT),
        'swapBytes': (gl.GL_UNPACK_SWAP_BYTES, gl.GL_PACK_SWAP_BYTES),
        'lsbFirst': (gl.GL_UNPACK_LSB_FIRST, gl.GL_PACK_LSB_FIRST),

        'rowLength': (gl.GL_UNPACK_ROW_LENGTH, gl.GL_PACK_ROW_LENGTH),
        'imgHeight': (gl.GL_UNPACK_IMAGE_HEIGHT, gl.GL_PACK_IMAGE_HEIGHT),
                                        
        'skipPixels': (gl.GL_UNPACK_SKIP_ROWS, gl.GL_PACK_SKIP_ROWS),
        'skipRows': (gl.GL_UNPACK_SKIP_PIXELS, gl.GL_PACK_SKIP_PIXELS),
        'skipImages': (gl.GL_UNPACK_SKIP_IMAGES, gl.GL_PACK_SKIP_IMAGES),
    }
    formatAttrs = None # a list of (pname, newValue, origValue)
    pack = False

    def __init__(self, **kwattrs):
        self.create(**kwattrs)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __setattr__(self, name, value):
        if name in self.fmtPNames:
            self[name] = value
        return object.__setattr__(self, name, value)

    def __getitem__(self, name):
        pname = self.fmtPNames[name][self.pack]
        return self.formatAttrs[pname]
    def __setitem__(self, name, value):
        pname = self.fmtPNames[name][self.pack]
        restoreValue = getattr(self.__class__, name)
        self.formatAttrs[pname] = (value, restoreValue)
        object.__setattr__(self, name, value)
    def __delitem__(self, name):
        pname = self.fmtPNames[name][self.pack]
        del self.formatAttrs[pname]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def create(self, pack=False, **kwattrs):
        self.formatAttrs = {}
        self.pack = pack
        self.set(kwattrs)

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    def select(self):
        for pname, value in self.formatAttrs.iteritems():
            gl.glPixelStorei(pname, value[0])
        return self

    def deselect(self):
        for pname, value in self.formatAttrs.iteritems():
            gl.glPixelStorei(pname, value[1])
        return self

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Texture Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextureImageBasic(object):
    formatMap = {
        'rgb': gl.GL_RGB, 'RGB': gl.GL_RGB,
        'rgba': gl.GL_RGBA, 'RGBA': gl.GL_RGBA,

        'bgr': gl.GL_BGR, 'BGR': gl.GL_BGR,
        'bgra': gl.GL_BGRA, 'BGRA': gl.GL_BGRA,

        'ci': gl.GL_COLOR_INDEX, 'color_index': gl.GL_COLOR_INDEX,
        'si': gl.GL_STENCIL_INDEX, 'stencil_index': gl.GL_STENCIL_INDEX,
        'dc': gl.GL_DEPTH_COMPONENT, 'depth': gl.GL_DEPTH_COMPONENT, 'depth_component': gl.GL_DEPTH_COMPONENT,

        'r': gl.GL_RED, 'R': gl.GL_RED, 'red': gl.GL_RED,
        'g': gl.GL_GREEN, 'G': gl.GL_GREEN, 'green': gl.GL_GREEN,
        'b': gl.GL_BLUE, 'B': gl.GL_BLUE, 'blue': gl.GL_BLUE,
        'a': gl.GL_ALPHA, 'A': gl.GL_ALPHA, 'alpha': gl.GL_ALPHA,

        'l': gl.GL_LUMINANCE, 'L': gl.GL_LUMINANCE, 'luminance': gl.GL_LUMINANCE,
        'la': gl.GL_LUMINANCE_ALPHA, 'LA': gl.GL_LUMINANCE_ALPHA, 'luminance_alpha': gl.GL_LUMINANCE_ALPHA,
    }

    dataTypeMap = {
        'bitmap': gl.GL_BITMAP,

        'ub': gl.GL_UNSIGNED_BYTE, 'ubyte': gl.GL_UNSIGNED_BYTE, 'B': gl.GL_UNSIGNED_BYTE,
        'b': gl.GL_BYTE, 'byte': gl.GL_BYTE,

        'us': gl.GL_UNSIGNED_SHORT, 'ushort': gl.GL_UNSIGNED_SHORT, 'S': gl.GL_UNSIGNED_SHORT,
        's': gl.GL_SHORT, 'short': gl.GL_SHORT,

        'ui': gl.GL_UNSIGNED_INT, 'uint': gl.GL_UNSIGNED_INT, 'I': gl.GL_UNSIGNED_INT,
        'ul': gl.GL_UNSIGNED_INT, 'ulong': gl.GL_UNSIGNED_INT, 'L': gl.GL_UNSIGNED_INT,
        'i': gl.GL_INT, 'int': gl.GL_INT, 'l': gl.GL_INT, 'long': gl.GL_INT,

        'f': gl.GL_FLOAT, 'f32': gl.GL_FLOAT, 'float': gl.GL_FLOAT, 'float32': gl.GL_FLOAT,

        'ub332': gl.GL_UNSIGNED_BYTE_3_3_2, 'ub233r': gl.GL_UNSIGNED_BYTE_2_3_3_REV,
        'us565': gl.GL_UNSIGNED_SHORT_5_6_5, 'us565r': gl.GL_UNSIGNED_SHORT_5_6_5_REV,
        'us4444': gl.GL_UNSIGNED_SHORT_4_4_4_4, 'us4444r': gl.GL_UNSIGNED_SHORT_4_4_4_4_REV,
        'us5551': gl.GL_UNSIGNED_SHORT_5_5_5_1, 'us1555r': gl.GL_UNSIGNED_SHORT_1_5_5_5_REV,
        'ui8888': gl.GL_UNSIGNED_INT_8_8_8_8, 'ui8888r': gl.GL_UNSIGNED_INT_8_8_8_8_REV, 
        'uiAAA2': gl.GL_UNSIGNED_INT_10_10_10_2, 'ui2AAAr': gl.GL_UNSIGNED_INT_2_10_10_10_REV, }
    border = False

    _dataTypeSizeMap = {
        1: 1, 2: 2, 3: 3, 4: 4,

        gl.GL_UNSIGNED_BYTE: 1,
        gl.GL_BYTE: 1,
        gl.GL_BITMAP: 1,
        gl.GL_UNSIGNED_SHORT: 2,
        gl.GL_SHORT: 2,
        gl.GL_UNSIGNED_INT: 4,
        gl.GL_INT: 4,
        gl.GL_FLOAT: 4,
        gl.GL_UNSIGNED_BYTE_3_3_2: 1,
        gl.GL_UNSIGNED_BYTE_2_3_3_REV: 1,
        gl.GL_UNSIGNED_SHORT_5_6_5: 2,
        gl.GL_UNSIGNED_SHORT_5_6_5_REV: 2,
        gl.GL_UNSIGNED_SHORT_4_4_4_4: 2,
        gl.GL_UNSIGNED_SHORT_4_4_4_4_REV: 2,
        gl.GL_UNSIGNED_SHORT_5_5_5_1: 2,
        gl.GL_UNSIGNED_SHORT_1_5_5_5_REV: 2,
        gl.GL_UNSIGNED_INT_8_8_8_8: 4,
        gl.GL_UNSIGNED_INT_8_8_8_8_REV: 4,
        gl.GL_UNSIGNED_INT_10_10_10_2: 4,
        gl.GL_UNSIGNED_INT_2_10_10_10_REV: 4,
        }

    def __init__(self, *args, **kwattrs):
        super(TextureImageBasic, self).__init__()

        self.create(*args, **kwattrs)

    def create(self, *args, **kwattrs):
        self.set(kwattrs)

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    def getDataTypeSize(self):
        return self._dataTypeSizeMap[self.dataType]

    def getSizeInBytes(self):
        byteSize = self.getDataTypeSize()

        if self._pixelStore:
            alignment = self._pixelStore.alignment
            byteSize += (alignment - byteSize) % alignment

        border = self.border and 2 or 0

        for e in self.size:
            if e > 1:
                byteSize *= e + border
        return byteSize

    _dataType = None
    def getDataType(self):
        return self._dataType
    def setDataType(self, dataType):
        self._dataType = dataType
    # dataType should be: gl.GL_UNSIGNED_BYTE, gl.GL_BYTE, gl.GL_BITMAP, gl.GL_UNSIGNED_SHORT, gl.GL_SHORT, gl.GL_UNSIGNED_INT, gl.GL_INT, gl.GL_FLOAT, ...
    dataType = property(getDataType, setDataType)

    _format = None
    def getFormat(self):
        return self._format
    def setFormat(self, format):
        if isinstance(format, basestring):
            format = self.formatMap[format]
        self._format = format
    # format should be: gl.GL_COLOR_INDEX,  gl.GL_RED, gl.GL_GREEN, gl.GL_BLUE, gl.GL_ALPHA,  gl.GL_RGB,  gl.GL_BGR  gl.GL_RGBA, gl.GL_BGRA, gl.GL_LUMINANCE, gl.GL_LUMINANCE_ALPHA
    format = property(getFormat, setFormat)

    _pointer = None
    def getPointer(self):
        return self._pointer
    ptr = property(getPointer)

    def clear(self):
        self.newPixelStore()
        self.texClear()

    _rawData = None
    def texData(self, rawData, pointer, pixelStoreSettings):
        if pixelStoreSettings:
            self.updatePixelStore(pixelStoreSettings)
        self._rawData = rawData
        if not pointer:
            self._pointer = c_void_p(None)
        elif isinstance(pointer, (int, long)):
            self._pointer = c_void_p(pointer)
        else:
            self._pointer = cast(pointer, c_void_p)

    def texClear(self, pixelStoreSettings=None):
        self.texData(None, None, pixelStoreSettings)
    texNull = texClear
    def texString(self, strdata, pixelStoreSettings=None):
        self.texData(strdata, strdata, pixelStoreSettings)
    strdata = property(fset=texString)
    def texCData(self, cdata, pixelStoreSettings=None):
        self.texData(cdata, cdata, pixelStoreSettings)
    cdata = property(fset=texCData)
    def texArray(self, array, pixelStoreSettings=None):
        self.texData(array, array.ctypes, pixelStoreSettings)
    array = property(fset=texArray)

    def texBlank(self):
        # setup the alignment properly
        ps = self.newPixelStore(alignment=1)
        byteCount = self.getSizeInBytes()
        data = (gl.GLubyte*byteCount)()
        self.texCData(data)

    def setImageOn(self, texture, level=0, **kw):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def setSubImageOn(self, texture, level=0, **kw):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def setCompressedImageOn(self, texture, level=0, **kw):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def setCompressedSubImageOn(self, texture, level=0, **kw):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _pixelStore = None
    def getPixelStore(self):
        if self._pixelStore is None:
            self.newPixelStore()
        return self._pixelStore
    def setPixelStore(self, pixelStore):
        if isinstance(pixelStore, dict):
            self.updatePixelStore(pixelStore)
        self._pixelStore = pixelStore
    def delPixelStore(self):
        if self._pixelStore is not None:
            del self._pixelStore
    pixelStore = property(getPixelStore, setPixelStore, delPixelStore)

    def newPixelStore(self, *args, **kw):
        pixelStore = PixelStore(*args, **kw)
        self.setPixelStore(pixelStore)
        return pixelStore
    def updatePixelStore(self, settings):
        ps = self.getPixelStore()
        ps.set(settings)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def select(self):
        pixelStore = self._pixelStore
        if pixelStore is not None:
            pixelStore.select()
        return self

    def deselect(self):
        pixelStore = self._pixelStore
        if pixelStore is not None:
            pixelStore.deselect()
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pos = TextureCoord.property([0, 0, 0], dtype='3l')
    size = TextureCoord.property([0, 0, 0], dtype='3l')

    def getPosSize(self):
        return (self.pos, self.size)
    def setPosSize(self, pos, size=None):
        if size is None:
            pos, size = pos
        self.pos.set(pos)
        self.size.set(size)
    posSize = property(getPosSize, setPosSize)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextureImage1D(TextureImageBasic):
    def setImageOn(self, texture, level=0, **kw):
        texture.setImage1d(self, level, **kw)
        return texture
    def setSubImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setSubImage1d(self, level)
        return texture
    def setCompressedImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setCompressedImage1d(self, level)
        return texture
    def setCompressedSubImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setCompressedSubImage1d(self, level)
        return texture

class TextureImage2D(TextureImageBasic):
    def setImageOn(self, texture, level=0, **kw):
        texture.setImage2d(self, level, **kw)
        return texture
    def setSubImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setSubImage2d(self, level)
        return texture
    def setCompressedImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setCompressedImage2d(self, level)
        return texture
    def setCompressedSubImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setCompressedSubImage2d(self, level)
        return texture

class TextureImage3D(TextureImageBasic):
    def setImageOn(self, texture, level=0, **kw):
        texture.setImage3d(self, level, **kw)
        return texture
    def setSubImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setSubImage3d(self, level)
        return texture
    def setCompressedImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setCompressedImage3d(self, level)
        return texture
    def setCompressedSubImageOn(self, texture, level=0, **kw):
        if kw: self.set(kw)
        texture.setCompressedSubImage3d(self, level)
        return texture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Texture object itself
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Texture(object):
    _as_parameter_ = None # GLenum returned from glGenTextures
    texParams = []
    targetMap = {
        '1d': gl.GL_TEXTURE_1D, '1D': gl.GL_TEXTURE_1D,
        'proxy-1d': gl.GL_PROXY_TEXTURE_1D, 'proxy-1D': gl.GL_PROXY_TEXTURE_1D,
        '2d': gl.GL_TEXTURE_2D, '2D': gl.GL_TEXTURE_2D,
        'proxy-2d': gl.GL_PROXY_TEXTURE_2D, 'proxy-2D': gl.GL_PROXY_TEXTURE_2D,
        '3d': gl.GL_TEXTURE_3D, '3D': gl.GL_TEXTURE_3D,
        'proxy-3d': gl.GL_PROXY_TEXTURE_3D, 'proxy-3D': gl.GL_PROXY_TEXTURE_3D,

        'rect': glext.GL_TEXTURE_RECTANGLE_ARB, 'proxy-rect': glext.GL_PROXY_TEXTURE_RECTANGLE_ARB,

        'cube': gl.GL_TEXTURE_CUBE_MAP, 'proxy-cube': gl.GL_PROXY_TEXTURE_CUBE_MAP,
        'cube+x': gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X, 'cube-x': gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        'cube+y': gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 'cube-y': gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        'cube+z': gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 'cube-z': gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    }
    formatMap = {
        1: 1, 2: 2, 3: 3, 4: 4, '1': 1, '2': 2, '3': 3, '4': 4,

        'rgb': gl.GL_RGB, 'RGB': gl.GL_RGB,
        'r3b3g2': gl.GL_R3_G3_B2, 'R3B3G2': gl.GL_R3_G3_B2,
        'rbg4': gl.GL_RGB4, 'RBG4': gl.GL_RGB4,
        'rbg5': gl.GL_RGB4, 'RBG5': gl.GL_RGB4,
        'rbg8': gl.GL_RGB8, 'RBG8': gl.GL_RGB8,
        'rbg12': gl.GL_RGB12, 'RBG12': gl.GL_RGB12,
        'rbg16': gl.GL_RGB16, 'RBG16': gl.GL_RGB16,

        'rgba': gl.GL_RGBA, 'RGBA': gl.GL_RGBA,
        'rbga2': gl.GL_RGBA2, 'RBGA2': gl.GL_RGBA2,
        'rbga4': gl.GL_RGBA4, 'RBGA4': gl.GL_RGBA4,
        'rbga8': gl.GL_RGBA8, 'RBGA8': gl.GL_RGBA8,
        'rbga12': gl.GL_RGBA12, 'RBGA12': gl.GL_RGBA12,
        'rbga16': gl.GL_RGBA16, 'RBGA16': gl.GL_RGBA16,

        'rbg5a1': gl.GL_RGB5_A1, 'RBG5A1': gl.GL_RGB5_A1,
        'rbg10a2': gl.GL_RGB10_A2, 'RBG10A2': gl.GL_RGB10_A2,

        'a': gl.GL_ALPHA, 'A': gl.GL_ALPHA,'alpha': gl.GL_ALPHA,
        'a4': gl.GL_ALPHA4, 'A4': gl.GL_ALPHA4,
        'a8': gl.GL_ALPHA8, 'A8': gl.GL_ALPHA8,
        'a12': gl.GL_ALPHA12, 'A12': gl.GL_ALPHA12,
        'a16': gl.GL_ALPHA16, 'A16': gl.GL_ALPHA16,

        'l': gl.GL_LUMINANCE, 'L': gl.GL_LUMINANCE, 'luminance': gl.GL_LUMINANCE,
        'l4': gl.GL_LUMINANCE4, 'L4': gl.GL_LUMINANCE4, 
        'l8': gl.GL_LUMINANCE8, 'L8': gl.GL_LUMINANCE8, 
        'l12': gl.GL_LUMINANCE12, 'L12': gl.GL_LUMINANCE12, 
        'l16': gl.GL_LUMINANCE16, 'L16': gl.GL_LUMINANCE16, 

        'la': gl.GL_LUMINANCE_ALPHA, 'LA': gl.GL_LUMINANCE_ALPHA, 'luminance_alpha': gl.GL_LUMINANCE_ALPHA,
        'l6a2': gl.GL_LUMINANCE6_ALPHA2, 'L6A2': gl.GL_LUMINANCE6_ALPHA2, 
        'l12a4': gl.GL_LUMINANCE12_ALPHA4, 'L12A4': gl.GL_LUMINANCE12_ALPHA4, 
        'la4': gl.GL_LUMINANCE4_ALPHA4, 'LA4': gl.GL_LUMINANCE4_ALPHA4, 'l4a4': gl.GL_LUMINANCE4_ALPHA4, 'L4A4': gl.GL_LUMINANCE4_ALPHA4, 
        'la8': gl.GL_LUMINANCE8_ALPHA8, 'LA8': gl.GL_LUMINANCE8_ALPHA8, 'l8a8': gl.GL_LUMINANCE8_ALPHA8, 'L8A8': gl.GL_LUMINANCE8_ALPHA8, 
        'la12': gl.GL_LUMINANCE12_ALPHA12, 'LA12': gl.GL_LUMINANCE12_ALPHA12, 'l12a12': gl.GL_LUMINANCE12_ALPHA12, 'L12A12': gl.GL_LUMINANCE12_ALPHA12, 
        'la16': gl.GL_LUMINANCE16_ALPHA16, 'LA16': gl.GL_LUMINANCE16_ALPHA16, 'l16a16': gl.GL_LUMINANCE16_ALPHA16, 'L16A16': gl.GL_LUMINANCE16_ALPHA16, 

        'i': gl.GL_INTENSITY, 'I': gl.GL_INTENSITY, 'intensity': gl.GL_INTENSITY,
        'i4': gl.GL_INTENSITY4, 'I4': gl.GL_INTENSITY4, 'intensity4': gl.GL_INTENSITY4,
        'i8': gl.GL_INTENSITY8, 'I8': gl.GL_INTENSITY8, 'intensity8': gl.GL_INTENSITY8,
        'i12': gl.GL_INTENSITY12, 'I12': gl.GL_INTENSITY12, 'intensity12': gl.GL_INTENSITY12,
        'i16': gl.GL_INTENSITY16, 'I16': gl.GL_INTENSITY16, 'intensity16': gl.GL_INTENSITY16,

        'c-rgb': gl.GL_COMPRESSED_RGB, 'C-RGB': gl.GL_COMPRESSED_RGB, 'compressed_rgb': gl.GL_COMPRESSED_RGB,
        'c-rgba': gl.GL_COMPRESSED_RGBA, 'C-RGBA': gl.GL_COMPRESSED_RGBA, 'compressed_rgba': gl.GL_COMPRESSED_RGBA,
        'c-a': gl.GL_COMPRESSED_ALPHA, 'C-A': gl.GL_COMPRESSED_ALPHA, 'compressed_alpha': gl.GL_COMPRESSED_ALPHA,
        'c-l': gl.GL_COMPRESSED_LUMINANCE, 'C-L': gl.GL_COMPRESSED_LUMINANCE, 'compressed_luminance': gl.GL_COMPRESSED_LUMINANCE,
        'c-la': gl.GL_COMPRESSED_LUMINANCE_ALPHA, 'C-LA': gl.GL_COMPRESSED_LUMINANCE_ALPHA, 'compressed_luminance_alpha': gl.GL_COMPRESSED_LUMINANCE_ALPHA,
        'c-i': gl.GL_COMPRESSED_INTENSITY, 'C-I': gl.GL_COMPRESSED_INTENSITY, 'compressed_intensity': gl.GL_COMPRESSED_INTENSITY, } 

    def __init__(self, *args, **kwargs):
        super(Texture, self).__init__()
        self.create(*args, **kwargs)

    def __del__(self):
        self.release()

    def create(self, target=None, **kwargs):
        if not self._as_parameter_ is None:
            raise Exception("Create has already been called for this instance")

        if target is None:
            for n,v in self.texParams:
                if n == 'target':
                    self.target = v
                    break

        self._genId()
        self.bind()
        self.set(self.texParams)
        self.set(kwargs)
        
    def set(self, val=None, **kwattr):
        if val:
            if isinstance(val, dict):
                val = val.iteritems()
            else: val = iter(val)
        else: val = kwattr.iteritems()

        for n,v in val:
            try:
                setattr(self, n, v)
            except GLError, e:
                print >> sys.stderr, '%r for name: %r value: %r' % (e, n, v)

    def release(self):
        if self._as_parameter_ is None:
            return

        self.unbind()
        self._delId()

    def _genId(self):
        if self._as_parameter_ is None:
            p = gl.GLenum(0)
            gl.glGenTextures(1, byref(p))
            self._as_parameter_ = p
    def _delId(self):
        p = self._as_parameter_
        if p is not None:
            gl.glDeleteTextures(1, byref(p))
            self._as_parameter_ = None

    def bind(self):
        gl.glBindTexture(self.target, self)
    def unbind(self):
        gl.glBindTexture(self.target, 0)

    def enable(self):
        gl.glEnable(self.target)
    def disable(self):
        gl.glDisable(self.target)

    def select(self, unit=None):
        if unit is not None:
            gl.glActiveTexture(unit)
        self.bind()
        self.enable()
        return self

    def deselect(self):
        self.disable()
        self.unbind()
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _target = 0 # gl.GL_TEXTURE_1D, gl.GL_TEXTURE_2D, etc.
    def getTarget(self):
        return self._target
    def setTarget(self, target):
        if isinstance(target, basestring):
            target = self.targetMap[target]
        self._target = target
    target = property(getTarget, setTarget)

    _format = None # gl.GL_RGBA, gl.GL_INTENSITY, etc.
    def getFormat(self):
        return self._format
    def setFormat(self, format):
        if isinstance(format, basestring):
            format = self.formatMap[format]
        self._format = format
    format = property(getFormat, setFormat)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    genMipmaps = glTexParamProperty_i(gl.GL_GENERATE_MIPMAP)

    wrapS = glTexParamProperty_i(gl.GL_TEXTURE_WRAP_S)
    wrapT = glTexParamProperty_i(gl.GL_TEXTURE_WRAP_T)
    wrapR = glTexParamProperty_i(gl.GL_TEXTURE_WRAP_R)

    def setWrap(self, wrap):
        if isinstance(wrap, tuple):
            wrapS, wrapT, wrapR = wrap
        else:
            wrapS = wrapT = wrapR = wrap
        self.wrapS = wrapS
        self.wrapT = wrapT
        self.wrapR = wrapR
    wrap = property(fset=setWrap)

    magFilter = glTexParamProperty_i(gl.GL_TEXTURE_MAG_FILTER)
    minFilter = glTexParamProperty_i(gl.GL_TEXTURE_MIN_FILTER)
    
    def setFilter(self, filter):
        if isinstance(filter, tuple):
            self.magFilter = filter[0]
            self.minFilter = filter[1]
        else:
            self.magFilter = filter
            self.minFilter = filter
    filter = property(fset=setFilter)

    baseLevel = glTexParamProperty_i(gl.GL_TEXTURE_BASE_LEVEL)
    maxLevel = glTexParamProperty_i(gl.GL_TEXTURE_MAX_LEVEL)

    depthMode = glTexParamProperty_i(gl.GL_DEPTH_TEXTURE_MODE)

    compareMode = glTexParamProperty_i(gl.GL_TEXTURE_COMPARE_MODE)
    compareFunc = glTexParamProperty_i(gl.GL_TEXTURE_COMPARE_FUNC)

    borderColor = glTexParamProperty_4f(gl.GL_TEXTURE_BORDER_COLOR)

    priority = glTexParamProperty_f(gl.GL_TEXTURE_PRIORITY)

    minLOD = glTexParamProperty_f(gl.GL_TEXTURE_MIN_LOD)
    maxLOD = glTexParamProperty_f(gl.GL_TEXTURE_MIN_LOD)
    biasLOD = glTexParamProperty_f(gl.GL_TEXTURE_LOD_BIAS)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Size & Dimensios
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    size = TextureCoord.property([0, 0, 0], dtype='3l')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Texture Data
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    TextureImage1DFactory = TextureImage1D
    @classmethod
    def data1d(klass, *args, **kw):
        return klass.TextureImage1DFactory(*args, **kw)

    TextureImage2DFactory = TextureImage2D
    @classmethod
    def data2d(klass, *args, **kw):
        return klass.TextureImage2DFactory(*args, **kw)

    TextureImage3DFactory = TextureImage3D
    @classmethod
    def data3d(klass, *args, **kw):
        return klass.TextureImage3DFactory(*args, **kw)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def blankImage(self, data, level=0, **kw):
        data.texBlank()
        data.setImageOn(self, level, **kw)
        data.texClear()
        return data

    def blankImage1d(self, *args, **kw):
        return self.blankImage(self.data1d(*args, **kw))
    def blankImage2d(self, *args, **kw):
        return self.blankImage(self.data2d(*args, **kw))
    def blankImage3d(self, *args, **kw):
        return self.blankImage(self.data3d(*args, **kw))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Tex Image Setting
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setImage(self, data, level=0, **kw):
        data.setImageOn(self, level, **kw)
        return data
    def setImage1d(self, data, level=0):
        data.select()
        try:
            size = data.size
            gl.glTexImage1D(self.target, level, self.format, 
                    size[0], data.border, 
                    data.format, data.dataType, data.ptr)
            self.size.set(size)
        finally:
            data.deselect()
        return data
    def setImage2d(self, data, level=0):
        data.select()
        try:
            size = data.size
            gl.glTexImage2D(self.target, level, self.format, 
                    size[0], size[1], data.border, 
                    data.format, data.dataType, data.ptr)

            self.size.set(size)
        finally:
            data.deselect()
        return data
    def setImage3d(self, data, level=0):
        data.select()
        try:
            size = data.size
            gl.glTexImage3D(self.target, level, self.format, 
                    size[0], size[1], size[2], data.border, 
                    data.format, data.dataType, data.ptr)
            self.size.set(size)
        finally:
            data.deselect()
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setSubImage(self, data, level=0, **kw):
        data.setSubImageOn(self, level, **kw)
        return self
    def setSubImage1d(self, data, level=0):
        data.select()
        try:
            pos = data.pos; size = data.size
            gl.glTexSubImage1D(self.target, level,
                    pos[0], 
                    size[0], 
                    data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setSubImage2d(self, data, level=0):
        data.select()
        try:
            pos = data.pos; size = data.size
            gl.glTexSubImage2D(self.target, level, 
                    pos[0], pos[1], 
                    size[0], size[1], 
                    data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setSubImage3d(self, data, level=0):
        data.select()
        try:
            pos = data.pos; size = data.size
            gl.glTexSubImage3D(self.target, level,
                    pos[0], pos[1], pos[2], 
                    size[0], size[1], size[2], 
                    data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setCompressedImage(self, data, level=0, **kw):
        data.setCompressedImageOn(self, level, **kw)
        return data
    def setCompressedImage1d(self, data, level=0):
        data.select()
        try:
            size = data.size
            gl.glCompressedTexImage1D(self.target, level, self.format, 
                    size[0], data.border, 
                    data.format, data.dataType, data.ptr)
            self.size.set(size)
        finally:
            data.deselect()
        return data
    def setCompressedImage2d(self, data, level=0):
        data.select()
        try:
            size = data.size
            gl.glCompressedTexImage2D(self.target, level, self.format, 
                    size[0], size[1], data.border, 
                    data.format, data.dataType, data.ptr)
            self.size.set(size)
        finally:
            data.deselect()
        return data
    def setCompressedImage3d(self, data, level=0):
        data.select()
        try:
            size = data.size
            gl.glCompressedTexImage3D(self.target, level, self.format, 
                    size[0], size[1], size[2], data.border, 
                    data.format, data.dataType, data.ptr)
            self.size.set(size)
        finally:
            data.deselect()
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setCompressedSubImage(self, data, level=0):
        data.setCompressedSubImageOn(self, level, **kw)
        return data
    def setCompressedSubImage1d(self, data, level=0):
        data.select()
        try:
            pos = data.pos; size = data.size
            gl.glCompressedTexSubImage1D(self.target, level,
                    pos[0],
                    size[0], 
                    data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setCompressedSubImage2d(self, data, level=0):
        data.select()
        try:
            pos = data.pos; size = data.size
            gl.glCompressedTexSubImage2D(self.target, level,
                    pos[0], pos[1], 
                    size[0], size[1], 
                    data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setCompressedSubImage3d(self, data, level=0):
        data.select()
        try:
            pos = data.pos; size = data.size
            gl.glCompressedTexSubImage3D(self.target, level, 
                    pos[0], pos[1], pos[2], 
                    size[0], size[1], size[2], 
                    data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _maxTextureSizeByTarget = {}

    _textureTargetToMaxPName = {
        gl.GL_TEXTURE_1D: gl.GL_MAX_TEXTURE_SIZE, gl.GL_PROXY_TEXTURE_1D: gl.GL_MAX_TEXTURE_SIZE,
        gl.GL_TEXTURE_2D: gl.GL_MAX_TEXTURE_SIZE, gl.GL_PROXY_TEXTURE_2D: gl.GL_MAX_TEXTURE_SIZE,
        gl.GL_TEXTURE_3D: gl.GL_MAX_3D_TEXTURE_SIZE, gl.GL_PROXY_TEXTURE_3D: gl.GL_MAX_3D_TEXTURE_SIZE,
        gl.GL_TEXTURE_CUBE_MAP: gl.GL_MAX_CUBE_MAP_TEXTURE_SIZE, gl.GL_PROXY_TEXTURE_CUBE_MAP: gl.GL_MAX_CUBE_MAP_TEXTURE_SIZE,
        glext.GL_TEXTURE_RECTANGLE_ARB: glext.GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB, glext.GL_PROXY_TEXTURE_RECTANGLE_ARB: glext.GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB,
        }

    def getMaxTextureSize(self):
        return self.getMaxTextureSizeFor(self.target)

    @classmethod
    def getMaxTextureSizeFor(klass, target):
        r = klass._maxTextureSizeByTarget.get(target, None)
        if r is None:
            pname = klass._textureTargetToMaxPName[target]
            i = gl.GLint(0)
            gl.glGetIntegerv(pname, byref(i))
            r = i.value
            klass._maxTextureSizeByTarget[target] = r
        return r

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _rstNormalizeTargets = {
        gl.GL_TEXTURE_1D: True,
        gl.GL_TEXTURE_2D: True,
        gl.GL_TEXTURE_3D: True,
        gl.GL_TEXTURE_CUBE_MAP: True,

        glext.GL_TEXTURE_RECTANGLE_ARB: False,
        }

    def texCoordsFor(self, texCoords):
        return self.texCoordsForData(texCoords, self.size, self.target)

    @classmethod
    def texCoordsForData(klass, texCoords, texSize, target):
        texSize = asarray(texSize)
        ndim = (texSize == 0).argmax()

        if klass._rstNormalizeTargets.get(target, True):
            return (texCoords[..., :ndim]/texSize[:ndim])
        else:
            return texCoords[..., :ndim]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _targetPowersOfTwo = {
        gl.GL_TEXTURE_1D: True,
        gl.GL_TEXTURE_2D: True,
        gl.GL_TEXTURE_3D: True,
        gl.GL_TEXTURE_CUBE_MAP: True,

        glext.GL_TEXTURE_RECTANGLE_ARB: False,
        }

    _powersOfTwo = [0] + [1<<s for s in xrange(31)]
    @staticmethod
    def idxNextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return bisect_left(powersOfTwo, v)
    @staticmethod
    def nextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return powersOfTwo[bisect_left(powersOfTwo, v)]
    del _powersOfTwo

    def validSizeForTarget(self, size):
        return self.validSizeForTargetData(size, self.target)
    @classmethod
    def validSizeForTargetData(klass, size, target):
        if klass._targetPowersOfTwo.get(target, False):
            size = [klass.nextPowerOf2(s) for s in size]

        return TextureCoord(size, copy=True)

