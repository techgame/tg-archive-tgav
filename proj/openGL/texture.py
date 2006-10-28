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

from bisect import bisect_left
import ctypes
from ctypes import cast, byref, c_void_p
from raw import gl, glu, glext

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class glTexParamProperty(object):
    _as_parameter_ = None
        
    def __init__(self, propertyEnum):
        self._as_parameter_ = propertyEnum

    glGetTexParameteriv = staticmethod(gl.glGetTexParameteriv)
    def __get__(self, obj, klass):
        if obj is None: 
            return self

        cValue = GLint(0)
        self.glGetTexParameteriv(obj.target, self, byref(cValue))
        return cValue.value

    glTexParameteri = staticmethod(gl.glTexParameteri)
    def __set__(self, obj, value):
        self.glTexParameteri(obj.target, self, value)

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

    def __init__(self, pack=False, **kwattrs):
        self.create(pack, **kwattrs)

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

    glPixelStorei = staticmethod(gl.glPixelStorei)
    def select(self):
        for pname, value in self.formatAttrs.iteritems():
            self.glPixelStorei(pname, value[0])
        return self

    def deselect(self):
        for pname, value in self.formatAttrs.iteritems():
            self.glPixelStorei(pname, value[1])
        return self

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Texture Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextureImageBasic(object):
    format = None # GL_COLOR_INDEX,  GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA,  GL_RGB,  GL_BGR  GL_RGBA, GL_BGRA, GL_LUMINANCE, GL_LUMINANCE_ALPHA
    dataType = None # GL_UNSIGNED_BYTE, GL_BYTE, GL_BITMAP, GL_UNSIGNED_SHORT, GL_SHORT, GL_UNSIGNED_INT, GL_INT, GL_FLOAT, ...
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
        for e in self.getSize():
            byteSize *= e + border
        return byteSize

    _pointer = None
    def getPointer(self):
        return self._pointer
    ptr = property(getPointer)

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
    def texString(self, strdata, pixelStoreSettings=None):
        self.texData(strdata, strdata, pixelStoreSettings)
    strdata = property(fset=texString)
    def texCData(self, cdata, pixelStoreSettings=None):
        self.texData(cdata, cdata, pixelStoreSettings)
    cdata = property(fset=texCData)
    def texArray(self, array, pixelStoreSettings=None):
        self.texData(array, array.buffer_info()[0], pixelStoreSettings)
    array = property(fset=texArray)

    def texBlank(self):
        # setup the alignment properly
        ps = self.newPixelStore(alignment=1)
        byteCount = self.getSizeInBytes()
        data = (ctypes.c_ubyte*byteCount)()
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
    pixelStore = property(getPixelStore, setPixelStore)
    def newPixelStore(self, *args, **kw):
        pixelStore = PixelStore(*args, **kw)
        self.setPixelStore(pixelStore)
        return pixelStore
    def updatePixelStore(self, settings):
        ps = self.getPixelStore()
        ps.set(settings)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getPosSize(self):
        return (self.pos, self.size)
    def setPosSize(self, pos, size=None):
        if size is None:
            pos, size = pos
        self.pos = pos
        self.size = size
    posSize = property(getPosSize, setPosSize)

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

class TextureImage1D(TextureImageBasic):
    x = 0
    width = 0

    def getPos(self):
        return (self.x,)
    def setPos(self, pos):
        (self.x,) = pos
    pos = property(getPos, setPos)
    
    def getSize(self):
        return (self.width,)
    def setSize(self, size):
        (self.width,) = size
    size = property(getSize, setSize)
    
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
    x = 0; y = 0
    width = 0; height = 0

    def getPos(self):
        return (self.x,self.y)
    def setPos(self, pos):
        (self.x,self.y) = pos
    pos = property(getPos, setPos)
    
    def getSize(self):
        return (self.width, self.height)
    def setSize(self, size):
        (self.width, self.height) = size
    size = property(getSize, setSize)
    
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
    x = 0; y = 0; z = 0
    width = 0; height = 0; depth = 0

    def getPos(self):
        return (self.x,self.y,self.z)
    def setPos(self, pos):
        (self.x,self.y,self.z) = pos
    pos = property(getPos, setPos)
    
    def getSize(self):
        return (self.width, self.height, self.depth)
    def setSize(self, size):
        (self.width, self.height, self.depth) = size
    size = property(getSize, setSize)
    
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
    target = 0 # GL_TEXTURE_1D, GL_TEXTURE_2D, etc.
    format = None # GL_RGBA, GL_INTENSITY, etc.

    texParams = dict()

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            self.create(*args, **kwargs)

    def __del__(self):
        try:
            self.release()
        except:
            import traceback
            traceback.print_exc()
            raise

    def create(self, target=None, **kwargs):
        if not self._as_parameter_ is None:
            raise Exception("Create has already been called for this instance")

        if target is not None:
            self.target = target

        self._genId()
        self.bind()
        self.set(self.texParams)
        self.set(kwargs)
        
    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    def release(self):
        if self._as_parameter_ is None:
            return

        self.unbind()
        self._delId()

    glGenTextures = staticmethod(gl.glGenTextures)
    def _genId(self):
        if self._as_parameter_ is None:
            p = gl.GLenum(0)
            self.glGenTextures(1, byref(p))
            self._as_parameter_ = p
    glDeleteTextures = staticmethod(gl.glDeleteTextures)
    def _delId(self):
        p = self._as_parameter_
        if p is not None:
            self.glDeleteTextures(1, byref(p))
            self._as_parameter_ = None

    glBindTexture = staticmethod(gl.glBindTexture)
    def bind(self):
        self.glBindTexture(self.target, self)
    def unbind(self):
        self.glBindTexture(self.target, 0)

    glEnable = staticmethod(gl.glEnable)
    def enable(self):
        self.glEnable(self.target)
    glDisable = staticmethod(gl.glDisable)
    def disable(self):
        self.glDisable(self.target)

    glActiveTexture = staticmethod(gl.glActiveTexture)
    def select(self, unit=None):
        if unit is not None:
            self.glActiveTexture(unit)
        self.bind()
        self.enable()
        return self

    def deselect(self):
        self.unbind()
        self.disable()
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    genMipmaps = glTexParamProperty(gl.GL_GENERATE_MIPMAP)

    wrapS = glTexParamProperty(gl.GL_TEXTURE_WRAP_S)
    wrapT = glTexParamProperty(gl.GL_TEXTURE_WRAP_T)
    wrapR = glTexParamProperty(gl.GL_TEXTURE_WRAP_R)

    def setWrap(self, wrap):
        self.wrapS = wrap
        self.wrapT = wrap
        self.wrapR = wrap
    wrap = property(fset=setWrap)

    magFilter = glTexParamProperty(gl.GL_TEXTURE_MAG_FILTER)
    minFilter = glTexParamProperty(gl.GL_TEXTURE_MIN_FILTER)
    
    def setFilter(self, filter):
        if isinstance(filter, tuple):
            self.magFilter = filter[0]
            self.minFilter = filter[1]
        else:
            self.magFilter = filter
            self.minFilter = filter
    filter = property(fset=setFilter)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Size & Dimensios
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    size = None # set when data is set into setImageND or setCompressedImageND

    @property
    def width(self): return self.size[0]
    @property
    def height(self): return self.size[1]
    @property
    def depth(self): return self.size[2]

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
    #~ Tex Image Setting
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setImage(self, data, level=0, **kw):
        data.setImageOn(self, level, **kw)
        return self
    def setImage1d(self, data, level=0):
        data.select()
        try:
            self.size = data.size
            gl.glTexImage1D(self.target, level, self.format, 
                data.width, data.border, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setImage2d(self, data, level=0):
        data.select()
        try:
            self.size = data.size
            gl.glTexImage2D(self.target, level, self.format, 
                data.width, data.height, data.border, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setImage3d(self, data, level=0):
        data.select()
        try:
            self.size = data.size
            gl.glTexImage3D(self.target, level, self.format, 
                data.width, data.height, data.depth, data.border, data.format, data.dataType, data.ptr)
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
            gl.glTexSubImage1D(self.target, level, data.x,
                data.width, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setSubImage2d(self, data, level=0):
        data.select()
        try:
            gl.glTexSubImage2D(self.target, level, data.x, data.y, 
                data.width, data.height, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setSubImage3d(self, data, level=0):
        data.select()
        try:
            gl.glTexSubImage3D(self.target, level, data.x, data.y, data.z, 
                data.width, data.height, data.depth, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setCompressedImage(self, data, level=0, **kw):
        data.setCompressedImageOn(self, level, **kw)
        return self
    def setCompressedImage1d(self, data, level=0):
        data.select()
        try:
            self.size = data.size
            gl.glCompressedTexImage1D(self.target, level, self.format, 
                    data.width, data.border, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setCompressedImage2d(self, data, level=0):
        data.select()
        try:
            self.size = data.size
            gl.glCompressedTexImage2D(self.target, level, self.format, 
                data.width, data.height, data.border, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setCompressedImage3d(self, data, level=0):
        data.select()
        try:
            self.size = data.size
            gl.glCompressedTexImage3D(self.target, level, self.format, 
                data.width, data.height, data.depth, data.border, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setCompressedSubImage(self, data, level=0):
        data.setCompressedSubImageOn(self, level, **kw)
        return self
    def setCompressedSubImage1d(self, data, level=0):
        data.select()
        try:
            gl.glCompressedTexSubImage1D(self.target, level, data.x,
                data.width, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setCompressedSubImage2d(self, data, level=0):
        data.select()
        try:
            gl.glCompressedTexSubImage2D(self.target, level, data.x, data.y, 
                data.width, data.height, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data
    def setCompressedSubImage3d(self, data, level=0):
        data.select()
        try:
            gl.glCompressedTexSubImage3D(self.target, level, data.x, data.y, data.z, 
                data.width, data.height, data.depth, data.format, data.dataType, data.ptr)
        finally:
            data.deselect()
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _maxTextureSizeByTarget = {}

    _textureTargetToMaxPName = {
        gl.GL_TEXTURE_1D: gl.GL_MAX_TEXTURE_SIZE,
        gl.GL_TEXTURE_2D: gl.GL_MAX_TEXTURE_SIZE,
        gl.GL_TEXTURE_3D: gl.GL_MAX_3D_TEXTURE_SIZE,
        gl.GL_TEXTURE_CUBE_MAP: gl.GL_MAX_CUBE_MAP_TEXTURE_SIZE,
        glext.GL_TEXTURE_RECTANGLE_ARB: glext.GL_MAX_RECTANGLE_TEXTURE_SIZE_ARB,
        }

    def getMaxTextureSize(self):
        return self.getMaxTextureSizeFor(self.target)

    @classmethod
    def getMaxTextureSizeFor(klass, target):
        r = klass._maxTextureSizeByTarget.get(target, None)
        if r is None:
            pname = klass._textureTargetToMaxPName[target]
            i = ctypes.c_int(0)
            gl.glGetIntegerv(pname, byref(i))
            r = i.value
            klass._maxTextureSizeByTarget[target] = r
        return r

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _textureTargetRequiresPowersOfTwo = {
        gl.GL_TEXTURE_1D: True,
        gl.GL_TEXTURE_2D: True,
        gl.GL_TEXTURE_3D: True,
        gl.GL_TEXTURE_CUBE_MAP: True,

        glext.GL_TEXTURE_RECTANGLE_ARB: False,
        }
    _powersOfTwo = [1<<s for s in xrange(31)]

    @staticmethod
    def idxNextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return bisect_left(powersOfTwo, v)
    @staticmethod
    def nextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return powersOfTwo[bisect_left(powersOfTwo, v)]

    def validSizeForTarget(self, size):
        if self._textureTargetRequiresPowersOfTwo.get(self.target, False):
            return tuple(self.nextPowerOf2(s) for s in size)
        else: return size

