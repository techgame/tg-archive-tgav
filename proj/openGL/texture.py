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

import ctypes
from ctypes import cast, byref, c_void_p
from raw import gl, glu, glext

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLObject(object):
    def set(self, **kwattr):
        for n,v in kwattr.iteritems():
            setattr(self, n, v)

class glTexParamProperty(object):
    _as_parameter_ = None
        
    def __init__(self, propertyEnum):
        self._as_parameter_ = propertyEnum

    def __get__(self, obj, klass):
        if obj is None: 
            return self

        cValue = GLint(0)
        gl.glGetTexParameteriv(obj.target, self, byref(cValue))
        return cValue.value

    def __set__(self, obj, value):
        gl.glTexParameteri(obj.target, self, value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PixelStore(GLObject):
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
        return GLObject.__setattr__(self, name, value)

    def __getitem__(self, name):
        pname = self.fmtPNames[name][self.pack]
        return self.formatAttrs[pname]
    def __setitem__(self, name, value):
        pname = self.fmtPNames[name][self.pack]
        restoreValue = getattr(self.__class__, name)
        self.formatAttrs[pname] = (value, restoreValue)
        GLObject.__setattr__(self, name, value)
    def __delitem__(self, name):
        pname = self.fmtPNames[name][self.pack]
        del self.formatAttrs[pname]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def create(self, pack=False, **kwattrs):
        self.formatAttrs = {}
        self.pack = pack
        self.set(**kwattrs)

    def set(self, **kwattrs):
        for name, value in kwattrs.iteritems():
            setattr(self, name, value)

    def select(self):
        formatAttrs = self.formatAttrs
        for pname, value in formatAttrs.iteritems():
            gl.glPixelStorei(pname, value[0])

        yield True

        for pname, value in formatAttrs.iteritems():
            gl.glPixelStorei(pname, value[1])
    __iter__ = select

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Texture Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextureImageBasic(GLObject):
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
        self.set(**kwattrs)

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
    def texData(self, rawData, pointer):
        self._rawData = rawData
        if not pointer:
            self._pointer = c_void_p(None)
        elif isinstance(pointer, (int, long)):
            self._pointer = c_void_p(pointer)
        else:
            self._pointer = cast(pointer, c_void_p)

    def texClear(self):
        self.texData(None, None)
    def texString(self, strdata):
        self.texData(strdata, strdata)
    strdata = property(fset=texString)
    def texCData(self, cdata):
        self.texData(cdata, cdata)
    cdata = property(fset=texCData)
    def texArray(self, array):
        self.texData(array, array.buffer_info()[0])
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
        self._pixelStore = pixelStore
    def newPixelStore(self, *args, **kw):
        pixelStore = PixelStore(*args, **kw)
        self.setPixelStore(pixelStore)
        return pixelStore
    pixelStore = property(getPixelStore, setPixelStore)

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
            for e in pixelStore.select():
                yield True
        else:
            yield True
    __iter__ = select

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
        texture.setSubImage1d(self, level, **kw)
        return texture
    def setCompressedImageOn(self, texture, level=0, **kw):
        texture.setCompressedImage1d(self, level, **kw)
        return texture
    def setCompressedSubImageOn(self, texture, level=0, **kw):
        texture.setCompressedSubImage1d(self, level, **kw)
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
        texture.setSubImage2d(self, level, **kw)
        return texture
    def setCompressedImageOn(self, texture, level=0, **kw):
        texture.setCompressedImage2d(self, level, **kw)
        return texture
    def setCompressedSubImageOn(self, texture, level=0, **kw):
        texture.setCompressedSubImage2d(self, level, **kw)
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
        texture.setSubImage3d(self, level, **kw)
        return texture
    def setCompressedImageOn(self, texture, level=0, **kw):
        texture.setCompressedImage3d(self, level, **kw)
        return texture
    def setCompressedSubImageOn(self, texture, level=0, **kw):
        texture.setCompressedSubImage3d(self, level, **kw)
        return texture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Texture object itself
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Texture(GLObject):
    _as_parameter_ = None # GLenum returned from glGenTextures
    target = 0 # GL_TEXTURE_1D, GL_TEXTURE_2D, etc.
    format = None # GL_RGBA, GL_INTENSITY, etc.

    def __init__(self, *args, **kwargs):
        if args:
            self.create(*args, **kwargs)

    def __del__(self):
        self.release()

    def create(self, target, format, **kwargs):
        if not self._as_parameter_ is None:
            raise Exception("Create has already been called for this instance")

        self._as_parameter_ = gl.GLenum(0)
        self.target = target
        self.format = format
        gl.glGenTextures(1, byref(self._as_parameter_))

        self.select().next()
        self.set(**kwargs)
        

    def release(self):
        if not self._as_parameter_:
            return

        gl.glDeleteTextures(1, byref(self._as_parameter_))

        self._as_parameter_ = None

    def select(self, unit=None):
        if unit is not None:
            gl.glActiveTexture(unit)
        gl.glEnable(self.target)
        gl.glBindTexture(self.target, self)

        yield True

        gl.glBindTexture(self.target, 0)
        gl.glDisable(self.target)
    __iter__ = select

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
        for _ in data.select():
            self.size = data.size
            gl.glTexImage1D(self.target, level, self.format, 
                data.width, data.border, data.format, data.dataType, data.ptr)
        return data
    def setImage2d(self, data, level=0):
        for _ in data.select():
            self.size = data.size
            gl.glTexImage2D(self.target, level, self.format, 
                data.width, data.height, data.border, data.format, data.dataType, data.ptr)
        return data
    def setImage3d(self, data, level=0):
        for _ in data.select():
            self.size = data.size
            gl.glTexImage3D(self.target, level, self.format, 
                data.width, data.height, data.depth, data.border, data.format, data.dataType, data.ptr)
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setSubImage(self, data, level=0, **kw):
        data.setSubImageOn(self, level, **kw)
        return self
    def setSubImage1d(self, data, level=0):
        for _ in data.select():
            gl.glTexSubImage1D(self.target, level, data.x,
                data.width, data.format, data.dataType, data.ptr)
        return data
    def setSubImage2d(self, data, level=0):
        for _ in data.select():
            gl.glTexSubImage2D(self.target, level, data.x, data.y, 
                data.width, data.height, data.format, data.dataType, data.ptr)
        return data
    def setSubImage3d(self, data, level=0):
        for _ in data.select():
            gl.glTexSubImage3D(self.target, level, data.x, data.y, data.z, 
                data.width, data.height, data.depth, data.format, data.dataType, data.ptr)
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setCompressedImage(self, data, level=0, **kw):
        data.setCompressedImageOn(self, level, **kw)
        return self
    def setCompressedImage1d(self, data, level=0):
        for _ in data.select():
            self.size = data.size
            gl.glCompressedTexImage1D(self.target, level, self.format, 
                    data.width, data.border, data.format, data.dataType, data.ptr)
        return data
    def setCompressedImage2d(self, data, level=0):
        for _ in data.select():
            self.size = data.size
            gl.glCompressedTexImage2D(self.target, level, self.format, 
                data.width, data.height, data.border, data.format, data.dataType, data.ptr)
        return data
    def setCompressedImage3d(self, data, level=0):
        for _ in data.select():
            self.size = data.size
            gl.glCompressedTexImage3D(self.target, level, self.format, 
                data.width, data.height, data.depth, data.border, data.format, data.dataType, data.ptr)
        return data

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setCompressedSubImage(self, data, level=0):
        data.setCompressedSubImageOn(self, level, **kw)
        return self
    def setCompressedSubImage1d(self, data, level=0):
        for _ in data.select():
            gl.glCompressedTexSubImage1D(self.target, level, data.x,
                data.width, data.format, data.dataType, data.ptr)
        return data
    def setCompressedSubImage2d(self, data, level=0):
        for _ in data.select():
            gl.glCompressedTexSubImage2D(self.target, level, data.x, data.y, 
                data.width, data.height, data.format, data.dataType, data.ptr)
        return data
    def setCompressedSubImage3d(self, data, level=0):
        for _ in data.select():
            gl.glCompressedTexSubImage3D(self.target, level, data.x, data.y, data.z, 
                data.width, data.height, data.depth, data.format, data.dataType, data.ptr)
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

    def maxTextureSize(self):
        r = self._maxTextureSizeByTarget.get(self.target, None)
        if r is None:
            pname = self._textureTargetToMaxPName[self.target]
            i = ctypes.c_int(0)
            gl.glGetIntegerv(pname, byref(i))
            r = i.value
            self._maxTextureSizeByTarget[self.target] = r
        return r

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _powersOfTwo = [1<<s for s in xrange(31)]

    @staticmethod
    def _idxNextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return bisect_left(powersOfTwo, v)
    @staticmethod
    def _nextPowerOf2(v, powersOfTwo=_powersOfTwo):
        return powersOfTwo[bisect_left(powersOfTwo, v)]

