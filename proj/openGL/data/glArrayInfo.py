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

from ..raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gldrawModeMap = {
    None: gl.GL_POINTS,
    'pts': gl.GL_POINTS,
    'points': gl.GL_POINTS,

    'lines': gl.GL_LINES,
    'lineLoop': gl.GL_LINE_LOOP,
    'lineStrip': gl.GL_LINE_STRIP,

    'tris': gl.GL_TRIANGLES,
    'triangles': gl.GL_TRIANGLES,
    'triStrip': gl.GL_TRIANGLE_STRIP,
    'triangleStrip': gl.GL_TRIANGLE_STRIP,
    'triFan': gl.GL_TRIANGLE_FAN,
    'triangleFan': gl.GL_TRIANGLE_FAN,

    'quads': gl.GL_QUADS,
    'quadStrip': gl.GL_QUAD_STRIP,
    }

glElementTypeIdMap = {
    'B': gl.GL_UNSIGNED_BYTE,
    'uint8': gl.GL_UNSIGNED_BYTE,
    'H': gl.GL_UNSIGNED_SHORT,
    'uint16': gl.GL_UNSIGNED_SHORT,
    'I': gl.GL_UNSIGNED_INT,
    'L': gl.GL_UNSIGNED_INT,
    'uint32': gl.GL_UNSIGNED_INT,
    }
glElementRangeTypeIdMap = {
    'I': gl.GL_UNSIGNED_INT,
    'L': gl.GL_UNSIGNED_INT,
    'uint32': gl.GL_UNSIGNED_INT,
    }

glDataTypeIdMap = {
    'uint8': gl.GL_UNSIGNED_BYTE,
    'B': gl.GL_UNSIGNED_BYTE,
    'int8': gl.GL_BYTE,
    'b': gl.GL_BYTE,
    'uint16': gl.GL_UNSIGNED_SHORT,
    'H': gl.GL_UNSIGNED_SHORT,
    'int16': gl.GL_SHORT,
    'h': gl.GL_SHORT,
    'uint32': gl.GL_UNSIGNED_INT,
    'I': gl.GL_UNSIGNED_INT,
    'L': gl.GL_UNSIGNED_INT,
    'int32': gl.GL_INT,
    'i': gl.GL_INT,
    'l': gl.GL_INT,
    'float32': gl.GL_FLOAT,
    'f': gl.GL_FLOAT,
    'float64': gl.GL_DOUBLE,
    'd': gl.GL_DOUBLE,
    }

glInterleavedTypeIdMap = {
    'v:2f': gl.GL_V2F,
    'v:3f': gl.GL_V3F,
    'c:4B;v:2f': gl.GL_C4UB_V2F,
    'c:4B;v:3f': gl.GL_C4UB_V3F,
    'c:3f;v:3f': gl.GL_C3F_V3F,
    'n:3f;v:3f': gl.GL_N3F_V3F,
    'c:4f;n:3f;v:3f': gl.GL_C4F_N3F_V3F,
    't:2f;v:3f': gl.GL_T2F_V3F,
    't:4f;v:4f': gl.GL_T4F_V4F,
    't:2f;c:4B;v:3f': gl.GL_T2F_C4UB_V3F,
    't:2f;c:3f;v:3f': gl.GL_T2F_C3F_V3F,
    't:2f;n:3f;v:3f': gl.GL_T2F_N3F_V3F,
    't:2f;c:4f;n:3f;v:3f': gl.GL_T2F_C4F_N3F_V3F,
    't:4f;c:4f;n:3f;v:4f': gl.GL_T4F_C4F_N3F_V4F,
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ GL Array Info Descriptor classes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLBaseArrayInfo(object):
    glBufferTypeId = None
    glTypeIdMap = None
    glArrayKindInfo = None
    kind = None
    glKindId = None

    glEnableArray = property(lambda self: gl.glEnableClientState)
    glDisableArray = property(lambda self: gl.glDisableClientState)

    def __init__(self, kind):
        self.glArrayKindInfo[kind] = self
        self.kind = kind

    @classmethod
    def arrayInfoFor(klass, kind):
        return klass.glArrayKindInfo[kind]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLDataArrayInfo(GLBaseArrayInfo):
    glBufferTypeId = gl.GL_ARRAY_BUFFER
    glTypeIdMap = glDataTypeIdMap
    glArrayKindInfo = {}

    _glArrayPointerName = None
    _glImmediateMap = None

    glArrayPointer = property(lambda self: getattr(gl, self._glArrayPointerName))

    def __init__(self, kind, glKindId, glArrayPointerName, glImmediateMap):
        GLBaseArrayInfo.__init__(self, kind)
        self.glKindId = glKindId
        self._glArrayPointerName = glArrayPointerName
        self._glImmediateMap = glImmediateMap

    def glImmediateFor(self, array):
        fnByShape = self._glImmediateMap.get(array.glTypeId) or {}
        glImmediate = fnByShape.get(array.shape[-1])
        if glImmediate is not None:
            glImmediate = getattr(gl, glImmediate)
        return glImmediate

    def glArrayPointerFor(self, array):
        pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLInterleavedArrayInfo(GLBaseArrayInfo):
    glBufferTypeId = gl.GL_ARRAY_BUFFER
    glTypeIdMap = glInterleavedTypeIdMap
    glArrayKindInfo = {}

    _glArrayPointerName = None
    _glImmediateMap = None

    glArrayPointer = property(lambda self: getattr(gl, self._glArrayPointerName))

    def __init__(self, kind, glArrayPointerName, glImmediateMap):
        GLBaseArrayInfo.__init__(self, kind)
        self._glArrayPointerName = glArrayPointerName
        self._glImmediateMap = glImmediateMap

    def glImmediateFor(self, array):
        raise NotImplementedError('Not supported')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLElementArrayInfo(GLBaseArrayInfo):
    glBufferTypeId = gl.GL_ELEMENT_ARRAY_BUFFER
    glTypeIdMap = glElementTypeIdMap
    glArrayKindInfo = {}
    glDrawUsageMap = None

    def __init__(self, kind, glDrawUsageMap):
        GLBaseArrayInfo.__init__(self, kind)
        self.glDrawUsageMap = glDrawUsageMap
    def glImmediateFor(self, array):
        glImmediate = self.glDrawUsageMap['drawSingle']
        if glImmediate is not None:
            glImmediate = getattr(gl, glImmediate)
        return glImmediate

class GLElementRangeInfo(GLElementArrayInfo):
    glBufferTypeId = None
    glTypeIdMap = glElementRangeTypeIdMap
    glArrayKindInfo = {}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLDataArrayInfo('vector', gl.GL_VERTEX_ARRAY, 'glVertexPointer', {
        gl.GL_UNSIGNED_BYTE: {1: None, 2: None, 3: None, 4: None},
        gl.GL_BYTE: {1: None, 2: None, 3: None, 4: None},
        gl.GL_UNSIGNED_SHORT: {1: None, 2: 'glVertex2sv', 3: 'glVertex3sv', 4: 'glVertex4sv',},
        gl.GL_SHORT: {1: None, 2: 'glVertex2sv', 3: 'glVertex3sv', 4: 'glVertex4sv',},
        gl.GL_UNSIGNED_INT: {1: None, 2: 'glVertex2iv', 3: 'glVertex3iv', 4: 'glVertex4iv',},
        gl.GL_INT: {1: None, 2: 'glVertex2iv', 3: 'glVertex3iv', 4: 'glVertex4iv',},
        gl.GL_FLOAT: {1: None, 2: 'glVertex2fv', 3: 'glVertex3fv', 4: 'glVertex4fv',},
        gl.GL_DOUBLE: {1: None, 2: 'glVertex2dv', 3: 'glVertex3dv', 4: 'glVertex4dv',},
        })

GLDataArrayInfo('vertex', gl.GL_VERTEX_ARRAY, 'glVertexPointer', {
        gl.GL_SHORT: {2: 'glVertex2sv', 3: 'glVertex3sv', 4: 'glVertex4sv',},
        gl.GL_INT: {2: 'glVertex2iv', 3: 'glVertex3iv', 4: 'glVertex4iv',},
        gl.GL_FLOAT: {2: 'glVertex2fv', 3: 'glVertex3fv', 4: 'glVertex4fv',},
        gl.GL_DOUBLE: {2: 'glVertex2dv', 3: 'glVertex3dv', 4: 'glVertex4dv',},
        })

GLDataArrayInfo('texture_coord', gl.GL_TEXTURE_COORD_ARRAY, 'glTexCoordPointer', {
        gl.GL_SHORT: {1: 'glTexCoord1sv', 2: 'glTexCoord2sv', 3: 'glTexCoord3sv', 4: 'glTexCoord4sv'},
        gl.GL_INT: {1: 'glTexCoord1iv', 2: 'glTexCoord2iv', 3: 'glTexCoord3iv', 4: 'glTexCoord4iv'},
        gl.GL_FLOAT: {1: 'glTexCoord1fv', 2: 'glTexCoord2fv', 3: 'glTexCoord3fv', 4: 'glTexCoord4fv'},
        gl.GL_DOUBLE: {1: 'glTexCoord1dv', 2: 'glTexCoord2dv', 3: 'glTexCoord3dv', 4: 'glTexCoord4dv'},
        })

GLDataArrayInfo('multi_texture_coord', gl.GL_TEXTURE_COORD_ARRAY, 'glTexCoordPointer', {
        gl.GL_SHORT: {1: 'glMultiTexCoord1sv', 2: 'glMultiTexCoord2sv', 3: 'glMultiTexCoord3sv', 4: 'glMultiTexCoord4sv'},
        gl.GL_INT: {1: 'glMultiTexCoord1iv', 2: 'glMultiTexCoord2iv', 3: 'glMultiTexCoord3iv', 4: 'glMultiTexCoord4iv'},
        gl.GL_FLOAT: {1: 'glMultiTexCoord1fv', 2: 'glMultiTexCoord2fv', 3: 'glMultiTexCoord3fv', 4: 'glMultiTexCoord4fv'},
        gl.GL_DOUBLE: {1: 'glMultiTexCoord1dv', 2: 'glMultiTexCoord2dv', 3: 'glMultiTexCoord3dv', 4: 'glMultiTexCoord4dv'},
        })

GLDataArrayInfo('normal', gl.GL_NORMAL_ARRAY, 'glNormalPointer', {
        gl.GL_BYTE: {3: 'glNormal3bv'},
        gl.GL_SHORT: {3: 'glNormal3sv'},
        gl.GL_INT: {3: 'glNormal3iv'},
        gl.GL_FLOAT: {3: 'glNormal3fv'},
        gl.GL_DOUBLE: {3: 'glNormal3dv'},
        })

GLDataArrayInfo('color', gl.GL_COLOR_ARRAY, 'glColorPointer', {
        gl.GL_UNSIGNED_BYTE: {3: 'glColor3ubv', 4: 'glColor4ubv'},
        gl.GL_UNSIGNED_SHORT: {3: 'glColor3usv', 4: 'glColor4usv'},
        gl.GL_UNSIGNED_INT: {3: 'glColor3uiv', 4: 'glColor4uiv'},
        gl.GL_BYTE: {3: 'glColor3bv', 4: 'glColor4bv'},
        gl.GL_SHORT: {3: 'glColor3sv', 4: 'glColor4sv'},
        gl.GL_INT: {3: 'glColor3iv', 4: 'glColor4iv'},
        gl.GL_FLOAT: {3: 'glColor3fv', 4: 'glColor4fv'},
        gl.GL_DOUBLE: {3: 'glColor3dv', 4: 'glColor4dv'},
        })

GLDataArrayInfo('secondary_color', gl.GL_SECONDARY_COLOR_ARRAY, 'glSecondaryColorPointer', {
        gl.GL_UNSIGNED_BYTE: {3: 'glSecondaryColor3ubv'},
        gl.GL_UNSIGNED_SHORT: {3: 'glSecondaryColor3usv'},
        gl.GL_UNSIGNED_INT: {3: 'glSecondaryColor3usv'},
        gl.GL_BYTE: {3: 'glSecondaryColor3usv'},
        gl.GL_SHORT: {3: 'glSecondaryColor3usv'},
        gl.GL_INT: {3: 'glSecondaryColor3usv'},
        gl.GL_FLOAT: {3: 'glSecondaryColor3usv'},
        gl.GL_DOUBLE: {3: 'glSecondaryColor3usv'},
        })

GLDataArrayInfo('color_index', gl.GL_INDEX_ARRAY, 'glIndexPointer', {
        gl.GL_UNSIGNED_BYTE: {1: 'glIndexubv'},
        gl.GL_SHORT: {1: 'glIndexsv'},
        gl.GL_INT: {1: 'glIndexiv'},
        gl.GL_FLOAT: {1: 'glIndexfv'},
        gl.GL_DOUBLE: {1: 'glIndexdv'},
        })

GLDataArrayInfo('fog_coord', gl.GL_FOG_COORD_ARRAY, 'glFogCoordPointer', {
        gl.GL_FLOAT: {1: 'glFogCoordfv'},
        gl.GL_DOUBLE: {1: 'glFogCoorddv'},
        })

GLDataArrayInfo('edge_flag', gl.GL_EDGE_FLAG_ARRAY, 'glEdgeFlagPointer', {
        gl.GL_UNSIGNED_BYTE: {1: 'glEdgeFlagv'},
        })

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLInterleavedArrayInfo('interleaved', 'glInterleavedArrays', {
        gl.GL_V2F: {'v': 'glVertex2fv', 't': None, 'n': None, 'c': None},
        gl.GL_V3F: {'v': 'glVertex3fv', 't': None, 'n': None, 'c': None},

        gl.GL_C4UB_V2F: {'v': 'glVertex2fv', 't': None, 'n': None, 'c': 'glColor4ubv'},
        gl.GL_C4UB_V3F: {'v': 'glVertex3fv', 't': None, 'n': None, 'c': 'glColor4ubv'},
        gl.GL_C3F_V3F: {'v': 'glVertex3fv', 't': None, 'n': None, 'c': 'glColor3fv'},

        gl.GL_N3F_V3F: {'v': 'glVertex3fv', 't': None, 'n': 'glNormal3fv', 'c': None},
        gl.GL_C4F_N3F_V3F: {'v': 'glVertex3fv', 't': None, 'n': 'glNormal3fv', 'c': 'glColor4fv'},

        gl.GL_T2F_V3F: {'v': 'glVertex3fv', 't': 'glTexCoord2fv', 'n': None, 'c': None},
        gl.GL_T4F_V4F: {'v': 'glVertex4fv', 't': 'glTexCoord4fv', 'n': None, 'c': None},

        gl.GL_T2F_C4UB_V3F: {'v': 'glVertex3fv', 't': 'glTexCoord2fv', 'n': None, 'c': 'glColor4ubv'},
        gl.GL_T2F_C3F_V3F: {'v': 'glVertex3fv', 't': 'glTexCoord2fv', 'n': None, 'c': 'glColor3fv'},

        gl.GL_T2F_N3F_V3F: {'v': 'glVertex3fv', 't': 'glTexCoord2fv', 'n': 'glNormal3fv', 'c': None},

        gl.GL_T2F_C4F_N3F_V3F: {'v': 'glVertex3fv', 't': 'glTexCoord2fv', 'n': 'glNormal3fv', 'c': 'glColor4fv'},
        gl.GL_T4F_C4F_N3F_V4F: {'v': 'glVertex4fv', 't': 'glTexCoord4fv', 'n': 'glNormal3fv', 'c': 'glColor4fv'},
        })

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLElementArrayInfo('element_array', dict(
        drawSingle='glArrayElement',
        draw='glDrawElements', 
        drawRange='glDrawRangeElements',
        drawMany='glMultiDrawElements', 
        ))

GLElementRangeInfo('element_range', dict(
        drawSingle='glDrawArrays',
        draw='glMultiDrawArrays',
        ))

