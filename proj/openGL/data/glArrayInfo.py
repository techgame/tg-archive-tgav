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

glTypeIdMap = {
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

glArrayKindInfo = {
    'interleaved': (None, {}),

    'vertex': (gl.GL_VERTEX_ARRAY, {
        gl.GL_SHORT: {2: gl.glVertex2sv, 3: gl.glVertex3sv, 4: gl.glVertex4sv,},
        gl.GL_INT: {2: gl.glVertex2iv, 3: gl.glVertex3iv, 4: gl.glVertex4iv,},
        gl.GL_FLOAT: {2: gl.glVertex2fv, 3: gl.glVertex3fv, 4: gl.glVertex4fv,},
        gl.GL_DOUBLE: {2: gl.glVertex2dv, 3: gl.glVertex3dv, 4: gl.glVertex4dv,},
        }),

    'texture_coord': (gl.GL_TEXTURE_COORD_ARRAY, {
        gl.GL_SHORT: {1: gl.glTexCoord1sv, 2: gl.glTexCoord2sv, 3: gl.glTexCoord3sv, 4: gl.glTexCoord4sv},
        gl.GL_INT: {1: gl.glTexCoord1iv, 2: gl.glTexCoord2iv, 3: gl.glTexCoord3iv, 4: gl.glTexCoord4iv},
        gl.GL_FLOAT: {1: gl.glTexCoord1fv, 2: gl.glTexCoord2fv, 3: gl.glTexCoord3fv, 4: gl.glTexCoord4fv},
        gl.GL_DOUBLE: {1: gl.glTexCoord1dv, 2: gl.glTexCoord2dv, 3: gl.glTexCoord3dv, 4: gl.glTexCoord4dv},
        }),

    'multi_texture_coord': (gl.GL_TEXTURE_COORD_ARRAY, {
        gl.GL_SHORT: {1: gl.glMultiTexCoord1sv, 2: gl.glMultiTexCoord2sv, 3: gl.glMultiTexCoord3sv, 4: gl.glMultiTexCoord4sv},
        gl.GL_INT: {1: gl.glMultiTexCoord1iv, 2: gl.glMultiTexCoord2iv, 3: gl.glMultiTexCoord3iv, 4: gl.glMultiTexCoord4iv},
        gl.GL_FLOAT: {1: gl.glMultiTexCoord1fv, 2: gl.glMultiTexCoord2fv, 3: gl.glMultiTexCoord3fv, 4: gl.glMultiTexCoord4fv},
        gl.GL_DOUBLE: {1: gl.glMultiTexCoord1dv, 2: gl.glMultiTexCoord2dv, 3: gl.glMultiTexCoord3dv, 4: gl.glMultiTexCoord4dv},
        }),

    'normal': (gl.GL_NORMAL_ARRAY, {
        gl.GL_BYTE: {3: gl.glNormal3bv},
        gl.GL_SHORT: {3: gl.glNormal3sv},
        gl.GL_INT: {3: gl.glNormal3iv},
        gl.GL_FLOAT: {3: gl.glNormal3fv},
        gl.GL_DOUBLE: {3: gl.glNormal3dv},
        }),

    'color': (gl.GL_COLOR_ARRAY, {
        gl.GL_UNSIGNED_BYTE: {3: gl.glColor3ubv, 4: gl.glColor4ubv},
        gl.GL_UNSIGNED_SHORT: {3: gl.glColor3usv, 4: gl.glColor4usv},
        gl.GL_UNSIGNED_INT: {3: gl.glColor3uiv, 4: gl.glColor4uiv},
        gl.GL_BYTE: {3: gl.glColor3bv, 4: gl.glColor4bv},
        gl.GL_SHORT: {3: gl.glColor3sv, 4: gl.glColor4sv},
        gl.GL_INT: {3: gl.glColor3iv, 4: gl.glColor4iv},
        gl.GL_FLOAT: {3: gl.glColor3fv, 4: gl.glColor4fv},
        gl.GL_DOUBLE: {3: gl.glColor3dv, 4: gl.glColor4dv},
        }),

    'secondary_color': (gl.GL_SECONDARY_COLOR_ARRAY, {
        gl.GL_UNSIGNED_BYTE: {3: gl.glSecondaryColor3ubv},
        gl.GL_UNSIGNED_SHORT: {3: gl.glSecondaryColor3usv},
        gl.GL_UNSIGNED_INT: {3: gl.glSecondaryColor3usv},
        gl.GL_BYTE: {3: gl.glSecondaryColor3usv},
        gl.GL_SHORT: {3: gl.glSecondaryColor3usv},
        gl.GL_INT: {3: gl.glSecondaryColor3usv},
        gl.GL_FLOAT: {3: gl.glSecondaryColor3usv},
        gl.GL_DOUBLE: {3: gl.glSecondaryColor3usv},
        }),

    'color_index': (gl.GL_INDEX_ARRAY, {
        gl.GL_UNSIGNED_BYTE: {1: gl.glIndexubv},
        gl.GL_SHORT: {1: gl.glIndexsv},
        gl.GL_INT: {1: gl.glIndexiv},
        gl.GL_FLOAT: {1: gl.glIndexfv},
        gl.GL_DOUBLE: {1: gl.glIndexdv},
        }),

    'fog_coord': (gl.GL_FOG_COORD_ARRAY, {
        gl.GL_FLOAT: {1: gl.glFogCoordfv},
        gl.GL_DOUBLE: {1: gl.glFogCoorddv},
        }),

    'edge_flag': (gl.GL_EDGE_FLAG_ARRAY, {
        gl.GL_UNSIGNED_BYTE: {1: gl.glEdgeFlagv},
        }),
    }

def glKindIdFrom(kind):
    return glArrayKindInfo[kind][0]
def glImmediateMapFrom(kind):
    return glArrayKindInfo[kind][-1]
