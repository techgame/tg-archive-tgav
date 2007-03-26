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

_dtype_gltype_map = {
    'B': (gl.GL_UNSIGNED_BYTE,'ub')
    'b': (gl.GL_BYTE,'b')
    'H': (gl.GL_UNSIGNED_SHORT,'us')
    'h': (gl.GL_SHORT,'s')
    'I': (gl.GL_UNSIGNED_INT,'ui')
    'L': (gl.GL_UNSIGNED_INT,'ui')
    'i': (gl.GL_INT,'i')
    'l': (gl.GL_INT,'i')
    'f': (gl.GL_FLOAT,'f')
    'd': (gl.GL_DOUBLE,'d')
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ GL Array Info Descriptor classes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

glArrayFormatInfo = {
    'vertex': (gl.GL_VERTEX_ARRAY, 'glVertex', True, 'hlifd', [2,3,4]),
    'texture_coord': (gl.GL_TEXTURE_COORD_ARRAY, 'glTexCoord', True, 'hlifd', [1,2,3,4]),
    'multi_texture_coord': (gl.GL_TEXTURE_COORD_ARRAY, 'glTexCoord', 'glMultiTexCoord', 'hlifd', [1,2,3,4]),
    'normal': (gl.GL_NORMAL_ARRAY, 'glNormal', True, 'bhlifd', [3]),
    'color': (gl.GL_COLOR_ARRAY, 'glColor', True, 'BHLIbhlifd', [1,3,4]),
    'secondary_color': (gl.GL_SECONDARY_COLOR_ARRAY, 'glSecondaryColor', True, 'BHLIbhlifd', [1,3]), 
    'color_index': (gl.GL_INDEX_ARRAY, 'glIndex', True, 'Bhlifd', 1),
    'fog_coord': (gl.GL_FOG_COORD_ARRAY, 'glFogCoord', True, 'fd', 1),
    'edge_flag': (gl.GL_EDGE_FLAG_ARRAY, 'glEdgeFlag', True, 'B', 1),
    }

class GLArrayView(object):
    glArrayFormatInfo = glArrayFormatInfo
    glid_buffer = gl.GL_ARRAY_BUFFER
    glid_kind = None
    glid_type = None
    glfn_single = None
    glfn_pointer = None

    _kind = None
    def getKind(self):
        return self._kind
    def setKind(self, kind):
        self._kind = kind
    kind = property(getKind, setKind)

    def config(klassOrSelf, kind):
        afinfo = klassOrSelf.glArrayFormatInfo[kind]
        klassOrSelf.glid_kind = afinfo[0]

        if isinstance(afinfo[4], (int, long)):
            # dimension is not used in format
            fmt = '%(fmt)s'
        else: fmt = '%(dim)d%(fmt)s'
        klassOrSelf.glfn_single = afinfo[1] + fmt
        klassOrSelf.glfn_pointer = afinfo[1] + 'Pointer'


    def enable(self, gl): 
        gl.glEnableClientState(self.glid_kind)
    def disable(self, gl): 
        gl.glDisableClientState(self.glid_kind)
    def bind(self, gl, arr):
        glid_type, gl_fmt = _dtype_gltype_map[arr.dtype.char]
        glc_dim = arr.shape[-1]
        glc_stride = arr.strides[-2]

        self.glc_dim = glc_dim
        self.glc_stride = glc_stride
        self.glid_type = glid_type

        glfn_single = self.glfn_single % dict(dim=glc_dim, fmt=glc_fmt)
        self._glsingle = getattr(gl, glfn_single)

        self._glpointer_raw = getattr(gl, self.glfn_pointer)
        self._glpointer = partial(self._glpointer_raw, glid_type, glc_stride)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ GL Element Array Info Descriptor
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

glDrawMode = {
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLElementArrayView(object):
    glid_buffer = gl.GL_ELEMENT_ARRAY_BUFFER
    glid_kind = None

    glfn_pointer = None
    glfn_single = None

    def bind(self, gl, arr):
        pass

    def single(self, gl): gl.glArrayViewElement
    def draw(self, gl): gl.glDrawElements
    def drawRange(self, gl): gl.glDrawRangeElements
    def drawMany(self, gl): gl.glMultiDrawElements
    def drawArray(self, gl): gl.glDrawArrays
    def drawMultiArray(self, gl): gl.glMultiDrawArrays

