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

from numpy import dtype

from TG.openGL.data.vertexArrays import ArrayBase, dtypefmt

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Interleaved Arrays
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class InterleavedArrays(ArrayBase):
    dataFormatMap = {
        'v2f': gl.GL_V2F,
        'v3f': gl.GL_V3F,
        'c4ub_v2f': gl.GL_C4UB_V2F,
        'c4ub_v3f': gl.GL_C4UB_V3F,
        'c3f_v3f': gl.GL_C3F_V3F,
        'n3f_v3f': gl.GL_N3F_V3F,
        'c4f_n3f_v3f': gl.GL_C4F_N3F_V3F,
        't2f_v3f': gl.GL_T2F_V3F,
        't4f_v4f': gl.GL_T4F_V4F,
        't2f_c4ub_v3f': gl.GL_T2F_C4UB_V3F,
        't2f_c3f_v3f': gl.GL_T2F_C3F_V3F,
        't2f_n3f_v3f': gl.GL_T2F_N3F_V3F,
        't2f_c4f_n3f_v3f': gl.GL_T2F_C4F_N3F_V3F,
        't4f_c4f_n3f_v4f': gl.GL_T4F_C4F_N3F_V4F,
        }
    dataFormatDTypeMapping = [
        (None, dtypefmt('v:2f')),
        (gl.GL_V2F, dtypefmt('v:2f')),
        (gl.GL_V3F, dtypefmt('v:3f')),
        (gl.GL_C4UB_V2F, dtypefmt('c:4B, v:2f')),
        (gl.GL_C4UB_V3F, dtypefmt('c:4B, v:3f')),
        (gl.GL_C3F_V3F, dtypefmt('c:3f, v:3f')),
        (gl.GL_N3F_V3F, dtypefmt('n:3f, v:3f')),
        (gl.GL_C4F_N3F_V3F, dtypefmt('c:4f, n:3f, v:3f')),
        (gl.GL_T2F_V3F, dtypefmt('t:2f, v:3f')),
        (gl.GL_T4F_V4F, dtypefmt('t:4f, v:4f')),
        (gl.GL_T2F_C4UB_V3F, dtypefmt('t:2f, c:4B, v:3f')),
        (gl.GL_T2F_C3F_V3F, dtypefmt('t:2f, c:3f, v:3f')),
        (gl.GL_T2F_N3F_V3F, dtypefmt('t:2f, n:3f, v:3f')),
        (gl.GL_T2F_C4F_N3F_V3F, dtypefmt('t:2f, c:4f, n:3f, v:3f')),
        (gl.GL_T4F_C4F_N3F_V4F, dtypefmt('t:4f, c:4f, n:3f, v:4f')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glInterleavedArrays = staticmethod(gl.glInterleavedArrays)

    def enable(self):
        pass

    def bind(self, vboOffset=None):
        if vboOffset is None:
            self.glInterleavedArrays(self.dataFormat, self.strides[-1], self.ctypes)
        else:
            self.glInterleavedArrays(self.dataFormat, self.strides[-1], vboOffset)

    def disable(self):
        pass

    def unbind(self):
        self.glInterleavedArrays(self.dataFormat, 0, None)

    def draw(self, drawMode=None, vboOffset=None):
        self.select(vboOffset)
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)
    def drawRaw(self, drawMode=None):
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)

