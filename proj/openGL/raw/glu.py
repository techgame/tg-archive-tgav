#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_opengl import *
from gl import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "inc/OpenGL/glu.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GLU_EXT_object_space_tess = 1
GLU_EXT_nurbs_tessellator = 1

GLU_FALSE = 0
GLU_TRUE = 1

GLU_VERSION_1_1 = 1
GLU_VERSION_1_2 = 1
GLU_VERSION_1_3 = 1

GLU_VERSION = 100800
GLU_EXTENSIONS = 100801

GLU_INVALID_ENUM = 100900
GLU_INVALID_VALUE = 100901
GLU_OUT_OF_MEMORY = 100902
GLU_INCOMPATIBLE_GL_VERSION = 100903
GLU_INVALID_OPERATION = 100904

#~ line: 44, skipped: 5 ~~~~~~

GLU_OUTLINE_POLYGON = 100240
GLU_OUTLINE_PATCH = 100241

GLU_NURBS_ERROR = 100103
GLU_ERROR = 100103
GLU_NURBS_BEGIN = 100164
GLU_NURBS_BEGIN_EXT = 100164
GLU_NURBS_VERTEX = 100165
GLU_NURBS_VERTEX_EXT = 100165
GLU_NURBS_NORMAL = 100166
GLU_NURBS_NORMAL_EXT = 100166
GLU_NURBS_COLOR = 100167
GLU_NURBS_COLOR_EXT = 100167
GLU_NURBS_TEXTURE_COORD = 100168
GLU_NURBS_TEX_COORD_EXT = 100168
GLU_NURBS_END = 100169
GLU_NURBS_END_EXT = 100169
GLU_NURBS_BEGIN_DATA = 100170
GLU_NURBS_BEGIN_DATA_EXT = 100170
GLU_NURBS_VERTEX_DATA = 100171
GLU_NURBS_VERTEX_DATA_EXT = 100171
GLU_NURBS_NORMAL_DATA = 100172
GLU_NURBS_NORMAL_DATA_EXT = 100172
GLU_NURBS_COLOR_DATA = 100173
GLU_NURBS_COLOR_DATA_EXT = 100173
GLU_NURBS_TEXTURE_COORD_DATA = 100174
GLU_NURBS_TEX_COORD_DATA_EXT = 100174
GLU_NURBS_END_DATA = 100175
GLU_NURBS_END_DATA_EXT = 100175

GLU_NURBS_ERROR1 = 100251
GLU_NURBS_ERROR2 = 100252
GLU_NURBS_ERROR3 = 100253
GLU_NURBS_ERROR4 = 100254
GLU_NURBS_ERROR5 = 100255
GLU_NURBS_ERROR6 = 100256
GLU_NURBS_ERROR7 = 100257
GLU_NURBS_ERROR8 = 100258
GLU_NURBS_ERROR9 = 100259
GLU_NURBS_ERROR10 = 100260
GLU_NURBS_ERROR11 = 100261
GLU_NURBS_ERROR12 = 100262
GLU_NURBS_ERROR13 = 100263
GLU_NURBS_ERROR14 = 100264
GLU_NURBS_ERROR15 = 100265
GLU_NURBS_ERROR16 = 100266
GLU_NURBS_ERROR17 = 100267
GLU_NURBS_ERROR18 = 100268
GLU_NURBS_ERROR19 = 100269
GLU_NURBS_ERROR20 = 100270
GLU_NURBS_ERROR21 = 100271
GLU_NURBS_ERROR22 = 100272
GLU_NURBS_ERROR23 = 100273
GLU_NURBS_ERROR24 = 100274
GLU_NURBS_ERROR25 = 100275
GLU_NURBS_ERROR26 = 100276
GLU_NURBS_ERROR27 = 100277
GLU_NURBS_ERROR28 = 100278
GLU_NURBS_ERROR29 = 100279
GLU_NURBS_ERROR30 = 100280
GLU_NURBS_ERROR31 = 100281
GLU_NURBS_ERROR32 = 100282
GLU_NURBS_ERROR33 = 100283
GLU_NURBS_ERROR34 = 100284
GLU_NURBS_ERROR35 = 100285
GLU_NURBS_ERROR36 = 100286
GLU_NURBS_ERROR37 = 100287

GLU_AUTO_LOAD_MATRIX = 100200
GLU_CULLING = 100201
GLU_SAMPLING_TOLERANCE = 100203
GLU_DISPLAY_MODE = 100204
GLU_PARAMETRIC_TOLERANCE = 100202
GLU_SAMPLING_METHOD = 100205
GLU_U_STEP = 100206
GLU_V_STEP = 100207
GLU_NURBS_MODE = 100160
GLU_NURBS_MODE_EXT = 100160
GLU_NURBS_TESSELLATOR = 100161
GLU_NURBS_TESSELLATOR_EXT = 100161
GLU_NURBS_RENDERER = 100162
GLU_NURBS_RENDERER_EXT = 100162

GLU_OBJECT_PARAMETRIC_ERROR = 100208
GLU_OBJECT_PARAMETRIC_ERROR_EXT = 100208
GLU_OBJECT_PATH_LENGTH = 100209
GLU_OBJECT_PATH_LENGTH_EXT = 100209
GLU_PATH_LENGTH = 100215
GLU_PARAMETRIC_ERROR = 100216
GLU_DOMAIN_DISTANCE = 100217

GLU_MAP1_TRIM_2 = 100210
GLU_MAP1_TRIM_3 = 100211

GLU_POINT = 100010
GLU_LINE = 100011
GLU_FILL = 100012
GLU_SILHOUETTE = 100013

#~ line: 153, skipped: 6 ~~~~~~

GLU_SMOOTH = 100000
GLU_FLAT = 100001
GLU_NONE = 100002

GLU_OUTSIDE = 100020
GLU_INSIDE = 100021

GLU_TESS_BEGIN = 100100
GLU_BEGIN = 100100
GLU_TESS_VERTEX = 100101
GLU_VERTEX = 100101
GLU_TESS_END = 100102
GLU_END = 100102
GLU_TESS_ERROR = 100103
GLU_TESS_EDGE_FLAG = 100104
GLU_EDGE_FLAG = 100104
GLU_TESS_COMBINE = 100105
GLU_TESS_BEGIN_DATA = 100106
GLU_TESS_VERTEX_DATA = 100107
GLU_TESS_END_DATA = 100108
GLU_TESS_ERROR_DATA = 100109
GLU_TESS_EDGE_FLAG_DATA = 100110
GLU_TESS_COMBINE_DATA = 100111

GLU_CW = 100120
GLU_CCW = 100121
GLU_INTERIOR = 100122
GLU_EXTERIOR = 100123
GLU_UNKNOWN = 100124

GLU_TESS_WINDING_RULE = 100140
GLU_TESS_BOUNDARY_ONLY = 100141
GLU_TESS_TOLERANCE = 100142

GLU_TESS_ERROR1 = 100151
GLU_TESS_ERROR2 = 100152
GLU_TESS_ERROR3 = 100153
GLU_TESS_ERROR4 = 100154
GLU_TESS_ERROR5 = 100155
GLU_TESS_ERROR6 = 100156
GLU_TESS_ERROR7 = 100157
GLU_TESS_ERROR8 = 100158
GLU_TESS_MISSING_BEGIN_POLYGON = 100151
GLU_TESS_MISSING_BEGIN_CONTOUR = 100152
GLU_TESS_MISSING_END_POLYGON = 100153
GLU_TESS_MISSING_END_CONTOUR = 100154
GLU_TESS_COORD_TOO_LARGE = 100155
GLU_TESS_NEED_COMBINE_CALLBACK = 100156

GLU_TESS_WINDING_ODD = 100130
GLU_TESS_WINDING_NONZERO = 100131
GLU_TESS_WINDING_POSITIVE = 100132
GLU_TESS_WINDING_NEGATIVE = 100133
GLU_TESS_WINDING_ABS_GEQ_TWO = 100134

#~ line: 218, skipped: 6 ~~~~~~

class GLUnurbs(Structure):
    _fields_ = []
class GLUquadric(Structure):
    _fields_ = []
class GLUtesselator(Structure):
    _fields_ = []

#~ line: 237, skipped: 17 ~~~~~~

GLU_TESS_MAX_COORD = 1.0e150

@bind(None, [POINTER(GLUnurbs)])
def gluBeginCurve(nurb): pass

@bind(None, [POINTER(GLUtesselator)])
def gluBeginPolygon(tess): pass

@bind(None, [POINTER(GLUnurbs)])
def gluBeginSurface(nurb): pass

@bind(None, [POINTER(GLUnurbs)])
def gluBeginTrim(nurb): pass

@bind(GLint, [GLenum, GLint, GLsizei, GLenum, GLenum, GLint, GLint, GLint, POINTER(None)])
def gluBuild1DMipmapLevels(target, internalFormat, width, format, type, level, base, max, data): pass

@bind(GLint, [GLenum, GLint, GLsizei, GLenum, GLenum, POINTER(None)])
def gluBuild1DMipmaps(target, internalFormat, width, format, type, data): pass

@bind(GLint, [GLenum, GLint, GLsizei, GLsizei, GLenum, GLenum, GLint, GLint, GLint, POINTER(None)])
def gluBuild2DMipmapLevels(target, internalFormat, width, height, format, type, level, base, max, data): pass

@bind(GLint, [GLenum, GLint, GLsizei, GLsizei, GLenum, GLenum, POINTER(None)])
def gluBuild2DMipmaps(target, internalFormat, width, height, format, type, data): pass

@bind(GLint, [GLenum, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, GLint, GLint, GLint, POINTER(None)])
def gluBuild3DMipmapLevels(target, internalFormat, width, height, depth, format, type, level, base, max, data): pass

@bind(GLint, [GLenum, GLint, GLsizei, GLsizei, GLsizei, GLenum, GLenum, POINTER(None)])
def gluBuild3DMipmaps(target, internalFormat, width, height, depth, format, type, data): pass

@bind(GLboolean, [POINTER(GLubyte), POINTER(GLubyte)])
def gluCheckExtension(extName, extString): pass

@bind(None, [POINTER(GLUquadric), GLdouble, GLdouble, GLdouble, GLint, GLint])
def gluCylinder(quad, base, top, height, slices, stacks): pass

@bind(None, [POINTER(GLUnurbs)])
def gluDeleteNurbsRenderer(nurb): pass

@bind(None, [POINTER(GLUquadric)])
def gluDeleteQuadric(quad): pass

@bind(None, [POINTER(GLUtesselator)])
def gluDeleteTess(tess): pass

@bind(None, [POINTER(GLUquadric), GLdouble, GLdouble, GLint, GLint])
def gluDisk(quad, inner, outer, slices, loops): pass

@bind(None, [POINTER(GLUnurbs)])
def gluEndCurve(nurb): pass

@bind(None, [POINTER(GLUtesselator)])
def gluEndPolygon(tess): pass

@bind(None, [POINTER(GLUnurbs)])
def gluEndSurface(nurb): pass

@bind(None, [POINTER(GLUnurbs)])
def gluEndTrim(nurb): pass

@bind(POINTER(GLubyte), [GLenum])
def gluErrorString(error): pass

@bind(None, [POINTER(GLUnurbs), GLenum, POINTER(GLfloat)])
def gluGetNurbsProperty(nurb, property, data): pass

@bind(POINTER(GLubyte), [GLenum])
def gluGetString(name): pass

@bind(None, [POINTER(GLUtesselator), GLenum, POINTER(GLdouble)])
def gluGetTessProperty(tess, which, data): pass

@bind(None, [POINTER(GLUnurbs), POINTER(GLfloat), POINTER(GLfloat), POINTER(GLint)])
def gluLoadSamplingMatrices(nurb, model, perspective, view): pass

@bind(None, [GLdouble, GLdouble, GLdouble, GLdouble, GLdouble, GLdouble, GLdouble, GLdouble, GLdouble])
def gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ): pass

@bind(POINTER(GLUnurbs), [])
def gluNewNurbsRenderer(): pass

@bind(POINTER(GLUquadric), [])
def gluNewQuadric(): pass

@bind(POINTER(GLUtesselator), [])
def gluNewTess(): pass

@bind(None, [POINTER(GLUtesselator), GLenum])
def gluNextContour(tess, type): pass

@bind(None, [POINTER(GLUnurbs), GLenum, POINTER(CFUNCTYPE(GLvoid, ))])
def gluNurbsCallback(nurb, which, CallBackFunc): pass

@bind(None, [POINTER(GLUnurbs), POINTER(GLvoid)])
def gluNurbsCallbackData(nurb, userData): pass

@bind(None, [POINTER(GLUnurbs), POINTER(GLvoid)])
def gluNurbsCallbackDataEXT(nurb, userData): pass

@bind(None, [POINTER(GLUnurbs), GLint, POINTER(GLfloat), GLint, POINTER(GLfloat), GLint, GLenum])
def gluNurbsCurve(nurb, knotCount, knots, stride, control, order, type): pass

@bind(None, [POINTER(GLUnurbs), GLenum, GLfloat])
def gluNurbsProperty(nurb, property, value): pass

@bind(None, [POINTER(GLUnurbs), GLint, POINTER(GLfloat), GLint, POINTER(GLfloat), GLint, GLint, POINTER(GLfloat), GLint, GLint, GLenum])
def gluNurbsSurface(nurb, sKnotCount, sKnots, tKnotCount, tKnots, sStride, tStride, control, sOrder, tOrder, type): pass

@bind(None, [GLdouble, GLdouble, GLdouble, GLdouble])
def gluOrtho2D(left, right, bottom, top): pass

@bind(None, [POINTER(GLUquadric), GLdouble, GLdouble, GLint, GLint, GLdouble, GLdouble])
def gluPartialDisk(quad, inner, outer, slices, loops, start, sweep): pass

@bind(None, [GLdouble, GLdouble, GLdouble, GLdouble])
def gluPerspective(fovy, aspect, zNear, zFar): pass

@bind(None, [GLdouble, GLdouble, GLdouble, GLdouble, POINTER(GLint)])
def gluPickMatrix(x, y, delX, delY, viewport): pass

@bind(GLint, [GLdouble, GLdouble, GLdouble, POINTER(GLdouble), POINTER(GLdouble), POINTER(GLint), POINTER(GLdouble), POINTER(GLdouble), POINTER(GLdouble)])
def gluProject(objX, objY, objZ, model, proj, view, winX, winY, winZ): pass

@bind(None, [POINTER(GLUnurbs), GLint, POINTER(GLfloat), GLint, GLenum])
def gluPwlCurve(nurb, count, data, stride, type): pass

@bind(None, [POINTER(GLUquadric), GLenum, POINTER(CFUNCTYPE(GLvoid, ))])
def gluQuadricCallback(quad, which, CallBackFunc): pass

@bind(None, [POINTER(GLUquadric), GLenum])
def gluQuadricDrawStyle(quad, draw): pass

@bind(None, [POINTER(GLUquadric), GLenum])
def gluQuadricNormals(quad, normal): pass

@bind(None, [POINTER(GLUquadric), GLenum])
def gluQuadricOrientation(quad, orientation): pass

@bind(None, [POINTER(GLUquadric), GLboolean])
def gluQuadricTexture(quad, texture): pass

@bind(GLint, [GLenum, GLsizei, GLsizei, GLenum, POINTER(None), GLsizei, GLsizei, GLenum, POINTER(GLvoid)])
def gluScaleImage(format, wIn, hIn, typeIn, dataIn, wOut, hOut, typeOut, dataOut): pass

@bind(None, [POINTER(GLUquadric), GLdouble, GLint, GLint])
def gluSphere(quad, radius, slices, stacks): pass

@bind(None, [POINTER(GLUtesselator)])
def gluTessBeginContour(tess): pass

@bind(None, [POINTER(GLUtesselator), POINTER(GLvoid)])
def gluTessBeginPolygon(tess, data): pass

@bind(None, [POINTER(GLUtesselator), GLenum, POINTER(CFUNCTYPE(GLvoid, ))])
def gluTessCallback(tess, which, CallBackFunc): pass

@bind(None, [POINTER(GLUtesselator)])
def gluTessEndContour(tess): pass

@bind(None, [POINTER(GLUtesselator)])
def gluTessEndPolygon(tess): pass

@bind(None, [POINTER(GLUtesselator), GLdouble, GLdouble, GLdouble])
def gluTessNormal(tess, valueX, valueY, valueZ): pass

@bind(None, [POINTER(GLUtesselator), GLenum, GLdouble])
def gluTessProperty(tess, which, data): pass

@bind(None, [POINTER(GLUtesselator), POINTER(GLdouble), POINTER(GLvoid)])
def gluTessVertex(tess, location, data): pass

@bind(GLint, [GLdouble, GLdouble, GLdouble, POINTER(GLdouble), POINTER(GLdouble), POINTER(GLint), POINTER(GLdouble), POINTER(GLdouble), POINTER(GLdouble)])
def gluUnProject(winX, winY, winZ, model, proj, view, objX, objY, objZ): pass

@bind(GLint, [GLdouble, GLdouble, GLdouble, GLdouble, POINTER(GLdouble), POINTER(GLdouble), POINTER(GLint), GLdouble, GLdouble, POINTER(GLdouble), POINTER(GLdouble), POINTER(GLdouble), POINTER(GLdouble)])
def gluUnProject4(winX, winY, winZ, clipW, model, proj, view, near, far, objX, objY, objZ, objW): pass



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "inc/OpenGL/glu.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

