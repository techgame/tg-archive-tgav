#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_freetype import *
from fttypes import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/usr/local/include/freetype2/freetype/ftimage.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# typedef FT_Pos
FT_Pos = c_long

class FT_Vector_(Structure):
    _fields_ = [
        ("x", FT_Pos),
        ("y", FT_Pos),
        ]

# typedef FT_Vector
FT_Vector = FT_Vector_

class FT_BBox_(Structure):
    _fields_ = [
        ("xMin", FT_Pos),
        ("yMin", FT_Pos),
        ("xMax", FT_Pos),
        ("yMax", FT_Pos),
        ]

# typedef FT_BBox
FT_BBox = FT_BBox_

class FT_Pixel_Mode_(c_int):
    '''enum FT_Pixel_Mode_''' 
    FT_PIXEL_MODE_NONE = 0
    FT_PIXEL_MODE_MONO = 1
    FT_PIXEL_MODE_GRAY = 2
    FT_PIXEL_MODE_GRAY2 = 3
    FT_PIXEL_MODE_GRAY4 = 4
    FT_PIXEL_MODE_LCD = 5
    FT_PIXEL_MODE_LCD_V = 6
    FT_PIXEL_MODE_MAX = 7

class FT_Bitmap_(Structure):
    _fields_ = [
        ("rows", c_int),
        ("width", c_int),
        ("pitch", c_int),
        ("buffer", POINTER(c_ubyte)),
        ("num_grays", c_short),
        ("pixel_mode", c_char),
        ("palette_mode", c_char),
        ("palette", FT_Pointer),
        ]

# typedef FT_Bitmap
FT_Bitmap = FT_Bitmap_

class FT_Outline_(Structure):
    _fields_ = [
        ("n_contours", c_short),
        ("n_points", c_short),
        ("points", POINTER(FT_Vector)),
        ("tags", c_char_p),
        ("contours", POINTER(c_short)),
        ("flags", c_int),
        ]

# typedef FT_Outline
FT_Outline = FT_Outline_

FT_OUTLINE_NONE = 0x0
FT_OUTLINE_OWNER = 0x1
FT_OUTLINE_EVEN_ODD_FILL = 0x2
FT_OUTLINE_REVERSE_FILL = 0x4
FT_OUTLINE_IGNORE_DROPOUTS = 0x8

FT_OUTLINE_HIGH_PRECISION = 0x100
FT_OUTLINE_SINGLE_PASS = 0x200

FT_CURVE_TAG_ON = 1
FT_CURVE_TAG_CONIC = 0
FT_CURVE_TAG_CUBIC = 2

FT_CURVE_TAG_TOUCH_X = 8
FT_CURVE_TAG_TOUCH_Y = 16

FT_CURVE_TAG_TOUCH_BOTH = ( FT_CURVE_TAG_TOUCH_X | FT_CURVE_TAG_TOUCH_Y )

FT_Curve_Tag_On = 1 # = FT_CURVE_TAG_ON
FT_Curve_Tag_Conic = 0 # = FT_CURVE_TAG_CONIC
FT_Curve_Tag_Cubic = 2 # = FT_CURVE_TAG_CUBIC
FT_Curve_Tag_Touch_X = 8 # = FT_CURVE_TAG_TOUCH_X
FT_Curve_Tag_Touch_Y = 16 # = FT_CURVE_TAG_TOUCH_Y

# typedef FT_Outline_MoveToFunc
FT_Outline_MoveToFunc = POINTER(CFUNCTYPE(c_int, POINTER(FT_Vector), FT_Pointer))

# typedef FT_Outline_LineToFunc
FT_Outline_LineToFunc = FT_Outline_MoveToFunc

# typedef FT_Outline_ConicToFunc
FT_Outline_ConicToFunc = POINTER(CFUNCTYPE(c_int, POINTER(FT_Vector), POINTER(FT_Vector), FT_Pointer))

# typedef FT_Outline_CubicToFunc
FT_Outline_CubicToFunc = POINTER(CFUNCTYPE(c_int, POINTER(FT_Vector), POINTER(FT_Vector), POINTER(FT_Vector), FT_Pointer))

class FT_Outline_Funcs_(Structure):
    _fields_ = [
        ("move_to", FT_Outline_MoveToFunc),
        ("line_to", FT_Outline_LineToFunc),
        ("conic_to", FT_Outline_ConicToFunc),
        ("cubic_to", FT_Outline_CubicToFunc),
        ("shift", c_int),
        ("delta", FT_Pos),
        ]

# typedef FT_Outline_Funcs
FT_Outline_Funcs = FT_Outline_Funcs_

class FT_Glyph_Format_(c_int):
    '''enum FT_Glyph_Format_''' 
    FT_GLYPH_FORMAT_NONE = 0
    FT_GLYPH_FORMAT_COMPOSITE = 1668246896
    FT_GLYPH_FORMAT_BITMAP = 1651078259
    FT_GLYPH_FORMAT_OUTLINE = 1869968492
    FT_GLYPH_FORMAT_PLOTTER = 1886154612

# typedef FT_Glyph_Format
FT_Glyph_Format = FT_Glyph_Format_

class FT_Span_(Structure):
    _fields_ = [
        ("x", c_short),
        ("len", c_ushort),
        ("coverage", c_ubyte),
        ]

# typedef FT_Span
FT_Span = FT_Span_

# typedef FT_SpanFunc
FT_SpanFunc = POINTER(CFUNCTYPE(None, c_int, c_int, POINTER(FT_Span), FT_Pointer))

# typedef FT_Raster_BitTest_Func
FT_Raster_BitTest_Func = POINTER(CFUNCTYPE(c_int, c_int, c_int, FT_Pointer))

# typedef FT_Raster_BitSet_Func
FT_Raster_BitSet_Func = POINTER(CFUNCTYPE(None, c_int, c_int, FT_Pointer))

FT_RASTER_FLAG_DEFAULT = 0x0
FT_RASTER_FLAG_AA = 0x1
FT_RASTER_FLAG_DIRECT = 0x2
FT_RASTER_FLAG_CLIP = 0x4

class FT_Raster_Params_(Structure):
    _fields_ = [
        ("target", POINTER(FT_Bitmap)),
        ("source", c_void_p),
        ("flags", c_int),
        ("gray_spans", FT_SpanFunc),
        ("black_spans", FT_SpanFunc),
        ("bit_test", FT_Raster_BitTest_Func),
        ("bit_set", FT_Raster_BitSet_Func),
        ("user", FT_Pointer),
        ("clip_box", FT_BBox),
        ]

# typedef FT_Raster_Params
FT_Raster_Params = FT_Raster_Params_


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/ftimage.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cleanupNamespace(globals())

