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
        ("points", POINTER(FT_Vector_)),
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

class FT_Glyph_Format_(c_int):
    '''enum FT_Glyph_Format_''' 
    FT_GLYPH_FORMAT_NONE = 0
    FT_GLYPH_FORMAT_COMPOSITE = 1668246896
    FT_GLYPH_FORMAT_BITMAP = 1651078259
    FT_GLYPH_FORMAT_OUTLINE = 1869968492
    FT_GLYPH_FORMAT_PLOTTER = 1886154612

# typedef FT_Glyph_Format
FT_Glyph_Format = FT_Glyph_Format_

FT_RASTER_FLAG_DEFAULT = 0x0
FT_RASTER_FLAG_AA = 0x1
FT_RASTER_FLAG_DIRECT = 0x2
FT_RASTER_FLAG_CLIP = 0x4


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/ftimage.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cleanupNamespace(globals())

