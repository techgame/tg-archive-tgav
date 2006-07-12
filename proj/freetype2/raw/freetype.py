#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_freetype import *
from fttypes import *
from ftimage import *
from ftsystem import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/usr/local/include/freetype2/freetype/freetype.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FREETYPE_MAJOR = 2
FREETYPE_MINOR = 1
FREETYPE_PATCH = 9

class FT_Glyph_Metrics_(Structure):
    _fields_ = [
        ("width", FT_Pos),
        ("height", FT_Pos),
        ("horiBearingX", FT_Pos),
        ("horiBearingY", FT_Pos),
        ("horiAdvance", FT_Pos),
        ("vertBearingX", FT_Pos),
        ("vertBearingY", FT_Pos),
        ("vertAdvance", FT_Pos),
        ]

FT_Glyph_Metrics = FT_Glyph_Metrics_ # typedef FT_Glyph_Metrics

class FT_Bitmap_Size_(Structure):
    _fields_ = [
        ("height", FT_Short),
        ("width", FT_Short),
        ("size", FT_Pos),
        ("x_ppem", FT_Pos),
        ("y_ppem", FT_Pos),
        ]

FT_Bitmap_Size = FT_Bitmap_Size_ # typedef FT_Bitmap_Size

class FT_LibraryRec_(Structure):
    _fields_ = []
FT_Library = POINTER(FT_LibraryRec_) # typedef FT_Library

class FT_ModuleRec_(Structure):
    _fields_ = []
FT_Module = POINTER(FT_ModuleRec_) # typedef FT_Module

class FT_DriverRec_(Structure):
    _fields_ = []
FT_Driver = POINTER(FT_DriverRec_) # typedef FT_Driver

FT_Face = POINTER("FT_FaceRec_") # typedef FT_Face

FT_Size = POINTER("FT_SizeRec_") # typedef FT_Size

FT_GlyphSlot = POINTER("FT_GlyphSlotRec_") # typedef FT_GlyphSlot

FT_CharMap = POINTER("FT_CharMapRec_") # typedef FT_CharMap

class FT_Encoding_(c_int):
    '''enum FT_Encoding_''' 
    FT_ENCODING_NONE = 0
    FT_ENCODING_MS_SYMBOL = 1937337698
    FT_ENCODING_UNICODE = 1970170211
    FT_ENCODING_SJIS = 1936353651
    FT_ENCODING_GB2312 = 1734484000
    FT_ENCODING_BIG5 = 1651074869
    FT_ENCODING_WANSUNG = 2002873971
    FT_ENCODING_JOHAB = 1785686113
    FT_ENCODING_MS_SJIS = 1936353651
    FT_ENCODING_MS_GB2312 = 1734484000
    FT_ENCODING_MS_BIG5 = 1651074869
    FT_ENCODING_MS_WANSUNG = 2002873971
    FT_ENCODING_MS_JOHAB = 1785686113
    FT_ENCODING_ADOBE_STANDARD = 1094995778
    FT_ENCODING_ADOBE_EXPERT = 1094992453
    FT_ENCODING_ADOBE_CUSTOM = 1094992451
    FT_ENCODING_ADOBE_LATIN_1 = 1818326065
    FT_ENCODING_OLD_LATIN_2 = 1818326066
    FT_ENCODING_APPLE_ROMAN = 1634889070

FT_Encoding = FT_Encoding_ # typedef FT_Encoding

class FT_CharMapRec_(Structure):
    _fields_ = [
        ("face", FT_Face),
        ("encoding", FT_Encoding),
        ("platform_id", FT_UShort),
        ("encoding_id", FT_UShort),
        ]
FT_CharMap.set_type(FT_CharMapRec_)

class FT_Face_InternalRec_(Structure):
    _fields_ = []
FT_Face_Internal = POINTER(FT_Face_InternalRec_) # typedef FT_Face_Internal

class FT_FaceRec_(Structure):
    _fields_ = [
        ("num_faces", FT_Long),
        ("face_index", FT_Long),
        ("face_flags", FT_Long),
        ("style_flags", FT_Long),
        ("num_glyphs", FT_Long),
        ("family_name", POINTER(FT_String)),
        ("style_name", POINTER(FT_String)),
        ("num_fixed_sizes", FT_Int),
        ("available_sizes", POINTER(FT_Bitmap_Size)),
        ("num_charmaps", FT_Int),
        ("charmaps", POINTER(FT_CharMap)),
        ("generic", FT_Generic),
        ("bbox", FT_BBox),
        ("units_per_EM", FT_UShort),
        ("ascender", FT_Short),
        ("descender", FT_Short),
        ("height", FT_Short),
        ("max_advance_width", FT_Short),
        ("max_advance_height", FT_Short),
        ("underline_position", FT_Short),
        ("underline_thickness", FT_Short),
        ("glyph", FT_GlyphSlot),
        ("size", FT_Size),
        ("charmap", FT_CharMap),
        ("driver", FT_Driver),
        ("memory", FT_Memory),
        ("stream", FT_Stream),
        ("sizes_list", FT_ListRec),
        ("autohint", FT_Generic),
        ("extensions", c_void_p),
        ("internal", FT_Face_Internal),
        ]
FT_Face.set_type(FT_FaceRec_)

FT_FACE_FLAG_SCALABLE = ( 1L << 0 )
FT_FACE_FLAG_FIXED_SIZES = ( 1L << 1 )
FT_FACE_FLAG_FIXED_WIDTH = ( 1L << 2 )
FT_FACE_FLAG_SFNT = ( 1L << 3 )
FT_FACE_FLAG_HORIZONTAL = ( 1L << 4 )
FT_FACE_FLAG_VERTICAL = ( 1L << 5 )
FT_FACE_FLAG_KERNING = ( 1L << 6 )
FT_FACE_FLAG_FAST_GLYPHS = ( 1L << 7 )
FT_FACE_FLAG_MULTIPLE_MASTERS = ( 1L << 8 )
FT_FACE_FLAG_GLYPH_NAMES = ( 1L << 9 )
FT_FACE_FLAG_EXTERNAL_STREAM = ( 1L << 10 )

FT_STYLE_FLAG_ITALIC = ( 1 << 0 )
FT_STYLE_FLAG_BOLD = ( 1 << 1 )

class FT_Size_InternalRec_(Structure):
    _fields_ = []
FT_Size_Internal = POINTER(FT_Size_InternalRec_) # typedef FT_Size_Internal

class FT_Size_Metrics_(Structure):
    _fields_ = [
        ("x_ppem", FT_UShort),
        ("y_ppem", FT_UShort),
        ("x_scale", FT_Fixed),
        ("y_scale", FT_Fixed),
        ("ascender", FT_Pos),
        ("descender", FT_Pos),
        ("height", FT_Pos),
        ("max_advance", FT_Pos),
        ]

FT_Size_Metrics = FT_Size_Metrics_ # typedef FT_Size_Metrics

class FT_SizeRec_(Structure):
    _fields_ = [
        ("face", FT_Face),
        ("generic", FT_Generic),
        ("metrics", FT_Size_Metrics),
        ("internal", FT_Size_Internal),
        ]
FT_Size.set_type(FT_SizeRec_)

class FT_SubGlyphRec_(Structure):
    _fields_ = []
FT_SubGlyph = POINTER(FT_SubGlyphRec_) # typedef FT_SubGlyph

class FT_Slot_InternalRec_(Structure):
    _fields_ = []
FT_Slot_Internal = POINTER(FT_Slot_InternalRec_) # typedef FT_Slot_Internal

class FT_GlyphSlotRec_(Structure):
    _fields_ = [
        ("library", FT_Library),
        ("face", FT_Face),
        ("next", FT_GlyphSlot),
        ("reserved", FT_UInt),
        ("generic", FT_Generic),
        ("metrics", FT_Glyph_Metrics),
        ("linearHoriAdvance", FT_Fixed),
        ("linearVertAdvance", FT_Fixed),
        ("advance", FT_Vector),
        ("format", FT_Glyph_Format),
        ("bitmap", FT_Bitmap),
        ("bitmap_left", FT_Int),
        ("bitmap_top", FT_Int),
        ("outline", FT_Outline),
        ("num_subglyphs", FT_UInt),
        ("subglyphs", FT_SubGlyph),
        ("control_data", c_void_p),
        ("control_len", c_long),
        ("lsb_delta", FT_Pos),
        ("rsb_delta", FT_Pos),
        ("other", c_void_p),
        ("internal", FT_Slot_Internal),
        ]
FT_GlyphSlot.set_type(FT_GlyphSlotRec_)

@bind(FT_Error, [POINTER(FT_Library)])
def FT_Init_FreeType(alibrary): pass


@bind(None, [FT_Library, POINTER(FT_Int), POINTER(FT_Int), POINTER(FT_Int)])
def FT_Library_Version(library, amajor, aminor, apatch): pass


@bind(FT_Error, [FT_Library])
def FT_Done_FreeType(library): pass


FT_OPEN_MEMORY = 0x1
FT_OPEN_STREAM = 0x2
FT_OPEN_PATHNAME = 0x4
FT_OPEN_DRIVER = 0x8
FT_OPEN_PARAMS = 0x10

class FT_Parameter_(Structure):
    _fields_ = [
        ("tag", FT_ULong),
        ("data", FT_Pointer),
        ]

FT_Parameter = FT_Parameter_ # typedef FT_Parameter

class FT_Open_Args_(Structure):
    _fields_ = [
        ("flags", FT_UInt),
        ("memory_base", POINTER(FT_Byte)),
        ("memory_size", FT_Long),
        ("pathname", POINTER(FT_String)),
        ("stream", FT_Stream),
        ("driver", FT_Module),
        ("num_params", FT_Int),
        ("params", POINTER(FT_Parameter)),
        ]

FT_Open_Args = FT_Open_Args_ # typedef FT_Open_Args

@bind(FT_Error, [FT_Library, POINTER(c_char), FT_Long, POINTER(FT_Face)])
def FT_New_Face(library, filepathname, face_index, aface): pass


@bind(FT_Error, [FT_Library, POINTER(FT_Byte), FT_Long, FT_Long, POINTER(FT_Face)])
def FT_New_Memory_Face(library, file_base, file_size, face_index, aface): pass


@bind(FT_Error, [FT_Library, POINTER(FT_Open_Args), FT_Long, POINTER(FT_Face)])
def FT_Open_Face(library, args, face_index, aface): pass


@bind(FT_Error, [FT_Face, POINTER(c_char)])
def FT_Attach_File(face, filepathname): pass


@bind(FT_Error, [FT_Face, POINTER(FT_Open_Args)])
def FT_Attach_Stream(face, parameters): pass


@bind(FT_Error, [FT_Face])
def FT_Done_Face(face): pass


@bind(FT_Error, [FT_Face, FT_F26Dot6, FT_F26Dot6, FT_UInt, FT_UInt])
def FT_Set_Char_Size(face, char_width, char_height, horz_resolution, vert_resolution): pass


@bind(FT_Error, [FT_Face, FT_UInt, FT_UInt])
def FT_Set_Pixel_Sizes(face, pixel_width, pixel_height): pass


@bind(FT_Error, [FT_Face, FT_UInt, FT_Int32])
def FT_Load_Glyph(face, glyph_index, load_flags): pass


@bind(FT_Error, [FT_Face, FT_ULong, FT_Int32])
def FT_Load_Char(face, char_code, load_flags): pass


FT_LOAD_DEFAULT = 0x0
FT_LOAD_NO_SCALE = 0x1
FT_LOAD_NO_HINTING = 0x2
FT_LOAD_RENDER = 0x4
FT_LOAD_NO_BITMAP = 0x8
FT_LOAD_VERTICAL_LAYOUT = 0x10
FT_LOAD_FORCE_AUTOHINT = 0x20
FT_LOAD_CROP_BITMAP = 0x40
FT_LOAD_PEDANTIC = 0x80
FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH = 0x200
FT_LOAD_NO_RECURSE = 0x400
FT_LOAD_IGNORE_TRANSFORM = 0x800
FT_LOAD_MONOCHROME = 0x1000
FT_LOAD_LINEAR_DESIGN = 0x2000

FT_LOAD_SBITS_ONLY = 0x4000
FT_LOAD_NO_AUTOHINT = 0x8000

@bind(None, [FT_Face, POINTER(FT_Matrix), POINTER(FT_Vector)])
def FT_Set_Transform(face, matrix, delta): pass


class FT_Render_Mode_(c_int):
    '''enum FT_Render_Mode_''' 
    FT_RENDER_MODE_NORMAL = 0
    FT_RENDER_MODE_LIGHT = 1
    FT_RENDER_MODE_MONO = 2
    FT_RENDER_MODE_LCD = 3
    FT_RENDER_MODE_LCD_V = 4
    FT_RENDER_MODE_MAX = 5

FT_Render_Mode = FT_Render_Mode_ # typedef FT_Render_Mode

@bind(FT_Error, [FT_GlyphSlot, FT_Render_Mode])
def FT_Render_Glyph(slot, render_mode): pass


@bind(FT_Error, [FT_Face, FT_UInt, FT_UInt, FT_UInt, POINTER(FT_Vector)])
def FT_Get_Kerning(face, left_glyph, right_glyph, kern_mode, akerning): pass


@bind(FT_Error, [FT_Face, FT_UInt, FT_Pointer, FT_UInt])
def FT_Get_Glyph_Name(face, glyph_index, buffer, buffer_max): pass


@bind(POINTER(c_char), [FT_Face])
def FT_Get_Postscript_Name(face): pass


@bind(FT_Error, [FT_Face, FT_Encoding])
def FT_Select_Charmap(face, encoding): pass


@bind(FT_Error, [FT_Face, FT_CharMap])
def FT_Set_Charmap(face, charmap): pass


@bind(FT_Int, [FT_CharMap])
def FT_Get_Charmap_Index(charmap): pass


@bind(FT_UInt, [FT_Face, FT_ULong])
def FT_Get_Char_Index(face, charcode): pass


@bind(FT_ULong, [FT_Face, POINTER(FT_UInt)])
def FT_Get_First_Char(face, agindex): pass


@bind(FT_ULong, [FT_Face, FT_ULong, POINTER(FT_UInt)])
def FT_Get_Next_Char(face, char_code, agindex): pass


@bind(FT_UInt, [FT_Face, POINTER(FT_String)])
def FT_Get_Name_Index(face, glyph_name): pass


@bind(FT_Long, [FT_Long, FT_Long, FT_Long])
def FT_MulDiv(a, b, c): pass


@bind(FT_Long, [FT_Long, FT_Long])
def FT_MulFix(a, b): pass


@bind(FT_Long, [FT_Long, FT_Long])
def FT_DivFix(a, b): pass


@bind(FT_Fixed, [FT_Fixed])
def FT_RoundFix(a): pass


@bind(FT_Fixed, [FT_Fixed])
def FT_CeilFix(a): pass


@bind(FT_Fixed, [FT_Fixed])
def FT_FloorFix(a): pass


@bind(None, [POINTER(FT_Vector), POINTER(FT_Matrix)])
def FT_Vector_Transform(vec, matrix): pass



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/freetype.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

