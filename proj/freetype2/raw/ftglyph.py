#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_freetype import *
from freetype import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/usr/local/include/freetype2/freetype/ftglyph.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FT_Glyph_Class_ = c_void_p # Structure with empty _fields_
# typedef FT_Glyph_Class
FT_Glyph_Class = FT_Glyph_Class_

# typedef FT_Glyph
FT_Glyph = POINTER("FT_GlyphRec_")

class FT_GlyphRec_(Structure):
    _fields_ = [
        ("library", FT_Library),
        ("clazz", POINTER(FT_Glyph_Class)),
        ("format", FT_Glyph_Format),
        ("advance", FT_Vector),
        ]
FT_Glyph.set_type(FT_GlyphRec_)

# typedef FT_GlyphRec
FT_GlyphRec = FT_GlyphRec_

# typedef FT_BitmapGlyph
FT_BitmapGlyph = POINTER("FT_BitmapGlyphRec_")

class FT_BitmapGlyphRec_(Structure):
    _fields_ = [
        ("root", FT_GlyphRec),
        ("left", FT_Int),
        ("top", FT_Int),
        ("bitmap", FT_Bitmap),
        ]
FT_BitmapGlyph.set_type(FT_BitmapGlyphRec_)

# typedef FT_BitmapGlyphRec
FT_BitmapGlyphRec = FT_BitmapGlyphRec_

# typedef FT_OutlineGlyph
FT_OutlineGlyph = POINTER("FT_OutlineGlyphRec_")

class FT_OutlineGlyphRec_(Structure):
    _fields_ = [
        ("root", FT_GlyphRec),
        ("outline", FT_Outline),
        ]
FT_OutlineGlyph.set_type(FT_OutlineGlyphRec_)

# typedef FT_OutlineGlyphRec
FT_OutlineGlyphRec = FT_OutlineGlyphRec_

@bind(FT_Error, [FT_GlyphSlot, POINTER(FT_Glyph)])
def FT_Get_Glyph(slot, aglyph, _api_=None): 
    """FT_Get_Glyph(slot, aglyph)
    
        slot : FT_GlyphSlot
        aglyph : POINTER(FT_Glyph)
    """
    return _api_(slot, aglyph)
    

@bind(FT_Error, [FT_Glyph, POINTER(FT_Glyph)])
def FT_Glyph_Copy(source, target, _api_=None): 
    """FT_Glyph_Copy(source, target)
    
        source : FT_Glyph
        target : POINTER(FT_Glyph)
    """
    return _api_(source, target)
    

@bind(FT_Error, [FT_Glyph, POINTER(FT_Matrix), POINTER(FT_Vector)])
def FT_Glyph_Transform(glyph, matrix, delta, _api_=None): 
    """FT_Glyph_Transform(glyph, matrix, delta)
    
        glyph : FT_Glyph
        matrix : POINTER(FT_Matrix)
        delta : POINTER(FT_Vector)
    """
    return _api_(glyph, matrix, delta)
    

class FT_Glyph_BBox_Mode_(c_int):
    '''enum FT_Glyph_BBox_Mode_''' 
    FT_GLYPH_BBOX_UNSCALED = 0
    FT_GLYPH_BBOX_SUBPIXELS = 0
    FT_GLYPH_BBOX_GRIDFIT = 1
    FT_GLYPH_BBOX_TRUNCATE = 2
    FT_GLYPH_BBOX_PIXELS = 3
    lookup = {
        0: "FT_GLYPH_BBOX_UNSCALED",
        0: "FT_GLYPH_BBOX_SUBPIXELS",
        1: "FT_GLYPH_BBOX_GRIDFIT",
        2: "FT_GLYPH_BBOX_TRUNCATE",
        3: "FT_GLYPH_BBOX_PIXELS",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

# typedef FT_Glyph_BBox_Mode
FT_Glyph_BBox_Mode = FT_Glyph_BBox_Mode_

@bind(None, [FT_Glyph, FT_UInt, POINTER(FT_BBox)])
def FT_Glyph_Get_CBox(glyph, bbox_mode, acbox, _api_=None): 
    """FT_Glyph_Get_CBox(glyph, bbox_mode, acbox)
    
        glyph : FT_Glyph
        bbox_mode : FT_UInt
        acbox : POINTER(FT_BBox)
    """
    return _api_(glyph, bbox_mode, acbox)
    

@bind(FT_Error, [POINTER(FT_Glyph), FT_Render_Mode, POINTER(FT_Vector), FT_Bool])
def FT_Glyph_To_Bitmap(the_glyph, render_mode, origin, destroy, _api_=None): 
    """FT_Glyph_To_Bitmap(the_glyph, render_mode, origin, destroy)
    
        the_glyph : POINTER(FT_Glyph)
        render_mode : FT_Render_Mode
        origin : POINTER(FT_Vector)
        destroy : FT_Bool
    """
    return _api_(the_glyph, render_mode, origin, destroy)
    

@bind(None, [FT_Glyph])
def FT_Done_Glyph(glyph, _api_=None): 
    """FT_Done_Glyph(glyph)
    
        glyph : FT_Glyph
    """
    return _api_(glyph)
    

@bind(None, [POINTER(FT_Matrix), POINTER(FT_Matrix)])
def FT_Matrix_Multiply(a, b, _api_=None): 
    """FT_Matrix_Multiply(a, b)
    
        a : POINTER(FT_Matrix)
        b : POINTER(FT_Matrix)
    """
    return _api_(a, b)
    

@bind(FT_Error, [POINTER(FT_Matrix)])
def FT_Matrix_Invert(matrix, _api_=None): 
    """FT_Matrix_Invert(matrix)
    
        matrix : POINTER(FT_Matrix)
    """
    return _api_(matrix)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/ftglyph.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cleanupNamespace(globals())

