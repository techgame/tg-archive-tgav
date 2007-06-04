#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_freetype import *
from freetype import *
from ftglyph import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/usr/local/include/freetype2/freetype/ftstroke.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FT_StrokerRec_ = c_void_p # Structure with empty _fields_
# typedef FT_Stroker
FT_Stroker = POINTER(FT_StrokerRec_)

class FT_Stroker_LineJoin(c_int):
    '''enum FT_Stroker_LineJoin''' 
    FT_STROKER_LINEJOIN_ROUND = 0
    FT_STROKER_LINEJOIN_BEVEL = 1
    FT_STROKER_LINEJOIN_MITER = 2
    lookup = {
        0: "FT_STROKER_LINEJOIN_ROUND",
        1: "FT_STROKER_LINEJOIN_BEVEL",
        2: "FT_STROKER_LINEJOIN_MITER",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

class FT_Stroker_LineCap(c_int):
    '''enum FT_Stroker_LineCap''' 
    FT_STROKER_LINECAP_BUTT = 0
    FT_STROKER_LINECAP_ROUND = 1
    FT_STROKER_LINECAP_SQUARE = 2
    lookup = {
        0: "FT_STROKER_LINECAP_BUTT",
        1: "FT_STROKER_LINECAP_ROUND",
        2: "FT_STROKER_LINECAP_SQUARE",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

class FT_StrokerBorder(c_int):
    '''enum FT_StrokerBorder''' 
    FT_STROKER_BORDER_LEFT = 0
    FT_STROKER_BORDER_RIGHT = 1
    lookup = {
        0: "FT_STROKER_BORDER_LEFT",
        1: "FT_STROKER_BORDER_RIGHT",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

@bind(FT_StrokerBorder, [POINTER(FT_Outline)])
def FT_Outline_GetInsideBorder(outline, _api_=None): 
    """FT_Outline_GetInsideBorder(outline)
    
        outline : POINTER(FT_Outline)
    """
    return _api_(outline)
    

@bind(FT_StrokerBorder, [POINTER(FT_Outline)])
def FT_Outline_GetOutsideBorder(outline, _api_=None): 
    """FT_Outline_GetOutsideBorder(outline)
    
        outline : POINTER(FT_Outline)
    """
    return _api_(outline)
    

@bind(FT_Error, [FT_Library, POINTER(FT_Stroker)])
def FT_Stroker_New(library, astroker, _api_=None): 
    """FT_Stroker_New(library, astroker)
    
        library : FT_Library
        astroker : POINTER(FT_Stroker)
    """
    return _api_(library, astroker)
    

@bind(None, [FT_Stroker, FT_Fixed, FT_Stroker_LineCap, FT_Stroker_LineJoin, FT_Fixed])
def FT_Stroker_Set(stroker, radius, line_cap, line_join, miter_limit, _api_=None): 
    """FT_Stroker_Set(stroker, radius, line_cap, line_join, miter_limit)
    
        stroker : FT_Stroker
        radius : FT_Fixed
        line_cap : FT_Stroker_LineCap
        line_join : FT_Stroker_LineJoin
        miter_limit : FT_Fixed
    """
    return _api_(stroker, radius, line_cap, line_join, miter_limit)
    

@bind(None, [FT_Stroker])
def FT_Stroker_Rewind(stroker, _api_=None): 
    """FT_Stroker_Rewind(stroker)
    
        stroker : FT_Stroker
    """
    return _api_(stroker)
    

@bind(FT_Error, [FT_Stroker, POINTER(FT_Outline), FT_Bool])
def FT_Stroker_ParseOutline(stroker, outline, opened, _api_=None): 
    """FT_Stroker_ParseOutline(stroker, outline, opened)
    
        stroker : FT_Stroker
        outline : POINTER(FT_Outline)
        opened : FT_Bool
    """
    return _api_(stroker, outline, opened)
    

@bind(FT_Error, [FT_Stroker, POINTER(FT_Vector), FT_Bool])
def FT_Stroker_BeginSubPath(stroker, to, open, _api_=None): 
    """FT_Stroker_BeginSubPath(stroker, to, open)
    
        stroker : FT_Stroker
        to : POINTER(FT_Vector)
        open : FT_Bool
    """
    return _api_(stroker, to, open)
    

@bind(FT_Error, [FT_Stroker])
def FT_Stroker_EndSubPath(stroker, _api_=None): 
    """FT_Stroker_EndSubPath(stroker)
    
        stroker : FT_Stroker
    """
    return _api_(stroker)
    

@bind(FT_Error, [FT_Stroker, POINTER(FT_Vector)])
def FT_Stroker_LineTo(stroker, to, _api_=None): 
    """FT_Stroker_LineTo(stroker, to)
    
        stroker : FT_Stroker
        to : POINTER(FT_Vector)
    """
    return _api_(stroker, to)
    

@bind(FT_Error, [FT_Stroker, POINTER(FT_Vector), POINTER(FT_Vector)])
def FT_Stroker_ConicTo(stroker, control, to, _api_=None): 
    """FT_Stroker_ConicTo(stroker, control, to)
    
        stroker : FT_Stroker
        control : POINTER(FT_Vector)
        to : POINTER(FT_Vector)
    """
    return _api_(stroker, control, to)
    

@bind(FT_Error, [FT_Stroker, POINTER(FT_Vector), POINTER(FT_Vector), POINTER(FT_Vector)])
def FT_Stroker_CubicTo(stroker, control1, control2, to, _api_=None): 
    """FT_Stroker_CubicTo(stroker, control1, control2, to)
    
        stroker : FT_Stroker
        control1 : POINTER(FT_Vector)
        control2 : POINTER(FT_Vector)
        to : POINTER(FT_Vector)
    """
    return _api_(stroker, control1, control2, to)
    

@bind(FT_Error, [FT_Stroker, FT_StrokerBorder, POINTER(c_uint), POINTER(c_uint)])
def FT_Stroker_GetBorderCounts(stroker, border, anum_points, anum_contours, _api_=None): 
    """FT_Stroker_GetBorderCounts(stroker, border, anum_points, anum_contours)
    
        stroker : FT_Stroker
        border : FT_StrokerBorder
        anum_points : POINTER(c_uint)
        anum_contours : POINTER(c_uint)
    """
    return _api_(stroker, border, anum_points, anum_contours)
    

@bind(None, [FT_Stroker, FT_StrokerBorder, POINTER(FT_Outline)])
def FT_Stroker_ExportBorder(stroker, border, outline, _api_=None): 
    """FT_Stroker_ExportBorder(stroker, border, outline)
    
        stroker : FT_Stroker
        border : FT_StrokerBorder
        outline : POINTER(FT_Outline)
    """
    return _api_(stroker, border, outline)
    

@bind(FT_Error, [FT_Stroker, POINTER(c_uint), POINTER(c_uint)])
def FT_Stroker_GetCounts(stroker, anum_points, anum_contours, _api_=None): 
    """FT_Stroker_GetCounts(stroker, anum_points, anum_contours)
    
        stroker : FT_Stroker
        anum_points : POINTER(c_uint)
        anum_contours : POINTER(c_uint)
    """
    return _api_(stroker, anum_points, anum_contours)
    

@bind(None, [FT_Stroker, POINTER(FT_Outline)])
def FT_Stroker_Export(stroker, outline, _api_=None): 
    """FT_Stroker_Export(stroker, outline)
    
        stroker : FT_Stroker
        outline : POINTER(FT_Outline)
    """
    return _api_(stroker, outline)
    

@bind(None, [FT_Stroker])
def FT_Stroker_Done(stroker, _api_=None): 
    """FT_Stroker_Done(stroker)
    
        stroker : FT_Stroker
    """
    return _api_(stroker)
    

@bind(FT_Error, [POINTER(FT_Glyph), FT_Stroker, FT_Bool])
def FT_Glyph_Stroke(pglyph, stroker, destroy, _api_=None): 
    """FT_Glyph_Stroke(pglyph, stroker, destroy)
    
        pglyph : POINTER(FT_Glyph)
        stroker : FT_Stroker
        destroy : FT_Bool
    """
    return _api_(pglyph, stroker, destroy)
    

@bind(FT_Error, [POINTER(FT_Glyph), FT_Stroker, FT_Bool, FT_Bool])
def FT_Glyph_StrokeBorder(pglyph, stroker, inside, destroy, _api_=None): 
    """FT_Glyph_StrokeBorder(pglyph, stroker, inside, destroy)
    
        pglyph : POINTER(FT_Glyph)
        stroker : FT_Stroker
        inside : FT_Bool
        destroy : FT_Bool
    """
    return _api_(pglyph, stroker, inside, destroy)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/ftstroke.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cleanupNamespace(globals())

