#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_freetype import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/usr/local/include/freetype2/freetype/ftoutln.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bind(FT_Error, [POINTER(FT_Outline), POINTER(FT_Outline_Funcs), c_void_p])
def FT_Outline_Decompose(outline, func_interface, user, _api_=None): 
    """FT_Outline_Decompose(outline, func_interface, user)
    
        outline : POINTER(FT_Outline)
        func_interface : POINTER(FT_Outline_Funcs)
        user : c_void_p
    """
    return _api_(outline, func_interface, user)
    

@bind(FT_Error, [FT_Library, FT_UInt, FT_Int, POINTER(FT_Outline)])
def FT_Outline_New(library, numPoints, numContours, anoutline, _api_=None): 
    """FT_Outline_New(library, numPoints, numContours, anoutline)
    
        library : FT_Library
        numPoints : FT_UInt
        numContours : FT_Int
        anoutline : POINTER(FT_Outline)
    """
    return _api_(library, numPoints, numContours, anoutline)
    

@bind(FT_Error, [FT_Memory, FT_UInt, FT_Int, POINTER(FT_Outline)])
def FT_Outline_New_Internal(memory, numPoints, numContours, anoutline, _api_=None): 
    """FT_Outline_New_Internal(memory, numPoints, numContours, anoutline)
    
        memory : FT_Memory
        numPoints : FT_UInt
        numContours : FT_Int
        anoutline : POINTER(FT_Outline)
    """
    return _api_(memory, numPoints, numContours, anoutline)
    

@bind(FT_Error, [FT_Library, POINTER(FT_Outline)])
def FT_Outline_Done(library, outline, _api_=None): 
    """FT_Outline_Done(library, outline)
    
        library : FT_Library
        outline : POINTER(FT_Outline)
    """
    return _api_(library, outline)
    

@bind(FT_Error, [FT_Memory, POINTER(FT_Outline)])
def FT_Outline_Done_Internal(memory, outline, _api_=None): 
    """FT_Outline_Done_Internal(memory, outline)
    
        memory : FT_Memory
        outline : POINTER(FT_Outline)
    """
    return _api_(memory, outline)
    

@bind(FT_Error, [POINTER(FT_Outline)])
def FT_Outline_Check(outline, _api_=None): 
    """FT_Outline_Check(outline)
    
        outline : POINTER(FT_Outline)
    """
    return _api_(outline)
    

@bind(None, [POINTER(FT_Outline), POINTER(FT_BBox)])
def FT_Outline_Get_CBox(outline, acbox, _api_=None): 
    """FT_Outline_Get_CBox(outline, acbox)
    
        outline : POINTER(FT_Outline)
        acbox : POINTER(FT_BBox)
    """
    return _api_(outline, acbox)
    

@bind(None, [POINTER(FT_Outline), FT_Pos, FT_Pos])
def FT_Outline_Translate(outline, xOffset, yOffset, _api_=None): 
    """FT_Outline_Translate(outline, xOffset, yOffset)
    
        outline : POINTER(FT_Outline)
        xOffset : FT_Pos
        yOffset : FT_Pos
    """
    return _api_(outline, xOffset, yOffset)
    

@bind(FT_Error, [POINTER(FT_Outline), POINTER(FT_Outline)])
def FT_Outline_Copy(source, target, _api_=None): 
    """FT_Outline_Copy(source, target)
    
        source : POINTER(FT_Outline)
        target : POINTER(FT_Outline)
    """
    return _api_(source, target)
    

@bind(None, [POINTER(FT_Outline), POINTER(FT_Matrix)])
def FT_Outline_Transform(outline, matrix, _api_=None): 
    """FT_Outline_Transform(outline, matrix)
    
        outline : POINTER(FT_Outline)
        matrix : POINTER(FT_Matrix)
    """
    return _api_(outline, matrix)
    

@bind(FT_Error, [POINTER(FT_Outline), FT_Pos])
def FT_Outline_Embolden(outline, strength, _api_=None): 
    """FT_Outline_Embolden(outline, strength)
    
        outline : POINTER(FT_Outline)
        strength : FT_Pos
    """
    return _api_(outline, strength)
    

@bind(None, [POINTER(FT_Outline)])
def FT_Outline_Reverse(outline, _api_=None): 
    """FT_Outline_Reverse(outline)
    
        outline : POINTER(FT_Outline)
    """
    return _api_(outline)
    

@bind(FT_Error, [FT_Library, POINTER(FT_Outline), POINTER(FT_Bitmap)])
def FT_Outline_Get_Bitmap(library, outline, abitmap, _api_=None): 
    """FT_Outline_Get_Bitmap(library, outline, abitmap)
    
        library : FT_Library
        outline : POINTER(FT_Outline)
        abitmap : POINTER(FT_Bitmap)
    """
    return _api_(library, outline, abitmap)
    

@bind(FT_Error, [FT_Library, POINTER(FT_Outline), POINTER(FT_Raster_Params)])
def FT_Outline_Render(library, outline, params, _api_=None): 
    """FT_Outline_Render(library, outline, params)
    
        library : FT_Library
        outline : POINTER(FT_Outline)
        params : POINTER(FT_Raster_Params)
    """
    return _api_(library, outline, params)
    

class FT_Orientation(c_int):
    '''enum FT_Orientation''' 
    FT_ORIENTATION_TRUETYPE = 0
    FT_ORIENTATION_POSTSCRIPT = 1
    FT_ORIENTATION_FILL_RIGHT = 0
    FT_ORIENTATION_FILL_LEFT = 1
    FT_ORIENTATION_NONE = 2

@bind(FT_Orientation, [POINTER(FT_Outline)])
def FT_Outline_Get_Orientation(outline, _api_=None): 
    """FT_Outline_Get_Orientation(outline)
    
        outline : POINTER(FT_Outline)
    """
    return _api_(outline)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/ftoutln.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cleanupNamespace(globals())

