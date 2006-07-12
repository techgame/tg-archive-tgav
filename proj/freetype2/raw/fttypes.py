#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_freetype import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/usr/local/include/freetype2/freetype/fttypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~ Code block from "/usr/local/include/freetype2/freetype/config/ftconfig.h" ~~~

class FT_Int32(c_int):
    """typedef FT_Int32"""

#~ Code block from "/usr/local/include/freetype2/freetype/config/ftoption.h" ~~~

FT_RENDER_POOL_SIZE = 16384L

#~ line: 297, skipped: 10 ~~~~~~

FT_MAX_MODULES = 32

#~ Code block from "/usr/local/include/freetype2/freetype/fttypes.h" ~~~

class FT_Byte(c_ubyte):
    """typedef FT_Byte"""

class FT_String(c_char):
    """typedef FT_String"""

class FT_Short(c_short):
    """typedef FT_Short"""

class FT_UShort(c_ushort):
    """typedef FT_UShort"""

class FT_Int(c_int):
    """typedef FT_Int"""

class FT_UInt(c_uint):
    """typedef FT_UInt"""

class FT_Long(c_long):
    """typedef FT_Long"""

class FT_ULong(c_ulong):
    """typedef FT_ULong"""

class FT_F26Dot6(c_long):
    """typedef FT_F26Dot6"""

class FT_Fixed(c_long):
    """typedef FT_Fixed"""

class FT_Error(c_int):
    """typedef FT_Error"""

FT_Pointer = c_void_p # typedef FT_Pointer

class FT_Matrix_(Structure):
    _fields_ = [
        ("xx", FT_Fixed),
        ("xy", FT_Fixed),
        ("yx", FT_Fixed),
        ("yy", FT_Fixed),
        ]

FT_Matrix = FT_Matrix_ # typedef FT_Matrix

FT_Generic_Finalizer = POINTER(CFUNCTYPE(None, FT_Pointer)) # typedef FT_Generic_Finalizer

class FT_Generic_(Structure):
    _fields_ = [
        ("data", FT_Pointer),
        ("finalizer", FT_Generic_Finalizer),
        ]

FT_Generic = FT_Generic_ # typedef FT_Generic

FT_ListNode = POINTER("FT_ListNodeRec_") # typedef FT_ListNode

class FT_ListNodeRec_(Structure):
    _fields_ = [
        ("prev", FT_ListNode),
        ("next", FT_ListNode),
        ("data", FT_Pointer),
        ]
FT_ListNode.set_type(FT_ListNodeRec_)

class FT_ListRec_(Structure):
    _fields_ = [
        ("head", FT_ListNode),
        ("tail", FT_ListNode),
        ]

FT_ListRec = FT_ListRec_ # typedef FT_ListRec


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/fttypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

