#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_freetype import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/usr/local/include/freetype2/freetype/fttypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~ Code block from "/usr/local/include/freetype2/freetype/config/ftconfig.h" ~~~

# typedef FT_Int32
FT_Int32 = c_int

#~ Code block from "/usr/local/include/freetype2/freetype/config/ftoption.h" ~~~

FT_RENDER_POOL_SIZE = 16384L

#~ line: 317, skipped: 10 ~~~~~~

FT_MAX_MODULES = 32

#~ Code block from "/usr/local/include/freetype2/freetype/fttypes.h" ~~~

# typedef FT_Bool
FT_Bool = c_ubyte

# typedef FT_Byte
FT_Byte = c_ubyte

# typedef FT_String
FT_String = c_char

# typedef FT_Short
FT_Short = c_short

# typedef FT_UShort
FT_UShort = c_ushort

# typedef FT_Int
FT_Int = c_int

# typedef FT_UInt
FT_UInt = c_uint

# typedef FT_Long
FT_Long = c_long

# typedef FT_ULong
FT_ULong = c_ulong

# typedef FT_F26Dot6
FT_F26Dot6 = c_long

# typedef FT_Fixed
FT_Fixed = c_long

# typedef FT_Error
FT_Error = c_int

# typedef FT_Pointer
FT_Pointer = c_void_p

class FT_Matrix_(Structure):
    _fields_ = [
        ("xx", FT_Fixed),
        ("xy", FT_Fixed),
        ("yx", FT_Fixed),
        ("yy", FT_Fixed),
        ]

# typedef FT_Matrix
FT_Matrix = FT_Matrix_

# typedef FT_Generic_Finalizer
FT_Generic_Finalizer = POINTER(CFUNCTYPE(None, FT_Pointer))

class FT_Generic_(Structure):
    _fields_ = [
        ("data", FT_Pointer),
        ("finalizer", FT_Generic_Finalizer),
        ]

# typedef FT_Generic
FT_Generic = FT_Generic_

# typedef FT_ListNode
FT_ListNode = POINTER("FT_ListNodeRec_")

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

# typedef FT_ListRec
FT_ListRec = FT_ListRec_


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/usr/local/include/freetype2/freetype/fttypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cleanupNamespace(globals())

