#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_cepstral import *
from swift_defs import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "inc/swift_params.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bind(POINTER(swift_params), [POINTER(swift_params)])
def swift_params_ref(params, _api_=None): 
    """swift_params_ref(params)
    
        params : POINTER(swift_params)
    """
    return _api_(params)
    

#~ line: 67, skipped: 12 ~~~~~~

@bind(swift_result_t, [POINTER(swift_params)])
def swift_params_delete(params, _api_=None): 
    """swift_params_delete(params)
    
        params : POINTER(swift_params)
    """
    return _api_(params)
    

#~ line: 76, skipped: 9 ~~~~~~

@bind(swift_result_t, [POINTER(swift_params)])
def swift_params_dump(params, _api_=None): 
    """swift_params_dump(params)
    
        params : POINTER(swift_params)
    """
    return _api_(params)
    

#~ line: 89, skipped: 13 ~~~~~~

@bind(c_char_p, [POINTER(swift_params), c_char_p, POINTER(swift_val)])
def swift_params_set_val(params, name, val, _api_=None): 
    """swift_params_set_val(params, name, val)
    
        params : POINTER(swift_params)
        name : c_char_p
        val : POINTER(swift_val)
    """
    return _api_(params, name, val)
    

#~ line: 119, skipped: 30 ~~~~~~

# typedef swift_params_iterator
swift_params_iterator = POINTER(CFUNCTYPE(swift_result_t, POINTER(swift_params), c_char_p, POINTER(swift_val), c_void_p))

#~ line: 139, skipped: 20 ~~~~~~

@bind(swift_result_t, [POINTER(swift_params), swift_params_iterator, c_void_p])
def swift_params_foreach(params, itor, udata, _api_=None): 
    """swift_params_foreach(params, itor, udata)
    
        params : POINTER(swift_params)
        itor : swift_params_iterator
        udata : c_void_p
    """
    return _api_(params, itor, udata)
    

#~ line: 155, skipped: 16 ~~~~~~

@bind(POINTER(swift_val), [POINTER(swift_params), c_char_p, POINTER(swift_val)])
def swift_params_get_val(params, name, def_, _api_=None): 
    """swift_params_get_val(params, name, def_)
    
        params : POINTER(swift_params)
        name : c_char_p
        def_ : POINTER(swift_val)
    """
    return _api_(params, name, def_)
    

#~ line: 173, skipped: 18 ~~~~~~

@bind(c_char_p, [POINTER(swift_params), c_char_p, c_char_p])
def swift_params_get_string(params, name, def_, _api_=None): 
    """swift_params_get_string(params, name, def_)
    
        params : POINTER(swift_params)
        name : c_char_p
        def_ : c_char_p
    """
    return _api_(params, name, def_)
    

#~ line: 188, skipped: 15 ~~~~~~

@bind(c_int, [POINTER(swift_params), c_char_p, c_int])
def swift_params_get_int(params, name, def_, _api_=None): 
    """swift_params_get_int(params, name, def_)
    
        params : POINTER(swift_params)
        name : c_char_p
        def_ : c_int
    """
    return _api_(params, name, def_)
    

#~ line: 204, skipped: 16 ~~~~~~

@bind(c_float, [POINTER(swift_params), c_char_p, c_float])
def swift_params_get_float(params, name, def_, _api_=None): 
    """swift_params_get_float(params, name, def_)
    
        params : POINTER(swift_params)
        name : c_char_p
        def_ : c_float
    """
    return _api_(params, name, def_)
    

#~ line: 238, skipped: 34 ~~~~~~

@bind(c_int, [c_char_p, POINTER(swift_val)])
def swift_param_validate(name, val, _api_=None): 
    """swift_param_validate(name, val)
    
        name : c_char_p
        val : POINTER(swift_val)
    """
    return _api_(name, val)
    

#~ line: 251, skipped: 13 ~~~~~~

@bind(swift_result_t, [POINTER(swift_params), POINTER(swift_params)])
def swift_params_set_params(to, from_, _api_=None): 
    """swift_params_set_params(to, from_)
    
        to : POINTER(swift_params)
        from_ : POINTER(swift_params)
    """
    return _api_(to, from_)
    

#~ line: 265, skipped: 14 ~~~~~~

@bind(swift_result_t, [POINTER(swift_params), c_char_p, POINTER(swift_val)])
def swift_params_set_param(to, name, val, _api_=None): 
    """swift_params_set_param(to, name, val)
    
        to : POINTER(swift_params)
        name : c_char_p
        val : POINTER(swift_val)
    """
    return _api_(to, name, val)
    

#~ line: 274, skipped: 9 ~~~~~~

@bind(POINTER(swift_val), [c_int])
def swift_val_int(i, _api_=None): 
    """swift_val_int(i)
    
        i : c_int
    """
    return _api_(i)
    

#~ line: 283, skipped: 9 ~~~~~~

@bind(POINTER(swift_val), [c_float])
def swift_val_float(f, _api_=None): 
    """swift_val_float(f)
    
        f : c_float
    """
    return _api_(f)
    

#~ line: 292, skipped: 9 ~~~~~~

@bind(POINTER(swift_val), [c_char_p])
def swift_val_string(s, _api_=None): 
    """swift_val_string(s)
    
        s : c_char_p
    """
    return _api_(s)
    

#~ line: 303, skipped: 11 ~~~~~~

@bind(POINTER(swift_val), [POINTER(c_ushort)])
def swift_val_string16(s, _api_=None): 
    """swift_val_string16(s)
    
        s : POINTER(c_ushort)
    """
    return _api_(s)
    

#~ line: 312, skipped: 9 ~~~~~~

@bind(POINTER(swift_val), [POINTER(swift_val)])
def swift_val_ref(val, _api_=None): 
    """swift_val_ref(val)
    
        val : POINTER(swift_val)
    """
    return _api_(val)
    

#~ line: 324, skipped: 12 ~~~~~~

@bind(swift_result_t, [POINTER(swift_val)])
def swift_val_delete(val, _api_=None): 
    """swift_val_delete(val)
    
        val : POINTER(swift_val)
    """
    return _api_(val)
    

#~ line: 333, skipped: 9 ~~~~~~

@bind(c_int, [POINTER(swift_val)])
def swift_val_get_int(val, _api_=None): 
    """swift_val_get_int(val)
    
        val : POINTER(swift_val)
    """
    return _api_(val)
    

#~ line: 342, skipped: 9 ~~~~~~

@bind(c_float, [POINTER(swift_val)])
def swift_val_get_float(val, _api_=None): 
    """swift_val_get_float(val)
    
        val : POINTER(swift_val)
    """
    return _api_(val)
    

#~ line: 359, skipped: 17 ~~~~~~

@bind(c_char_p, [POINTER(swift_val)])
def swift_val_get_string(val, _api_=None): 
    """swift_val_get_string(val)
    
        val : POINTER(swift_val)
    """
    return _api_(val)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "inc/swift_params.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

