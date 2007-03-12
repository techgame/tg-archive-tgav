##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2007  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from struct import pack, unpack
import ctypes, ctypes.util
from ctypes import c_void_p

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libCoreFoundationPath = ctypes.util.find_library("CoreFoundation")
libCoreFoundation = ctypes.cdll.LoadLibrary(libCoreFoundationPath)

CFStringRef = ctypes.c_void_p
kCFStringEncodingUTF8 = 0x8000100
CFURLRef = ctypes.c_void_p

Boolean = ctypes.c_ubyte
booleanFalse = Boolean(0)
booleanTrue = Boolean(1)

c_appleid = ctypes.c_uint32

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def asCFString(astr):
    utf8_astr = astr.encode('utf8')
    p_astr = ctypes.c_char_p(utf8_astr)
    cfs_astr = libCoreFoundation.CFStringCreateWithCString(0, p_astr, kCFStringEncodingUTF8)
    assert len(astr) == libCoreFoundation.CFStringGetLength(cfs_astr)
    return CFStringRef(cfs_astr)

def asCFURL(astr):
    cfs_astr = asCFString(astr)
    cfurl_astr = libCoreFoundation.CFURLCreateWithString(0, cfs_astr, None)
    return CFURLRef(cfurl_astr)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def fromAppleId(strAppleId): 
    return unpack('!I', strAppleId)[0]
def toAppleId(intAppleId): 
    return pack('!I', intAppleId)

