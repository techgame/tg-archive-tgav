/* swift_exports.h: special Win32 DLL magic.
 *
 * Copyright (c) 2004-2006 Cepstral LLC, all rights reserved.
 *
 * Redistribution of this file, in whole or in part, with or without
 * modification, is not allowed without express written permission of
 * the copyright holder.
 */

/**
 * \file swift_exports.h
 * DLL import/export definitions for Windows platforms.
 **/

#ifndef _SWIFT_EXPORTS_H__
#define _SWIFT_EXPORTS_H__

/**
 * Tells API functions to be exported from swift.dll.
 * This is ignored on non-Windows platforms.
 **/
#if defined(_WIN32) && !defined(SWIFT_STATIC_LIBS)
# if defined SWIFT_EXPORTS
#  define SWIFT_API __declspec(dllexport)
# else
#  define SWIFT_API __declspec(dllimport)
# endif /* SWIFT_EXPORTS */
# if defined (_DEBUG)
#  define DEBUG
# endif
#else
# define SWIFT_API
#endif /* WIN32 */

/**
 * Declares exposed API functions as __cdecl for the PocketPC 2000 x86Emulator.
 * It is ignored by all other platforms.
 **/
#if defined (_WIN32_WCE_EMULATION)
# define SWIFT_CALLCONV __cdecl
#else
# define SWIFT_CALLCONV
#endif

#endif /* _SWIFT_EXPORTS_H__ */
