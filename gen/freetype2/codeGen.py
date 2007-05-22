##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2006  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
from TG.gccxml.codeAnalyzer import CodeAnalyzer
from TG.gccxml.xforms.ctypes import AtomFilterVisitor, CCodeGenContext
from TG.gccxml.xforms.ctypes import utils

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

analyzer = CodeAnalyzer(
        inc=['/usr/local/include/freetype2'], 
        src=['src/genFreeType2.c'], 
        baseline=['src/baseline.c'])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FilterVisitor(AtomFilterVisitor):
    def onFunction(self, item):
        if item.extern and item.name.startswith('FT_'):
            self.select(item)

    def onEnumeration(self, item):
        if item.name.startswith('FT_'):
            self.select(item)
            print 'SWH:', item

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    includePrefixes = set([
        'FREETYPE', 'FT', 'TT',
        ])
    exludeDefines = set([
        'FT_CHAR_BIT',
        'FT_ALIGNMENT',
        'FT_MACINTOSH',
        'FT_INT64',
        'FT_BEGIN_STMNT',
        'FT_END_STMNT',
        'FT_DUMMY_STMNT',
        'FT_CALLBACK_TABLE',
        'FT_CALLBACK_TABLE_DEF',

        'FT_ERR_PREFIX',
        'FT_ERR_BASE',
        'FT_ERROR_START_LIST',
        'FT_ERROR_END_LIST',

        'FT_MODERR_START_LIST',
        'FT_MODERR_END_LIST',

        'FT_INT_MAX',
        'FT_LONG_MAX',
        'FT_UINT_MAX',
        'FT_ULONG_MAX',

        'FT_Outline_MoveTo_Func',
        'FT_Outline_LineTo_Func',
        'FT_Outline_ConicTo_Func',
        'FT_Outline_CubicTo_Func',
        'FT_Raster_Span_Func',
        'FT_Raster_New_Func',
        'FT_Raster_Done_Func',
        'FT_Raster_Reset_Func',
        'FT_Raster_Set_Mode_Func',
        'FT_Raster_Render_Func',
        ])

    def onPPDefine(self, item):
        body = item.body
        ident = item.ident
        if not item.body:
            return

        prefix = ident[:ident.find('_')]
        if prefix not in self.includePrefixes:
            return
        if 'SIZEOF' in ident:
            return
        if 'LOAD_TARGET' in ident:
            return
        if ident in self.exludeDefines:
            return

        if body.lower().startswith('0x'):
            if body[-1:] == 'U':
                item.body = body[:-1]

        self.select(item)

    def onPPInclude(self, item):
        pass #print '"%s" includes "%s"' % (item.file.name, item.filename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    root = analyzer.loadModel()
    context = CCodeGenContext(root)
    context.atomFilter = FilterVisitor()

    ciFilesByName = dict((os.path.basename(f.name), f) for f in context if f)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # remove these files because we don't need what they have

    ciFilesByName.pop('ftheader.h', None)
    ciFilesByName.pop('fterror.h', None)
    ciFilesByName.pop('ftstdlib.h', None)
    ciFilesByName.pop('internal.h', None)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # collapse ftconfig and ftoption into one file

    ftconfig = ciFilesByName.pop('ftconfig.h')
    ftconfig.lines.insert(0, (0, '#~ Code block from "%s" ~~~\n' % (ftconfig.name,)))

    ftoption = ciFilesByName.pop('ftoption.h')
    ftoption.lines.insert(0, (0, '#~ Code block from "%s" ~~~\n' % (ftoption.name,)))

    fttypes = ciFilesByName['fttypes.h']
    fttypes.lines.insert(0, (0, '#~ Code block from "%s" ~~~\n' % (fttypes.name,)))
    fttypes.prependFiles = [ftconfig, ftoption]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # setup imports

    for ciFile in ciFilesByName.itervalues():
        ciFile.importAll('_ctypes_freetype')
        ciFile.footer += '\ncleanupNamespace(globals())\n'

    ftimage = ciFilesByName['ftimage.h']
    ftimage.importAll(fttypes)

    ftsystem = ciFilesByName['ftsystem.h']
    ftsystem.importAll(fttypes)

    freetype = ciFilesByName['freetype.h']
    freetype.importAll(fttypes, ftimage, ftsystem)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # write output files

    context.outputPath = '../../proj/freetype2/raw/'
    print
    print "Writing out ctypes code:"
    print "========================"
    for ciFile in ciFilesByName.values():
        print 'Writing:', ciFile.filename
        ciFile.blockSeparator = ''
        ciFile.writeToFile()
        print 'Done Writing:', ciFile.filename
        print
    print

    utils.includeSupportIn(context.getOutputFilename('_ctypes_support.py'), copySource=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    main()

