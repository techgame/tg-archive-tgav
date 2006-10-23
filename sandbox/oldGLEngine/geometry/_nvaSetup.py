##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

"""
PyOpenGL is a wonderful extension to python, allowing nearly direct interfacing with OpenGL in a cross-platform way.
However, myself and others have encountered a few quirks with VertexArrays and Numeric support, leading to copies (gasp!)
of large memory blocks.  As you can imagine (or may have experienced) this behavior rapidly degrades performance of
python applications using large vertex arrays.  Many have tried to correct the problem in the library, myself included,
yet it remains unsolved.  For me, it was far to difficult to try and patch the existing system in such a way as to insure
existing code would function, while expunging the data copying code.

So, the approach taken with NumericVertexArray is to reduce the complexity.  Therefore, Numeric extensions are *required*,
arrays are assumed to be contigious, and data is *never* copied.  The philiosophy is that if a copy is required, an exception
should be raised instead.  Hopefully, this extension will assist you in your pursuits.  And if not, use the source and make
something new; or send me a patch.  ;)

Enjoy!
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
from distutils.core import setup, Extension
from distutils.sysconfig import get_python_inc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__version__ = '0.1.1'
__license__ = 'BSD-style (See LICENSE)'
__platforms__ ='Windows', 
__author__ = 'Shane Holloway'
__author_email__ = 'shane.holloway@runeblade.com'
__url__ = 'http://www.runeblade.com/'
__keywords__ = ['OpenGL', 'PyOpenGL', 'Vertex Array', 'Numeric']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NumericDir = os.path.join(get_python_inc(plat_specific=1), "Numeric")

NumericVertexArray = Extension('NumericVertexArray', 
    include_dirs=[NumericDir], 
    sources=['NumericVertexArray.c'], 
    libraries=['opengl32', 'glu32'])

setup (name='NumericVertexArray',
       long_description=__doc__,
       version=__version__,
       author=__author__,
       author_email=__author_email__,
       url=__url__,
       platforms=__platforms__, 
       license=__license__,
       keywords=__keywords__,

       ext_modules=[NumericVertexArray],
       )

