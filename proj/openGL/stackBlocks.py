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

from contextlib import contextmanager

from TG.openGL.raw import gl
from TG.openGL.raw.errors import GLError

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _glPurgeStackOf(method):
    GLError.check()
    try:
        method = getattr(method, 'api', method)
        while 1:
            method()
            if glGetError() != 0:
                return
        
    except Exception:
        pass

@contextmanager
def glName(name):
    gl.glPushName(name)
    try:
        yield
    finally:
        gl.glPopName()

def glPurgeNames():
    _glPurgeStackOf(gl.glPopName)

@contextmanager
def glClientAttribs(mask):
    gl.glPushClientAttrib(mask)
    try:
        yield
    finally:
        gl.glPopClientAttrib()

def glPurgeClientAttribs():
    _glPurgeStackOf(gl.glPopClientAttribs)

@contextmanager
def glAttribs(mask):
    gl.glPushAttrib(mask)
    try:
        yield
    finally:
        gl.glPopAttrib()

def glPurgeAttribs():
    _glPurgeStackOf(gl.glPopAttribs)

@contextmanager
def glImmediate(mode=None):
    gl.glBegin(mode)
    try:
        yield
    finally:
        gl.glEnd()
glBlock = glImmediate

@contextmanager
def glMatrix(mode=None):
    if mode is not None:
        gl.glMatrixMode(mode)
        gl.glPushMatrix()
        try:
            yield
        finally:
            gl.glMatrixMode(mode)
            gl.glPopMatrix()
            gl.glMatrixMode(gl.GL_MODEL_VIEW)

    else:
        gl.glPushMatrix()
        try:
            yield
        finally:
            gl.glPopMatrix()

def glPurgeMatrix(mode=None):
    if mode is None:
        _glPurgeStackOf(gl.glPopMatrix)
        gl.glLoadIdentity()
    else:
        gl.glMatrixMode(mode)
        _glPurgeStackOf(gl.glPopMatrix)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODEL_VIEW)

