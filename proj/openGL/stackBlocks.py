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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _glPurgeStackMethod(method):
    try:
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
    _glPurgeStackMethod(gl.glPopName)

@contextmanager
def glClientAttribs(mask):
    gl.glPushClientAttrib(mask)
    try:
        yield
    finally:
        gl.glPopClientAttrib()

def glPurgeClientAttribs():
    _glPurgeStackMethod(gl.glPopClientAttribs)

@contextmanager
def glAttribs(mask):
    gl.glPushAttrib(mask)
    try:
        yield
    finally:
        gl.glPopAttrib()

def glPurgeAttribs():
    _glPurgeStackMethod(gl.glPopAttribs)

@contextmanager
def glMatrixMode(mode):
    gl.glMatrixMode(mode)
    try:
        yield
    finally:
        gl.glMatrixMode(gl.GL_MODEL_VIEW)

@contextmanager
def glBlock(mode=None):
    gl.glBegin(mode)
    try:
        yield
    finally:
        gl.glEnd()

@contextmanager
def glMatrix(mode=None):
    if mode is not None:
        gl.glMatrixMode(mode)
        try:
            gl.glPushMatrix()
        finally:
            gl.glMatrixMode(gl.GL_MODEL_VIEW)

        try:
            yield
        finally:

            gl.glMatrixMode(mode)
            try:
                gl.glPopMatrix()
            finally:
                gl.glMatrixMode(gl.GL_MODEL_VIEW)

    else:
        gl.glPushMatrix()
        try:
            yield
        finally:
            gl.glPopMatrix()

def glPurgeMatrix(mode=None):
    if mode is not None:
        gl.glMatrixMode(mode)

    _glPurgeStackMethod(gl.glPopMatrix)
    gl.glLoadIdentity()

    if mode is not None:
        gl.glMatrixMode(gl.GL_MODEL_VIEW)

