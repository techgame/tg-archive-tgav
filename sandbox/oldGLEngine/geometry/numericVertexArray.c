/**~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * ~ License 
 * ~ 
 * - The RuneBlade Foundation library is intended to ease some 
 * - aspects of writing intricate Jabber, XML, and User Interface (wxPython, etc.) 
 * - applications, while providing the flexibility to modularly change the 
 * - architecture. Enjoy.
 * ~ 
 * ~ Copyright (C) 2002  TechGame Networks, LLC.
 * ~ 
 * ~ This library is free software; you can redistribute it and/or
 * ~ modify it under the terms of the BSD style License as found in the 
 * ~ LICENSE file included with this distribution.
 * ~ 
 * ~ TechGame Networks, LLC can be reached at:
 * ~ 3578 E. Hartsel Drive #211
 * ~ Colorado Springs, Colorado, USA, 80920
 * ~
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 *  Derived from work done by:
 *          Tom Schwaller     <tom.schwaller@linux-magazin.de>
 *          Jim Hugunin       <hugunin@python.org>
 *          David Ascher      <da@skivs.ski.org>
 *          Michel Sanner     <sanner@scripps.edu>
 * 
***/

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~ Includes                                          */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include "Python.h"
#include "arrayobject.h"

#ifdef MS_WIN32
#include <windows.h>
#endif
#include <math.h>
#include <GL/gl.h>
#include <GL/glu.h>

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~ Constants and Variables                           */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

static GLenum GLTypeFromNumericType[] = { 
    GL_UNSIGNED_BYTE,   /* 0 -- Numerc.Character */
    GL_UNSIGNED_BYTE,   /* 1 -- Numerc.UnsignedInt8 */
    GL_BYTE,            /* 2 -- Numerc.Int8 */ 
    GL_SHORT,           /* 3 -- Numerc.Int16 */
    GL_UNSIGNED_SHORT,  /* 4 -- Numerc.UnsignedInt16 */
    GL_INT,             /* 5 -- Numerc.Int32  */
    GL_UNSIGNED_INT,    /* 6 -- Numerc.UnsignedInt32 */
    GL_FALSE,           /* 7 -- unknown */
    GL_FLOAT,           /* 8 -- Numerc.Float32 */
    GL_DOUBLE,          /* 9 -- Numerc.Float64 */
    GL_FALSE,           /* 10 -- Numerc.Complex32 */
    GL_FALSE,           /* 11 -- Numerc.Complex64 */

    GL_FALSE,           /* 12 -- unknown */
    GL_FALSE,           /* 13 -- unknown */
    GL_FALSE,           /* 14 -- unknown */
    GL_FALSE,           /* 15 -- unknown */
    };

static PyObject *nvaError = NULL;
static PyObject *nvaGLError = NULL;

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~ Macros                                            */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#define ASSURE(E,M) if(!(E)) return ErrorReturn(&nvaError, M)
#define GLASSURE(E,M) if(!(E)) return ErrorReturn(&nvaGLError, M)

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/*~ Definitions                                       */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

static PyObject *ErrorReturn(PyObject** Error, const char *message)
{
    PyErr_SetString(*Error, message);
    return NULL;
}

static PyObject *nvaVertexArray(PyObject * self, PyObject * args)
{
    GLint size = 0;
    GLenum type = GL_FLOAT;
    GLenum _glerr = GL_NO_ERROR;
    PyObject *objptr = NULL;
    PyArrayObject *arrayptr = NULL;

    if (!PyArg_ParseTuple(args, "O", &objptr)) return NULL;

    if (objptr == Py_None) {
        glVertexPointer(0, 0, 0, NULL);
    } else {
        ASSURE(PyArray_Check(objptr), "NumericVertexArray assumes a Numeric array object");
        arrayptr = (PyArrayObject*) objptr;
        ASSURE(PyArray_ISCONTIGUOUS(arrayptr), "NumericVertexArray assumes contigious Numeric arrays");
        type = GLTypeFromNumericType[arrayptr->descr->type_num];
        ASSURE(type!=GL_FALSE, "Numeric type incompatible with OpenGL types");
        ASSURE(arrayptr->nd > 0, "Numeric scalars are not supported by NumericVertexArray"); 
        size = arrayptr->dimensions[arrayptr->nd-1];
        ASSURE((2 <= size) && (size <= 4), "Invalid size for last dimension of Array.  Expected dimension of 2, 3, or 4.");

        glVertexPointer(size, type, 0, arrayptr->data);
    }

    GLASSURE((_glerr = glGetError()) == GL_NO_ERROR, gluErrorString(_glerr));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *nvaColorArray(PyObject * self, PyObject * args)
{
    GLint size = 0;
    GLenum type = GL_FLOAT;
    GLenum _glerr = GL_NO_ERROR;
    PyObject *objptr = NULL;
    PyArrayObject *arrayptr = NULL;

    if (!PyArg_ParseTuple(args, "O", &objptr)) return NULL;

    if (objptr == Py_None) {
        glColorPointer(0, 0, 0, NULL);
    } else {
        ASSURE(PyArray_Check(objptr), "NumericVertexArray assumes a Numeric array object");
        arrayptr = (PyArrayObject*) objptr;
        ASSURE(PyArray_ISCONTIGUOUS(arrayptr), "NumericVertexArray assumes contigious Numeric arrays");
        type = GLTypeFromNumericType[arrayptr->descr->type_num];
        ASSURE(type!=GL_FALSE, "Numeric type incompatible with OpenGL types");
        ASSURE(arrayptr->nd > 0, "Numeric scalars are not supported by NumericVertexArray"); 
        size = arrayptr->dimensions[arrayptr->nd-1];
        ASSURE((3 <= size) && (size <= 4), "Invalid size for last dimension of Array.  Expected dimension of 3 or 4.");

        glColorPointer(size, type, 0, arrayptr->data);
    }

    GLASSURE((_glerr = glGetError()) == GL_NO_ERROR, gluErrorString(_glerr));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *nvaNormalArray(PyObject * self, PyObject * args)
{
    GLint size = 0;
    GLenum type = GL_FLOAT;
    GLenum _glerr = GL_NO_ERROR;
    PyObject *objptr = NULL;
    PyArrayObject *arrayptr = NULL;

    if (!PyArg_ParseTuple(args, "O", &objptr)) return NULL;

    if (objptr == Py_None) {
        glNormalPointer(0, 0, NULL);
    } else {
        ASSURE(PyArray_Check(objptr), "NumericVertexArray assumes a Numeric array object");
        arrayptr = (PyArrayObject*) objptr;
        ASSURE(PyArray_ISCONTIGUOUS(arrayptr), "NumericVertexArray assumes contigious Numeric arrays");
        type = GLTypeFromNumericType[arrayptr->descr->type_num];
        ASSURE(type!=GL_FALSE, "Numeric type incompatible with OpenGL types");
        ASSURE(arrayptr->nd > 0, "Numeric scalars are not supported by NumericVertexArray"); 
        size = arrayptr->dimensions[arrayptr->nd-1];
        ASSURE(3 == size, "Invalid size for last dimension of Array.  Expected dimension of 3.");

        glNormalPointer(type, 0, arrayptr->data);
    }

    GLASSURE((_glerr = glGetError()) == GL_NO_ERROR, gluErrorString(_glerr));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *nvaTexCoordArray(PyObject * self, PyObject * args)
{
    GLint size = 0;
    GLenum type = GL_FLOAT;
    GLenum _glerr = GL_NO_ERROR;
    PyObject *objptr = NULL;
    PyArrayObject *arrayptr = NULL;

    if (!PyArg_ParseTuple(args, "O", &objptr)) return NULL;

    if (objptr == Py_None) {
        glTexCoordPointer(0, 0, 0, NULL);
    } else {
        ASSURE(PyArray_Check(objptr), "NumericVertexArray assumes a Numeric array object");
        arrayptr = (PyArrayObject*) objptr;
        ASSURE(PyArray_ISCONTIGUOUS(arrayptr), "NumericVertexArray assumes contigious Numeric arrays");
        type = GLTypeFromNumericType[arrayptr->descr->type_num];
        ASSURE(type!=GL_FALSE, "Numeric type incompatible with OpenGL types");
        ASSURE(arrayptr->nd > 0, "Numeric scalars are not supported by NumericVertexArray"); 
        size = arrayptr->dimensions[arrayptr->nd-1];
        ASSURE((1 <= size) && (size <= 4), "Invalid size for last dimension of Array.  Expected dimension of 1, 2, 3, or 4.");

        glTexCoordPointer(size, type, 0, arrayptr->data);
    }

    GLASSURE((_glerr = glGetError()) == GL_NO_ERROR, gluErrorString(_glerr));

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *nvaDrawElementsArray(PyObject * self, PyObject * args)
{
    GLenum mode = GL_POINTS;
    GLint count = 0;
    GLenum type = GL_UNSIGNED_INT;
    GLenum _glerr = GL_NO_ERROR;
    PyObject *objptr = NULL;
    PyArrayObject *arrayptr = NULL;

    if (!PyArg_ParseTuple(args, "iO", &mode, &objptr)) return NULL;

    ASSURE(PyArray_Check(objptr), "NumericVertexArray assumes a Numeric array object");
    arrayptr = (PyArrayObject*) objptr;
    ASSURE(PyArray_ISCONTIGUOUS(arrayptr), "NumericVertexArray assumes contigious Numeric arrays");
    type = GLTypeFromNumericType[arrayptr->descr->type_num];
    ASSURE(type!=GL_FALSE, "Numeric type incompatible with OpenGL types");
    count = PyArray_SIZE(arrayptr);

    glDrawElements(mode, count, type, arrayptr->data);

    GLASSURE((_glerr = glGetError()) == GL_NO_ERROR, gluErrorString(_glerr));

    Py_INCREF(Py_None);
    return Py_None;
}

static const char* nvaversion = "0.1.0";
static const char* nvadocstring = "NumericVertexArray bridge between Numeric and PyOpenGL\n\n" \
    "PyOpenGL is a wonderful extension to python, allowing nearly direct interfacing with OpenGL in a cross-platform way.\n" \
    "However, myself and others have encountered a few quirks with VertexArrays and Numeric support, leading to copies (gasp!)\n" \
    "of large memory blocks.  As you can imagine (or may have experienced) this behavior rapidly degrades performance of\n" \
    "python applications using large vertex arrays.  Many have tried to correct the problem in the library, myself included,\n" \
    "yet it remains unsolved.  For me, it was far to difficult to try and patch the existing system in such a way as to insure\n" \
    "existing code would function, while expunging the data copying code.\n" \
    "\n" \
    "So, the approach taken with NumericVertexArray is to reduce the complexity.  Therefore, Numeric extensions are *required*,\n" \
    "arrays are assumed to be contigious, and data is *never* copied.  The philiosophy is that if a copy is required, an exception\n" \
    "should be raised instead.  Hopefully, this extension will assist you in your pursuits.  And if not, use the source and make\n" \
    "something new; or send me a patch.  ;)\n" \
    "\nEnjoy!\n";

static PyMethodDef nvamethods[] = {
    {"VertexArray", nvaVertexArray, METH_VARARGS, "glVertexArray(data) -> None\n\nCalls glVertexPointer(size, type, stride, pointer) where:\n    size = data.shape[-1]\n    type = GLtype corresponding to the Numeric code\n    stride = 0\n    pointer = Numeric array's data pointer\n\nNOTE: Be very careful not to pull the rug out from under the module by changing or deleting the array."},
    {"ColorArray", nvaColorArray, METH_VARARGS, "glColorArray(data) -> None\n\nCalls glColorPointer(size, type, stride, pointer) where:\n    size = data.shape[-1]\n    type = GLtype corresponding to the Numeric code\n    stride = 0\n    pointer = Numeric array's data pointer\n\nNOTE: Be very careful not to pull the rug out from under the module by changing or deleting the array."},
    {"NormalArray", nvaNormalArray, METH_VARARGS, "glNormalArray(data) -> None\n\nCalls glNormalPointer(size, type, stride, pointer) where:\n    size = data.shape[-1]\n    type = GLtype corresponding to the Numeric code\n    stride = 0\n    pointer = Numeric array's data pointer\n\nNOTE: Be very careful not to pull the rug out from under the module by changing or deleting the array."},
    {"TexCoordArray", nvaTexCoordArray, METH_VARARGS, "glTexCoordArray(data) -> None\n\nCalls glTexCoordPointer(size, type, stride, pointer) where:\n    size = data.shape[-1]\n    type = GLtype corresponding to the Numeric code\n    stride = 0\n    pointer = Numeric array's data pointer\n\nNOTE: Be very careful not to pull the rug out from under the module by changing or deleting the array."},
    {"DrawElementsArray", nvaDrawElementsArray, METH_VARARGS, "DrawElementsArray(mode, data) -> None\n\nCalls glDrawElements(mode, size, type, pointer) where:\n    size = total number of elements in data\n    type = GLtype corresponding to the Numeric code\n    pointer = Numeric array's data pointer."},
    {NULL, NULL, 0, NULL}
};

#ifdef WIN32
__declspec(dllexport)
#endif
void initNumericVertexArray(void)
{
    PyObject *module = NULL;
    PyObject *dict = NULL;
    PyObject *__doc__ = NULL;
    PyObject *__version__ = NULL;

    /* Initialize our new module */
    module = Py_InitModule("NumericVertexArray", nvamethods);

    /* Initialize Numeric, because we're going to need it */
    import_array();

    /* Need to set some variables in the module */
    dict = PyModule_GetDict(module);

    /* Set error to empty, and allow access from python */
    nvaError = PyErr_NewException("NumericVertexArray.UsageError", PyExc_EnvironmentError, NULL);
	nvaGLError = PyErr_NewException("NumericVertexArray.GLerror", PyExc_EnvironmentError, NULL);

    PyDict_SetItemString(dict, "UsageError", nvaError);
    PyDict_SetItemString(dict, "GLerror", nvaGLError);

    /* Set the docstring */
    __doc__ = Py_BuildValue("s", nvadocstring);
    PyDict_SetItemString(dict, "__doc__", __doc__);

    /* Set the version */
    __version__ = Py_BuildValue("s", nvaversion);
    PyDict_SetItemString(dict, "__version__", __version__);

    if (PyErr_Occurred())
       Py_FatalError("can not initialize module NumericVertexArray");
}

