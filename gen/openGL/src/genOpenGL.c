#define GL_GLEXT_LEGACY
#define GL_GLEXT_PROTOTYPES
#include <OpenGL/gl.h>
#include <OpenGL/glext.h>
#include <OpenGL/glu.h>

// Check for X Windows
#if defined(X_PROTOCOL)
#include <OpenGL/glxext.h>
#endif

// Check for Microsoft Windows
#if defined(_WIN32) 
#include <OpenGL/wglext.h>
#endif

