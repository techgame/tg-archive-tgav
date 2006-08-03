#!/usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import re

from TG.common.path import path

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class OpenGLSpecPartParser(re.Scanner):
    lexicon = []
    def match(pattern, lexicon=lexicon ):
        def addPattern(func):
            lexicon.append((pattern, func))
        return addPattern

    def __init__(self):
        re.Scanner.__init__(self, self.lexicon, re.MULTILINE)

    def __call__(self, spec):
        return self.parse(spec)

    def parse(self, spec):
        self._parts = []
        self._title = ''
        self._block = []

        spec = self.scan(spec)[1]
        if spec:
            print "Error at:", repr(spec[:80])
            assert False
        result = self._parts
        del self._title
        del self._block
        del self._parts
        return result

    @match('^(\S.*)$\n')
    def _onParseTitle(self, groups):
        if self._block:
            self._parts.append((self._title, self._block))
        else:
            groups = self._title.rstrip() + ' ' + groups.lstrip()

        self._title = groups.strip()
        self._block = []

    @match(r'^([ \t]+.*)$\n?')
    def _onParseContent(self, groups):
        self._block.append(groups)

    @match(r'^([ \t]*)\n')
    def _onParseBlank(self, groups):
        self._block.append(groups)

    del match
parser = OpenGLSpecPartParser()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class OpenGLRegistryToCParts(object):
    partMapping = {
        'Name': 'name',
        'Name String': 'strings',
        'Name Strings': 'strings',
        'New Procedures and Functions': 'functions', 
        'New Tokens': 'tokens',
    }
    partFnMap = {}
    def token(key, partFnMap=partFnMap):
        def addMatch(fn):
            partFnMap[key] = fn
        return addMatch

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, vendor, filename, specFile):
        self.vendor = vendor
        self.filename = filename
        self.parseSpec(specFile.read())

    def parseSpec(self, spec):
        self.names = None
        self.strings = None
        self.tokens = None
        self.functions = None
        self.invalid = False

        self._parts = {}
        for title, block in parser.parse(spec):
            mapTitle = self.partMapping.get(title)
            fn = self.partFnMap.get(mapTitle)
            if fn:
                fn(self, title, block)

        self._assert()

    def _assert(self):
        assert self.names is not None
        assert self.strings is not None
        #if self.invalid:
        #    print "Invalid:", self.filename

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @token('name')
    def _parseNameSection(self, title, block):
        self.names = self._blockToIdentifiers(block)

    @token('strings')
    def _parseStringsSection(self, title, block):
        self.strings = self._blockToIdentifiers(block)

    def _blockToIdentifiers(self, block):
        result = (b.strip() for b in block)
        result = list(b for b in result if b and ' ' not in b)
        return result

    re_token_line = re.compile(r'^\s*([A-Z][A-Z0-9_]+)\s+(\S+)\s$')
    re_token_name = re.compile(r'[AWP]?GL[XU]?_\w+')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @token('tokens')
    def _parseTokenSection(self, title, block):
        self.tokens = []
        matcher = self.re_token_line.match
        for line in block:
            match = matcher(line)
            if match is not None:
                self._addTokenForMatch(match)

    def _addTokenForMatch(self, match):
        name, value = match.groups()
        try: 
            eval(value)
            valid = True
        except (NameError, ValueError, SyntaxError),e:
            valid = False
            self.invalid = True

        if not self.re_token_name.match(name):
            name = 'GL_'+name
        self.tokens.append((name, value, valid))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _iterBlocksByGroup(self, block):
        lines = []
        for l in block:
            if l.strip():
                lines.append(l)
            elif lines:
                yield lines
                lines = []
        if lines:
            yield lines


    glTypeMap = dict(
        boolean="GLboolean",
        Boolean="GLboolean",
        GLboolean="GLboolean",

        DMbuffer="GLvoid*",

        void='GLvoid',
        GLvoid="GLvoid",

        enum="GLenum",
        GLenum="GLenum",
        Glenum="GLenum",

        bitfield="GLbitfield",
        GLbitfield="GLbitfield",

        byte="GLbyte",
        GLbyte="GLbyte",
        ubyte="GLubyte",
        GLubyte="GLubyte",

        clampf="GLclampf",
        GLclampf="GLclampf",

        clampd="GLclampd",
        GLclampd="GLclampd",

        double="GLdouble",
        GLdouble="GLdouble",

        float="GLfloat",
        GLfloat="GLfloat",

        half="GLhalf",
        GLhalf="GLhalf",

        int="GLint",
        GLint="GLint",

        uint="GLint",
        GLuint="GLint",

        short="GLshort",
        GLshort="GLshort",

        ushort="GLushort",
        GLushort="GLushort",

        sizei="GLsizei",
        GLsizei="GLsizei",
        )

    re_func = re.compile(
            r'^\s*(\w+[^()\n\r\f]*)'
            r'\(([^)]*)\);?'
            , re.MULTILINE)

    def _checkBlockStartsWithNone(self, block):
        for line in block[:10]:
            line = line.strip().lower()
            if line.startswith('none'):
                return True
            elif line: 
                return False
        return False

    def _iterFuncsByGroup(self, block):
        if self._checkBlockStartsWithNone(block):
            return

        n = 0
        for lines in self._iterBlocksByGroup(block):
            joined = ''.join(lines).strip()

            for fn in self.re_func.finditer(joined):
                pre, args = fn.groups()
                pre, _, name = pre.strip().rpartition(' ')

                if not name: 
                    # pre is actually our name, so swap
                    name, pre = pre, name
                if name in self.glTypeMap:
                    # or we have a function type declaration
                    continue

                pre = pre.replace('*', ' * ').split(' ')
                args = [a.strip() for a in args.split(',')]

                if '[' in name or '{' in name:
                    for name in self._expandComplexName(name, args):
                        yield name, (pre, name, args)
                else:
                    yield name, (pre, name, args)

    def _expandComplexName(self, name, args):
        print name, args
        return []

    @token('functions')
    def _parseFunctionsSection(self, title, block):
        self.functions = []
        for fn in self._iterFuncsByGroup(block):
            self.functions.append(fn)
            #print fn[0]

    del token

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    if 0:
        specials = map(path, [
            './specs/EXT/framebuffer_object.txt',
            #'./specs/ARB/vertex_program.txt',
            #'./specs/ATI/envmap_bumpmap.txt',
            #'./specs/ATI/text_fragment_shader.txt',
            ])
        for f in specials:
            vendor = f.dirname().name
            #print
            #print vendor, f
            OpenGLRegistryToCParts(vendor, str(f), f.open('rb'))
    else:

        for d in path('./specs').dirs():
            vendor = d.name
            for f in d.files('*.txt'):
                #print
                #print vendor, f
                OpenGLRegistryToCParts(vendor, str(f), f.open('rb'))


if __name__=='__main__':
    main()

