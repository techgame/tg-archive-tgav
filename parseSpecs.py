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
        if self.invalid:
            print "Invalid:", self.filename

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

    @token('functions')
    def _parseFunctionsSection(self, title, block):
        #funcs = (b.strip() for b in block if b.strip())
        #funcs = ' '.join(f for f in funcs if f)
        #funcs = [f.strip()+';' for f in funcs.split(';') if f and not f.lower().startswith('none')]

        #self.functions = funcs
        #for f in funcs:
        #    print f
        return
        print
        print '##', self.filename, self.names
        for b in block:
            print b,
        print '## end'

    re_token_line = re.compile('^\s*([A-Z][A-Z0-9_]+)\s+(\S+)\s$')
    re_token_name = re.compile('[AWP]?GL[XU]?_\w+')

    @token('tokens')
    def _parseTokenSection(self, title, block):
        self.tokens = []
        matcher = self.re_token_line.match
        tokenName = self.re_token_name.match
        for b in block:
            m = matcher(b)
            if m is not None:
                content = b[m.start(1):m.end(1)] + b[m.start(2):m.end(2)]
                leftover = b[:m.start(1)] + b[m.end(1):m.start(2)] + b[m.end(2):]
                assert len(content+leftover) == len(b), (m.groups(), b)

                name, value = m.groups()

                try: 
                    eval(value)
                    valid = True
                except (NameError, ValueError, SyntaxError),e:
                    valid = False
                    self.invalid = True

                if not tokenName(name):
                    name = 'GL_'+name
                self.tokens.append((name, value, valid))

    del token

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
    #f = path('./specs/ARB/vertex_program.txt')
    #OpenGLRegistryToCParts("ARB", str(f), f.open('rb'))

    for d in path('./specs').dirs():
        for f in d.files('*.txt'):
            OpenGLRegistryToCParts(d.name, str(f), f.open('rb'))

if __name__=='__main__':
    main()

