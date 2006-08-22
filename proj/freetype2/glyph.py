class FaceGlyph(object):
    #~ FreeType API interation ~~~~~~~~~~~~~~~~~~~~~~~~~~
    _as_parameter_ = None
    _as_parameter_type_ = FT.FT_GlyphSlot

    def __init__(self, glyph, face):
        self._as_parameter_ = glyph
        self._face = face

    @property
    def _glyph(self):
        return self._as_parameter_

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def metrics(self):
        return self._glyph.metrics # FT_Glyph_Metrics
    @property
    def linearHoriAdvance(self):
        return self._glyph.linearHoriAdvance # FT_Fixed
    @property
    def linearVertAdvance(self):
        return self._glyph.linearVertAdvance # FT_Fixed
    @property
    def advance(self):
        adv = self._glyph.advance # FT_Vector
        return (adv.x, adv.y)
    @property
    def format(self):
        return self._glyph.format # FT_Glyph_Format
    @property
    def bitmap(self):
        return self._glyph.bitmap # FT_Bitmap
    @property
    def bitmapLeft(self):
        return self._glyph.bitmap_left # FT_Int
    @property
    def bitmapTop(self):
        return self._glyph.bitmap_top # FT_Int
    @property
    def outline(self):
        return self._glyph.outline # FT_Outline
    @property
    def numSubglyphs(self):
        return self._glyph.num_subglyphs # FT_UInt
    @property
    def subglyphs(self):
        return self._glyph.subglyphs # FT_SubGlyph
    @property
    def controlData(self):
        return self._glyph.control_data # c_void_p
    @property
    def controlLen(self):
        return self._glyph.control_len # c_long
    @property
    def lsbDelta(self):
        return self._glyph.lsb_delta # FT_Pos
    @property
    def rsbDelta(self):
        return self._glyph.rsb_delta # FT_Pos

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _renderMode = None
    def getRenderMode(self):
        if self._renderMode is None:
            return self._face._renderMode
        return self._renderMode
    def setRenderMode(self, renderMode):
        self._renderMode = renderMode
    renderMode = property(getRenderMode, setRenderMode)

    _ft_renderGlyph = FT_Render_Glyph
    def render(self, renderMode=None):
        if renderMode is None: 
            renderMode = self.renderMode
        self._ft_renderGlyph(renderMode)
