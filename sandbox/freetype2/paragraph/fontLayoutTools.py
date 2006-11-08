#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy

import textwrap
import re
from TG.openGL.font import Font
from TG.openGL.textLayout import TextWrapper, TextObject

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bigSampleText = '''\
Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Pellentesque quis lacus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Curabitur facilisis, ante at adipiscing ullamcorper, libero dolor rutrum felis, nec vehicula turpis diam id lacus. Quisque tincidunt tempus orci. Quisque sit amet sem. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Ut hendrerit, tortor quis laoreet feugiat, quam lectus suscipit orci, ornare tristique risus dui fermentum massa. Quisque scelerisque ullamcorper libero. Sed adipiscing sapien eget enim porttitor volutpat. Proin porttitor. Vivamus semper lectus mattis pede. Mauris pretium odio sit amet enim.

Nunc at mauris eleifend leo sollicitudin aliquam. Vivamus fermentum ipsum. Integer augue nulla, semper sit amet, viverra in, tempor eget, sapien. Phasellus id ipsum. Maecenas pharetra, risus a imperdiet vestibulum, velit ligula porttitor orci, sit amet pharetra diam erat vitae dolor. Quisque bibendum. Suspendisse dictum. Vivamus at risus. Nam nonummy mauris in tortor. Nunc nisl ante, placerat a, consequat et, mattis vel, magna.

Etiam mauris turpis, pretium et, tristique vel, interdum vitae, nulla. Nunc nisl augue, lacinia ac, iaculis nec, tempor vitae, nisl. Morbi lacinia scelerisque nulla. Donec quam orci, consectetuer at, eleifend sed, tincidunt vel, justo. Suspendisse nec augue. Duis vel est ut quam ultricies placerat. Proin dui massa, faucibus ut, imperdiet ut, volutpat a, augue. Vivamus quis ipsum. Aliquam nonummy risus non lectus semper feugiat. Donec id enim eu mi semper varius. Nunc neque neque, semper et, adipiscing in, consectetuer in, orci. Nunc in quam. Aliquam ultrices commodo lacus. Etiam semper. Aliquam quis pede. Maecenas id mauris. Vestibulum ut nunc. Quisque accumsan malesuada dui.

Nullam nulla mi, consequat id, posuere quis, mattis eu, quam. Integer blandit. Praesent pharetra nunc eget eros. Etiam volutpat quam nec pede. Nulla leo est, sagittis ut, euismod quis, ultrices nec, justo. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Nullam euismod. Pellentesque nisi ligula, consectetuer ac, aliquam varius, laoreet eget, erat. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos hymenaeos.

Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Cras mauris eros, pulvinar non, vulputate mattis, vehicula fermentum, sapien. Sed ut leo. Cras vitae turpis. Sed risus nisi, sollicitudin interdum, posuere id, egestas eu, augue. Phasellus pulvinar. Nullam varius tortor eget urna. Nam lorem. Nam diam. Sed velit enim, adipiscing id, volutpat vel, dignissim at, magna. Nullam sodales. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos hymenaeos. Maecenas vel nunc non leo mollis tincidunt. Etiam venenatis placerat dui. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos hymenaeos. Morbi vulputate vestibulum tellus.

Maecenas condimentum scelerisque odio. Morbi accumsan. Sed vel nulla. Quisque commodo erat in metus. Nam id tellus ac magna vehicula consectetuer. In at nunc ut leo nonummy luctus. Vivamus luctus. Phasellus quis risus hendrerit lectus pellentesque mollis. Morbi id felis iaculis tellus pretium dapibus. Aliquam nibh. Quisque posuere velit ac elit. Vivamus suscipit nisi vel lectus. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. In tempor. Pellentesque quis mauris rhoncus metus dictum faucibus. Fusce id enim.

Etiam ultricies augue non metus. Maecenas laoreet. Donec tincidunt tortor quis elit. Morbi sit amet tellus et pede imperdiet elementum. Maecenas ut magna in dolor euismod faucibus. Vivamus tempus placerat metus. In hac habitasse platea dictumst. Nulla felis metus, viverra vel, pretium at, luctus et, massa. Praesent vel ipsum. Sed id nulla ut purus euismod elementum. Nullam feugiat sollicitudin elit. Aenean ut elit venenatis augue aliquet suscipit. Quisque sit amet mi posuere nisl imperdiet cursus. Aenean porttitor ligula vel ante. Ut mi erat, varius at, pellentesque consequat, pretium a, nisl. Praesent dictum risus vel sem. Duis accumsan ullamcorper risus. Fusce porttitor pretium sapien. Etiam tincidunt, dolor et commodo dapibus, mauris turpis aliquam pede, vitae pharetra ligula tortor id dui. Sed non tellus ac elit tempus rutrum.

Ut eu sapien. Phasellus odio mi, consectetuer non, elementum in, feugiat eu, pede. Aenean luctus, nunc vitae interdum vulputate, lectus lacus bibendum eros, quis dignissim felis eros vitae sem. Maecenas diam odio, mollis vitae, rhoncus ac, congue non, nisl. Mauris hendrerit nulla sit amet odio. In sed leo. Phasellus mattis. Aliquam erat volutpat. Etiam eu arcu eu erat lobortis fermentum. Pellentesque iaculis magna sit amet felis.

Donec vulputate enim adipiscing ligula. Nullam semper neque at lacus. Donec feugiat vulputate orci. Vivamus id sapien. Nunc non odio. Vivamus sodales, ipsum a tristique malesuada, sapien lorem pretium orci, at sollicitudin dolor magna vitae ligula. Ut id nulla. Phasellus lacus felis, imperdiet nec, varius sit amet, tristique vestibulum, odio. In volutpat. Sed leo massa, iaculis at, hendrerit sed, rhoncus sit amet, nulla. Sed vitae lorem. Curabitur massa lectus, ultricies non, tempor eget, mattis et, ipsum. Nulla vulputate felis id orci. Donec placerat vulputate dolor. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos hymenaeos. Sed aliquet, felis a volutpat tempus, leo sapien eleifend velit, quis interdum dui nunc at risus. Donec sit amet orci. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Pellentesque bibendum iaculis tellus. Etiam tempor nibh.

Nam felis lorem, consequat nec, tincidunt at, malesuada molestie, magna. Nulla facilisi. Quisque egestas justo at nisi. Suspendisse a sapien. Nunc eget sem in lorem cursus accumsan. Curabitur at dolor at justo facilisis sagittis. In a mauris. Mauris leo. Vestibulum dictum dapibus lacus. Phasellus sed est. Cras sit amet sapien. Quisque massa eros, malesuada ac, ultricies nec, fringilla at, lorem. Pellentesque lectus diam, nonummy in, adipiscing ut, lacinia eu, purus. Lorem ipsum dolor sit amet, consectetuer adipiscing elit.
'''

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    # disable loading of texture
    Font.FontTexture = None

    arial = Font.fromFilename('/Library/Fonts/Zapfino.dfont', 25)
    ArialTextObj = TextObject.factoryFor(arial)

    tobj = ArialTextObj(bigSampleText)
    for l in tobj.wrap(1000):
        print l.replace('\n', '')#encode('string_escape')

