import os.path as op
import os
import subprocess

__version__ = ''


try:
    # if we are running in a git repo, look up the hash
    __version__ = subprocess.Popen(
        ('git','-C',os.path.dirname(__file__),'describe','--always','--dirty','--broken'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].strip()
    assert __version__
except:
    # otherwise check for a version file
    try:
        from . version import version as __version__
    except:
        pass

__all__ = ['template_fit']

from template_fit import TemplateFit