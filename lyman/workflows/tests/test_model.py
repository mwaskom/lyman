import os.path as op
import shutil
from tempfile import mkdtemp
import numpy as np
from nipype.testing import assert_equal, assert_raises
from nipype.interfaces.base import Bunch

from .. import model
