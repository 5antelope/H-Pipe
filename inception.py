import numpy as np
import os
import sys

caffe2_root = '/home/15-418/caffe2/'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

from caffe2.proto import caffe2_pb2
from caffe2.python import core

print("CAFFE2 IMPORT TEST PASS")
