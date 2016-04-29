import numpy as np
import os
import sys

caffe2_root = '/home/15-418/caffe2/'
sys.path.insert(0, os.path.join(caffe2_root, 'gen'))

from caffe2.proto import caffe2_pb2
from caffe2.python import core

print("CAFFE2 IMPORT TEST PASS")

# net is the network definition.
net = caffe2_pb2.NetDef()
net.ParseFromString(open('inception_net.pb').read())

# tensors contain the parameter tensors.
tensors = caffe2_pb2.TensorProtos()
tensors.ParseFromString(open('inception_tensors.pb').read())

print("LOAD INCEPTION DBS")

def ClassifyImageWithInception(image_file, show_image=True, output_name="softmax2"):
    # TODO:
    # 1. read image_file
    # 2. crop image
    # 3. resize the image to 224 * 224
    # 4. normalize the image and feed it into the network. The network expects
    # a four-dimensional tensor, since it can process images in batches. In our
    # case, we will basically make the image as a batch of size one.
    pass

# We will also load the synsets file where we can look up the actual words for each of our prediction.
synsets = [l.strip() for l in open('synsets.txt').readlines()]

predictions = ClassifyImageWithInception("dog.jpg").flatten()
idx = np.argmax(predictions)
print "Prediction: %d, synset %s" % (idx, synsets[idx])

print "DONE"
