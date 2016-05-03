CXX ?= g++
GXX ?= g++

CXXFLAGS += -std=c++11 -g

INC += -I/home/15-418/protobuf-2.6.1/include

LDFLAGS += -L/home/15-418/protobuf-2.6.1/lib -lprotobuf
LDFLAGS += -L/home/15-418/opencv/build/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/lib -lglog -lleveldb
LDFLAGS += -L/usr/local/cuda/lib64/ -lcudart
LDFLAGS += -L/home/yangwu/lib -lsnappy
LDFLAGS += -L/home/yangwu/lmdb/libraries/liblmdb -llmdb
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/lib -lboost_system -lm
LDFLAGS += -L/afs/cs/academic/class/15418-s13/public/boost_1_53_0/bin.v2/libs/thread/build/gcc-4.8.0/release/link-static/threading-multi -lboost_thread

all: test_io

data.pb.h data.pb.cc: ../data/caffe2.proto
	protoc caffe2.proto --cpp_out=./

test_io: test_io.cpp caffe2.pb.h caffe2.pb.cc
	$(CXX) $(CXXFLAGS) $(INC) caffe2.pb.cc test_io.cpp -o test_io.out $(LDFLAGS)

clean:
	rm -f io.o test_io.out caffe2.pb.h caffe2.pb.cc
